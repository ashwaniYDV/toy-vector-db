#include "toyvdb/engine.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "toyvdb/bitset.hpp"
#include "toyvdb/flat_index.hpp"
#include "toyvdb/hnsw_index.hpp"
#include "toyvdb/op_codec.hpp"

namespace toyvdb {

namespace {
constexpr const char* kSnapshotKey = "snapshot";
}

Engine::Engine(EngineConfig cfg) : cfg_(cfg), store_(cfg.dim) {
    switch (cfg_.index) {
        case IndexKind::Flat:
            index_ = make_flat_index(store_, cfg_.metric);
            break;
        case IndexKind::Hnsw:
            index_ = make_hnsw_index(store_, cfg_.metric, cfg_.hnsw);
            break;
    }
    if (!index_) throw std::invalid_argument("Engine: unsupported index kind");

    if (cfg_.persistence) {
        const auto& p = *cfg_.persistence;
        std::filesystem::create_directories(p.dir);
        blobs_ = std::make_unique<FileBlobStore>(p.dir / "blobs");
        wal_ = std::make_unique<FileLogStore>(p.dir / "wal.log", p.sync);
        recover();
    }
}

InternalId Engine::apply_op(const Op& op, bool log_it) {
    // Write-ahead: durably record the intent before mutating in-memory state.
    if (log_it && wal_) {
        const std::vector<std::byte> bytes = encode_op(op);
        wal_->append(bytes);  // FileLogStore fsyncs here when SyncPolicy::Always
    }

    switch (op.type) {
        case OpType::Insert: {
            // Only index a genuinely new id. Re-inserting an existing external id
            // reuses its slot (an update), so adding again would create duplicate
            // index entries and duplicate search results.
            const bool       is_new = !store_.resolve(op.ext_id).has_value();
            const InternalId id = store_.apply(op).value();  // Insert always yields an id
            if (is_new) index_->add(id, store_.get(id));
            return id;
        }
        case OpType::Update: {
            store_.apply(op);  // index reads live vectors from the store
            return kInvalidId;
        }
        case OpType::Delete: {
            // apply() returns the deleted id (or nullopt if it wasn't live).
            if (const auto id = store_.apply(op)) index_->remove(*id);
            return kInvalidId;
        }
    }
    return kInvalidId;
}

InternalId Engine::insert(std::string ext_id, std::span<const float> vec, Metadata meta) {
    Op op{OpType::Insert, std::move(ext_id), std::vector<float>(vec.begin(), vec.end()),
          std::move(meta)};
    return apply_op(op, /*log_it=*/true);
}

bool Engine::update(const std::string& ext_id, std::span<const float> vec,
                    std::optional<Metadata> meta) {
    // Only an existing (live) id can be updated. Skip otherwise so we don't
    // durably log a mutation that changes nothing.
    if (!store_.resolve(ext_id).has_value()) return false;
    Op op{OpType::Update, ext_id, std::vector<float>(vec.begin(), vec.end()), std::move(meta)};
    apply_op(op, /*log_it=*/true);
    return true;
}

bool Engine::erase(const std::string& ext_id) {
    // Nothing to delete -> no-op, and don't write a useless WAL record.
    if (!store_.resolve(ext_id).has_value()) return false;
    Op op{OpType::Delete, ext_id, {}, {}};
    apply_op(op, /*log_it=*/true);
    return true;
}

std::vector<SearchResult> Engine::search(std::span<const float> query, std::size_t k,
                                         const Filter* filter, std::size_t ef) const {
    if (query.size() != static_cast<std::size_t>(store_.dim())) {
        throw std::invalid_argument("Engine::search: query dimensionality mismatch");
    }
    if (k == 0) return {};  // nothing to return; skip building the filter set

    SearchParams params;
    if (ef > 0) params.ef = ef;

    if (filter == nullptr) {
        return index_->search(query, k, params);
    }

    Bitset allowed(store_.slot_count());
    for (InternalId id = 0; id < store_.slot_count(); ++id) {
        if (!store_.is_live(id)) continue;
        const Metadata* m = store_.metadata(id);
        if (m != nullptr && filter->matches(*m)) allowed.set(id);
    }
    params.allowed = &allowed;
    return index_->search(query, k, params);
}

std::vector<std::vector<SearchResult>> Engine::search_batch(
    const std::vector<std::vector<float>>& queries, std::size_t k, const Filter* filter,
    std::size_t ef, unsigned num_threads) const {
    const std::size_t qn = queries.size();
    std::vector<std::vector<SearchResult>> results(qn);
    if (qn == 0) return results;

    // Validate every query dimension up front, in this (calling) thread -- an
    // exception thrown inside a worker thread would terminate the process.
    const auto dim = static_cast<std::size_t>(store_.dim());
    for (const auto& q : queries) {
        if (q.size() != dim) {
            throw std::invalid_argument("Engine::search_batch: query dimensionality mismatch");
        }
    }
    if (k == 0) return results;  // every result is already an empty vector

    // Build the allowed-set once; it is shared read-only across worker threads.
    Bitset     allowed;
    const bool has_filter = (filter != nullptr);
    if (has_filter) {
        allowed.resize(store_.slot_count());
        for (InternalId id = 0; id < store_.slot_count(); ++id) {
            if (!store_.is_live(id)) continue;
            const Metadata* m = store_.metadata(id);
            if (m != nullptr && filter->matches(*m)) allowed.set(id);
        }
    }

    const auto run_range = [&](std::size_t lo, std::size_t hi) {
        SearchParams params;
        if (ef > 0) params.ef = ef;
        if (has_filter) params.allowed = &allowed;
        for (std::size_t i = lo; i < hi; ++i) {
            results[i] = index_->search(
                std::span<const float>(queries[i].data(), queries[i].size()), k, params);
        }
    };

    unsigned threads = num_threads;
    if (threads == 0) threads = std::max(1U, std::thread::hardware_concurrency());
    const std::size_t tcount = std::min(static_cast<std::size_t>(threads), qn);

    // Serial fast path (also keeps single-thread benchmarks free of spawn overhead).
    if (tcount <= 1) {
        run_range(0, qn);
        return results;
    }

    const std::size_t         chunk = (qn + tcount - 1) / tcount;
    std::vector<std::jthread> workers;
    workers.reserve(tcount);
    for (std::size_t t = 0; t < tcount; ++t) {
        const std::size_t lo = t * chunk;
        if (lo >= qn) break;
        const std::size_t hi = std::min(lo + chunk, qn);
        workers.emplace_back([&run_range, lo, hi] { run_range(lo, hi); });
    }
    // Join before returning so all writes to `results` happen-before the read.
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
    return results;
}

void Engine::flush() {
    if (wal_) wal_->sync();
}

void Engine::compact() {
    // Compaction renumbers internal ids, which the index references, so it is
    // done here (not in VectorStore): copy out the live entries, rebuild a fresh
    // store + index, and re-insert. Result: no tombstones, dense ids 0..n-1.
    struct LiveEntry {
        std::string        ext_id;
        std::vector<float> vec;
        Metadata           meta;
    };
    std::vector<LiveEntry> live;
    live.reserve(store_.size());
    for (InternalId id = 0; id < store_.slot_count(); ++id) {
        if (!store_.is_live(id)) continue;
        const auto      sp = store_.get(id);
        const Metadata* m = store_.metadata(id);
        live.push_back({store_.external_id(id), std::vector<float>(sp.begin(), sp.end()),
                        m != nullptr ? *m : Metadata{}});
    }

    index_.reset();  // drop the old index (it references store_) before replacing it
    store_ = VectorStore(cfg_.dim);
    switch (cfg_.index) {
        case IndexKind::Flat:
            index_ = make_flat_index(store_, cfg_.metric);
            break;
        case IndexKind::Hnsw:
            index_ = make_hnsw_index(store_, cfg_.metric, cfg_.hnsw);
            break;
    }
    for (auto& e : live) {
        const InternalId id = store_.insert(e.ext_id, e.vec, std::move(e.meta));
        index_->add(id, store_.get(id));
    }

    if (blobs_) snapshot();  // persist the compacted state and reset the WAL
}

void Engine::snapshot() {
    if (!blobs_) return;

    // A snapshot is just the current live state expressed as a sequence of
    // framed Insert ops -- it reuses the op codec and replays exactly like the
    // WAL. Each record is [u32 len][op bytes].
    std::vector<std::byte> out;
    for (InternalId id = 0; id < store_.slot_count(); ++id) {
        if (!store_.is_live(id)) continue;
        Op op;
        op.type = OpType::Insert;
        op.ext_id = store_.external_id(id);
        const auto sp = store_.get(id);
        op.vec.assign(sp.begin(), sp.end());
        if (const Metadata* m = store_.metadata(id)) op.meta = *m;

        const std::vector<std::byte> rec = encode_op(op);
        const auto                   len = static_cast<std::uint32_t>(rec.size());
        const auto* lp = reinterpret_cast<const std::byte*>(&len);
        out.insert(out.end(), lp, lp + 4);
        out.insert(out.end(), rec.begin(), rec.end());
    }

    blobs_->put(kSnapshotKey, out);
    if (wal_) wal_->truncate();  // post-snapshot WAL starts empty
}

void Engine::replay_blob(std::span<const std::byte> data) {
    std::size_t off = 0;
    while (off + 4 <= data.size()) {
        std::uint32_t len = 0;
        std::memcpy(&len, data.data() + off, 4);
        off += 4;
        if (off + len > data.size()) break;  // truncated snapshot record
        apply_op(decode_op(data.subspan(off, len)), /*log_it=*/false);
        off += len;
    }
}

void Engine::recover() {
    // Order matters: the snapshot is the older base state, the WAL holds the
    // newer ops applied on top. Replaying both in order rebuilds final state,
    // and rebuilds the index incrementally (each Insert re-adds to the graph).
    if (blobs_ && blobs_->exists(kSnapshotKey)) {
        const std::vector<std::byte> snap = blobs_->get(kSnapshotKey);
        replay_blob(snap);
    }
    if (wal_) {
        wal_->replay([&](std::span<const std::byte> rec) {
            apply_op(decode_op(rec), /*log_it=*/false);
        });
    }
}

}  // namespace toyvdb
