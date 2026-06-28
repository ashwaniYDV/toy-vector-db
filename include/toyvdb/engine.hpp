#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "blob_store.hpp"
#include "distance.hpp"
#include "hnsw_index.hpp"
#include "index.hpp"
#include "log_store.hpp"
#include "metadata.hpp"
#include "types.hpp"
#include "vector_store.hpp"

namespace toyvdb {

enum class IndexKind { Flat, Hnsw };

/// Optional durability. When set, the Engine logs every mutation to a
/// write-ahead log and can snapshot + recover from `dir`.
struct PersistenceConfig {
    std::filesystem::path dir;
    SyncPolicy            sync = SyncPolicy::Always;
};

struct EngineConfig {
    Dim        dim;
    MetricKind metric = MetricKind::Cosine;
    IndexKind  index = IndexKind::Flat;
    HnswParams hnsw{};                             // used only when index == Hnsw
    std::optional<PersistenceConfig> persistence{};  // in-memory only when unset
};

/// Facade that owns the store + index and exposes the public DB operations.
///
/// This is the integration seam: the WAL (`LogStore`) and snapshot store
/// (`BlobStore`) get injected here in week 7 without changing the query path.
/// Search resolves metadata filters into an allowed-set bitset, then delegates
/// to the index — so filtered ANN is wired end-to-end from day one.
class Engine {
public:
    explicit Engine(EngineConfig cfg);

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    InternalId insert(std::string ext_id, std::span<const float> vec, Metadata meta = {});
    /// Update an existing entry's vector. `meta`: nullopt leaves metadata unchanged,
    /// an engaged value replaces it (an empty map clears it). No-op (returns false)
    /// if the id is absent.
    bool       update(const std::string& ext_id, std::span<const float> vec,
                      std::optional<Metadata> meta = std::nullopt);
    bool       erase(const std::string& ext_id);

    /// Top-k nearest neighbours, optionally restricted by a metadata filter.
    /// `ef` overrides the index's default search breadth when > 0 (HNSW only).
    [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query, std::size_t k,
                                                   const Filter* filter = nullptr,
                                                   std::size_t ef = 0) const;

    /// Run many queries in parallel over the (immutable) read path. The filter's
    /// allowed-set is computed once and shared read-only across worker threads;
    /// queries are partitioned into contiguous chunks. `num_threads == 0` uses
    /// the hardware concurrency. Assumes no concurrent mutations during the call.
    [[nodiscard]] std::vector<std::vector<SearchResult>> search_batch(
        const std::vector<std::vector<float>>& queries, std::size_t k,
        const Filter* filter = nullptr, std::size_t ef = 0, unsigned num_threads = 0) const;

    /// Flush the write-ahead log to stable storage (no-op unless persistent).
    /// Useful with SyncPolicy::GroupCommit, where appends are not fsynced eagerly.
    void flush();

    /// Persist current state to a snapshot blob and truncate the WAL, so future
    /// recovery loads the snapshot then replays only post-snapshot ops.
    void snapshot();

    /// Reclaim tombstoned slots: rebuild the store with only live vectors (fresh
    /// dense ids) and rebuild the index. After this, slot_count() == size(). On a
    /// durable engine the compacted state is snapshotted and the WAL reset.
    void compact();

    [[nodiscard]] bool durable() const noexcept { return wal_ != nullptr; }

    [[nodiscard]] const VectorStore& store() const noexcept { return store_; }
    [[nodiscard]] const Index&       index() const noexcept { return *index_; }

private:
    InternalId apply_op(const Op& op, bool log_it);
    void       recover();
    void       replay_blob(std::span<const std::byte> data);

    EngineConfig               cfg_;
    VectorStore                store_;
    std::unique_ptr<Index>     index_;
    std::unique_ptr<LogStore>  wal_;
    std::unique_ptr<BlobStore> blobs_;
};

}  // namespace toyvdb
