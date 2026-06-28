#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <span>
#include <string_view>
#include <vector>

#include "distance.hpp"
#include "index.hpp"
#include "vector_store.hpp"

namespace toyvdb {

/// HNSW tuning knobs. Defaults follow the common FAISS/hnswlib values.
struct HnswParams {
    int         M = 16;               ///< neighbours per layer (layer-0 cap = 2*M)
    int         efConstruction = 200;  ///< candidate breadth while inserting
    std::size_t efSearch = 64;         ///< default candidate breadth while querying
    unsigned    seed = 42;             ///< RNG seed for layer assignment (determinism)
};

/// Hierarchical Navigable Small World index.
///
/// Reference: Malkov & Yashunin, "Efficient and Robust ANN Search Using
/// Hierarchical Navigable Small World Graphs" (arXiv:1603.09320). The methods
/// below map onto the paper's algorithms: greedy descent through upper layers
/// (Alg. 5 zoom-in), `search_layer` (Alg. 2), neighbour heuristic (Alg. 4),
/// and insertion (Alg. 1).
///
/// Like FlatIndex, this stores only internal ids and reads vectors from the
/// store on demand; deletion is a soft tombstone at the store level (search
/// skips non-live nodes but still routes through them to preserve graph
/// connectivity). Templated on the metric so the distance kernel inlines.
template <DistanceMetric Met>
class HnswIndex final : public Index {
public:
    HnswIndex(const VectorStore& store, HnswParams params)
        : store_(store),
          params_(params),
          mL_(1.0 / std::log(static_cast<double>(std::max(2, params.M)))),
          rng_(params.seed) {}

    void add(InternalId id, std::span<const float> vec) override;
    void remove(InternalId /*id*/) override {}  // soft delete via store tombstone + is_live gate

    [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query, std::size_t k,
                                                   const SearchParams& params) const override;

    [[nodiscard]] std::string_view name() const override { return Met::name(); }

    // Introspection (used by invariant tests).
    [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
    [[nodiscard]] int         max_level() const noexcept { return max_level_; }
    [[nodiscard]] InternalId  entry_point() const noexcept { return entry_point_; }
    [[nodiscard]] std::size_t neighbor_count(InternalId id, int layer) const {
        return nodes_[id].links[static_cast<std::size_t>(layer)].size();
    }
    [[nodiscard]] int node_level(InternalId id) const { return nodes_[id].level; }

private:
    struct Node {
        int                                   level = 0;
        std::vector<std::vector<InternalId>>  links;  // links[l] = neighbours at layer l
    };
    struct Cand {
        Score      dist;
        InternalId id;
    };
    struct FarOnTop {  // max-heap: top() is the farthest
        bool operator()(const Cand& a, const Cand& b) const { return a.dist < b.dist; }
    };
    struct NearOnTop {  // min-heap: top() is the nearest
        bool operator()(const Cand& a, const Cand& b) const { return a.dist > b.dist; }
    };

    /// Epoch-stamped visited set; avoids reallocation/clearing per call.
    struct Visited {
        std::vector<std::uint32_t> stamp;
        std::uint32_t              cur = 0;
        void ensure(std::size_t n) {
            if (stamp.size() < n) stamp.resize(n, 0);
        }
        void bump() {
            ++cur;
            if (cur == 0) {  // wrapped: clear and restart
                std::fill(stamp.begin(), stamp.end(), 0);
                cur = 1;
            }
        }
        bool test_set(InternalId id) {
            if (stamp[id] == cur) return true;
            stamp[id] = cur;
            return false;
        }
    };

    [[nodiscard]] Score dist_to(const float* q, InternalId id) const {
        return Met::distance(q, store_.get(id).data(), store_.dim());
    }
    [[nodiscard]] std::size_t max_links(int layer) const {
        return layer == 0 ? static_cast<std::size_t>(2 * params_.M)
                          : static_cast<std::size_t>(params_.M);
    }

    int                 random_level();
    [[nodiscard]] Cand  greedy_upper(const float* q, Cand best, int layer) const;
    [[nodiscard]] std::vector<Cand> search_layer(const float* q, const std::vector<Cand>& entries,
                                                 std::size_t ef, int layer, const Bitset* allowed,
                                                 Visited& vis) const;
    [[nodiscard]] std::vector<Cand> select_neighbors(std::vector<Cand> cands, std::size_t m) const;
    void connect(InternalId a, InternalId b, int layer);
    void prune(InternalId id, int layer, std::size_t cap);

    const VectorStore&        store_;
    HnswParams                params_;
    double                    mL_;
    std::mt19937              rng_;
    std::vector<Node>         nodes_;
    std::vector<std::uint8_t> built_;  // 1 once a node has been inserted into the graph
    InternalId                entry_point_ = kInvalidId;
    int                       max_level_ = -1;
};

// ---------------------------------------------------------------------------
// Implementation (header-only template).
// ---------------------------------------------------------------------------

template <DistanceMetric Met>
int HnswIndex<Met>::random_level() {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double r = 1.0 - u(rng_);  // map [0,1) -> (0,1] so log is finite
    return static_cast<int>(-std::log(r) * mL_);
}

template <DistanceMetric Met>
typename HnswIndex<Met>::Cand HnswIndex<Met>::greedy_upper(const float* q, Cand best,
                                                           int layer) const {
    const std::size_t l = static_cast<std::size_t>(layer);
    bool changed = true;
    while (changed) {
        changed = false;
        if (l >= nodes_[best.id].links.size()) break;
        for (const InternalId e : nodes_[best.id].links[l]) {
            const Score d = dist_to(q, e);
            if (d < best.dist) {
                best = Cand{d, e};
                changed = true;
            }
        }
    }
    return best;
}

template <DistanceMetric Met>
std::vector<typename HnswIndex<Met>::Cand> HnswIndex<Met>::search_layer(
    const float* q, const std::vector<Cand>& entries, std::size_t ef, int layer,
    const Bitset* allowed, Visited& vis) const {
    vis.bump();

    std::priority_queue<Cand, std::vector<Cand>, NearOnTop> cand;  // frontier (min by dist)
    std::priority_queue<Cand, std::vector<Cand>, FarOnTop>  res;   // results  (max by dist)

    const auto admit = [&](InternalId id) {
        return store_.is_live(id) && (allowed == nullptr || allowed->test(id));
    };

    for (const Cand& e : entries) {
        vis.test_set(e.id);
        cand.push(e);
        if (admit(e.id)) res.push(e);
    }

    const std::size_t l = static_cast<std::size_t>(layer);
    while (!cand.empty()) {
        const Cand c = cand.top();
        cand.pop();
        const Score bound = res.empty() ? std::numeric_limits<Score>::max() : res.top().dist;
        if (res.size() >= ef && c.dist > bound) break;  // frontier can't improve results

        if (l >= nodes_[c.id].links.size()) continue;
        for (const InternalId e : nodes_[c.id].links[l]) {
            if (vis.test_set(e)) continue;
            const Score d = dist_to(q, e);
            const Score wbound = res.empty() ? std::numeric_limits<Score>::max() : res.top().dist;
            // Navigate through any node, but only admit allowed+live ones as results.
            if (res.size() < ef || d < wbound) {
                cand.push(Cand{d, e});
                if (admit(e)) {
                    res.push(Cand{d, e});
                    if (res.size() > ef) res.pop();
                }
            }
        }
    }

    std::vector<Cand> out;
    out.reserve(res.size());
    while (!res.empty()) {
        out.push_back(res.top());
        res.pop();
    }
    return out;
}

template <DistanceMetric Met>
std::vector<typename HnswIndex<Met>::Cand> HnswIndex<Met>::select_neighbors(std::vector<Cand> cands,
                                                                            std::size_t m) const {
    // Heuristic (Alg. 4): prefer candidates that are closer to the query than to
    // any already-selected neighbour, which spreads links and improves recall.
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.dist < b.dist; });

    std::vector<Cand> result;
    result.reserve(std::min(m, cands.size()));
    for (const Cand& cand : cands) {
        if (result.size() >= m) break;
        const float* cv = store_.get(cand.id).data();
        bool keep = true;
        for (const Cand& r : result) {
            const Score d = Met::distance(cv, store_.get(r.id).data(), store_.dim());
            if (d < cand.dist) {  // closer to an existing pick than to the query
                keep = false;
                break;
            }
        }
        if (keep) result.push_back(cand);
    }
    return result;
}

template <DistanceMetric Met>
void HnswIndex<Met>::connect(InternalId a, InternalId b, int layer) {
    const std::size_t l = static_cast<std::size_t>(layer);
    nodes_[a].links[l].push_back(b);
    nodes_[b].links[l].push_back(a);
}

template <DistanceMetric Met>
void HnswIndex<Met>::prune(InternalId id, int layer, std::size_t cap) {
    const std::size_t          l = static_cast<std::size_t>(layer);
    std::vector<InternalId>&   lst = nodes_[id].links[l];
    if (lst.size() <= cap) return;

    const float* v = store_.get(id).data();
    std::vector<Cand> cc;
    cc.reserve(lst.size());
    for (const InternalId x : lst) {
        cc.push_back(Cand{Met::distance(v, store_.get(x).data(), store_.dim()), x});
    }
    const std::vector<Cand> kept = select_neighbors(std::move(cc), cap);
    lst.clear();
    for (const Cand& kc : kept) lst.push_back(kc.id);
}

template <DistanceMetric Met>
void HnswIndex<Met>::add(InternalId id, std::span<const float> vec) {
    (void)vec;  // vector is read from the store; param exists for the Index contract
    const std::size_t needed = static_cast<std::size_t>(id) + 1;
    if (nodes_.size() < needed) {
        nodes_.resize(needed);
        built_.resize(needed, 0);
    }
    if (built_[id]) return;  // revive/update keeps the existing graph position (MVP)

    const float* q = store_.get(id).data();
    const int    level = random_level();
    nodes_[id].level = level;
    nodes_[id].links.assign(static_cast<std::size_t>(level) + 1, {});
    built_[id] = 1;

    if (entry_point_ == kInvalidId) {  // first node
        entry_point_ = id;
        max_level_ = level;
        return;
    }

    Visited vis;
    vis.ensure(nodes_.size());

    Cand ep{dist_to(q, entry_point_), entry_point_};
    for (int lc = max_level_; lc > level; --lc) {
        ep = greedy_upper(q, ep, lc);  // zoom in through upper layers
    }

    // From this layer down, carry the full candidate set as the next layer's
    // entry points (per Alg. 1) -- this materially improves graph quality vs
    // seeding each layer with only the single closest node.
    std::vector<Cand> entries{ep};
    const int         start = std::min(level, max_level_);
    for (int lc = start; lc >= 0; --lc) {
        std::vector<Cand> w = search_layer(q, entries, static_cast<std::size_t>(params_.efConstruction),
                                           lc, nullptr, vis);
        const std::vector<Cand> neighbors =
            select_neighbors(w, static_cast<std::size_t>(params_.M));

        for (const Cand& nb : neighbors) connect(id, nb.id, lc);

        const std::size_t cap = max_links(lc);
        for (const Cand& nb : neighbors) prune(nb.id, lc, cap);

        if (!w.empty()) entries = std::move(w);
    }

    if (level > max_level_) {
        max_level_ = level;
        entry_point_ = id;
    }
}

template <DistanceMetric Met>
std::vector<SearchResult> HnswIndex<Met>::search(std::span<const float> query, std::size_t k,
                                                 const SearchParams& params) const {
    std::vector<SearchResult> out;
    if (k == 0 || entry_point_ == kInvalidId) return out;

    const float* q = query.data();
    Visited      vis;
    vis.ensure(nodes_.size());

    Cand ep{dist_to(q, entry_point_), entry_point_};
    for (int lc = max_level_; lc >= 1; --lc) {
        ep = greedy_upper(q, ep, lc);  // route to the base layer
    }

    const std::size_t ef = std::max(params.ef, k);
    std::vector<Cand> w = search_layer(q, std::vector<Cand>{ep}, ef, 0, params.allowed, vis);
    std::sort(w.begin(), w.end(), [](const Cand& a, const Cand& b) { return a.dist < b.dist; });

    const std::size_t n = std::min(k, w.size());
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) out.push_back(SearchResult{w[i].id, w[i].dist});
    return out;
}

/// Build an HnswIndex for a runtime-chosen metric, as a polymorphic Index.
[[nodiscard]] std::unique_ptr<Index> make_hnsw_index(const VectorStore& store, MetricKind metric,
                                                     HnswParams params = {});

}  // namespace toyvdb
