#pragma once

#include <algorithm>
#include <memory>
#include <queue>
#include <vector>

#include "distance.hpp"
#include "index.hpp"
#include "vector_store.hpp"

namespace toyvdb {

/// Exact brute-force index. O(N * dim) per query.
///
/// This is the project's **correctness oracle**: every approximate index (HNSW)
/// is validated by comparing its results against FlatIndex on the same data.
/// It stores only internal ids and reads vectors from the store on demand.
///
/// Templated on the distance metric so the inner loop inlines the kernel with no
/// virtual dispatch. `make_flat_index` bridges a runtime MetricKind to the right
/// instantiation behind the polymorphic Index interface.
template <DistanceMetric M>
class FlatIndex final : public Index {
public:
    explicit FlatIndex(const VectorStore& store) : store_(store) {}

    void add(InternalId id, std::span<const float> /*vec*/) override {
        // Flat reads vectors straight from the store, so we only track ids.
        ids_.push_back(id);
    }

    void remove(InternalId id) override {
        // Search already skips tombstoned ids via the store, so removal is lazy.
        // Drop the id eagerly too, keeping the scan list tight.
        ids_.erase(std::remove(ids_.begin(), ids_.end(), id), ids_.end());
    }

    [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query, std::size_t k,
                                                   const SearchParams& params) const override {
        std::vector<SearchResult> out;
        if (k == 0) return out;

        // Bounded max-heap of size k keyed by distance: the top is the current
        // worst (largest distance) kept candidate, so we evict it when a closer
        // one arrives. (smaller distance == closer; see distance.hpp)
        const auto worse = [](const SearchResult& a, const SearchResult& b) {
            return a.score < b.score;  // makes top() == largest distance
        };
        std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(worse)> heap(worse);

        const Dim          d = store_.dim();
        const float* const q = query.data();
        for (const InternalId id : ids_) {
            if (!store_.is_live(id)) continue;
            if (params.allowed != nullptr && !params.allowed->test(id)) continue;

            const Score dist = M::distance(q, store_.get(id).data(), d);
            if (heap.size() < k) {
                heap.push({id, dist});
            } else if (dist < heap.top().score) {
                heap.pop();
                heap.push({id, dist});
            }
        }

        out.resize(heap.size());
        for (std::size_t i = heap.size(); i-- > 0;) {  // emit ascending by distance
            out[i] = heap.top();
            heap.pop();
        }
        return out;
    }

    [[nodiscard]] std::string_view name() const override { return M::name(); }

private:
    const VectorStore&      store_;
    std::vector<InternalId> ids_;
};

/// Build a FlatIndex for a runtime-chosen metric, as a polymorphic Index.
[[nodiscard]] std::unique_ptr<Index> make_flat_index(const VectorStore& store, MetricKind metric);

}  // namespace toyvdb
