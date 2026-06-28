#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include "bitset.hpp"
#include "types.hpp"

namespace toyvdb {

/// Parameters that tune a single search.
struct SearchParams {
    /// HNSW search breadth (size of the dynamic candidate list). Ignored by Flat.
    std::size_t ef = 64;

    /// Optional allowed-set for metadata-filtered search. When non-null, only
    /// internal ids whose bit is set are eligible to be returned. The bitset is
    /// expected to be sized to the store's slot_count().
    const Bitset* allowed = nullptr;
};

/// Polymorphic index interface.
///
/// Indexes store **internal ids only** and read vectors back from the store on
/// demand. This boundary is crossed once per query, so virtual dispatch here is
/// free; the hot inner loop uses compile-time distance policies instead.
class Index {
public:
    virtual ~Index() = default;

    /// Insert `id` (vector provided to avoid a store lookup during build).
    virtual void add(InternalId id, std::span<const float> vec) = 0;

    /// Logically remove `id` from the index.
    virtual void remove(InternalId id) = 0;

    /// Return up to `k` nearest hits to `query`, ascending by distance.
    [[nodiscard]] virtual std::vector<SearchResult> search(
        std::span<const float> query, std::size_t k, const SearchParams& params) const = 0;

    [[nodiscard]] virtual std::string_view name() const = 0;
};

}  // namespace toyvdb
