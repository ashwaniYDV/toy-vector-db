#pragma once

#include <cstdint>
#include <limits>

namespace toyvdb {

/// Dense, index-internal identifier. Assigned in insertion order (0, 1, 2, ...)
/// and used directly as an array index: vector `id` lives at `arena[id*dim ..]`.
/// User-facing string ids are mapped to/from these by the VectorStore; indexes
/// only ever store InternalId.
using InternalId = std::uint32_t;

/// Vector dimensionality.
using Dim = std::uint32_t;

/// Score / distance value. Smaller == closer for every metric (see distance.hpp).
using Score = float;

/// Log sequence number (used by the WAL, week 7).
using Lsn = std::uint64_t;

/// Sentinel for "no id".
inline constexpr InternalId kInvalidId = std::numeric_limits<InternalId>::max();

/// A single search hit. `score` is a distance: smaller is a closer match.
struct SearchResult {
    InternalId id{kInvalidId};
    Score      score{0};
};

}  // namespace toyvdb
