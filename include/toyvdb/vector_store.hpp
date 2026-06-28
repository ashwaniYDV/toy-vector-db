#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "metadata.hpp"
#include "types.hpp"

namespace toyvdb {

/// Logical mutation kinds. Every write to the store is one of these.
enum class OpType : std::uint8_t { Insert = 1, Update = 2, Delete = 3 };

/// A logical operation: the universal write contract.
///
/// This is the seam that makes durability and storage pluggable. The WAL (week 7)
/// serializes an `Op`, the store materializes it via `apply`, and indexes are
/// derived from the same stream. Replaying ops == replaying history.
struct Op {
    OpType                  type;
    std::string             ext_id;
    std::vector<float>      vec;   ///< empty for Delete
    std::optional<Metadata> meta;  ///< Insert: metadata to store. nullopt leaves it unchanged.
                                   ///< Unused for Delete.
};

/// In-memory vector store: the storage layer.
///
/// Memory layout: a single contiguous `float` arena in Structure-of-Arrays (SoA)
/// form, so vector `i` occupies `arena[i*dim .. (i+1)*dim)`. This maximizes cache
/// locality and is SIMD-friendly.
///
/// ID scheme: callers refer to vectors by an arbitrary **external id** (a string,
/// e.g. "doc-1"). Internally each vector gets a dense **internal id** assigned in
/// insertion order -- the first insert is 0, the next 1, then 2, and so on. These
/// dense ids double as array indices: vector `id` lives at `arena[id*dim ..]`, its
/// metadata at `meta_[id]`, its liveness flag at `live_[id]`. Two structures map
/// between the id spaces:
///   - `ext_ids_[id]`      internal id -> external id   (a vector, indexed by id)
///   - `ext_to_int_[ext]`  external id -> internal id   (a hash map)
///
/// Example: insert("doc-1"), insert("doc-2") assigns ids 0 and 1, giving
///   ext_ids_    = ["doc-1", "doc-2"]
///   ext_to_int_ = {"doc-1": 0, "doc-2": 1}
/// Indexes (Flat/HNSW) store only the small dense internal ids and never see the
/// external string; resolution happens here at the store boundary.
///
/// Deletes are tombstones (O(1)); the slot is reused if the same external id is
/// re-inserted. Tombstoned slots are physically reclaimed -- and ids renumbered --
/// only by Engine::compact().
class VectorStore {
public:
    explicit VectorStore(Dim dim, std::size_t reserve = 0);

    // --- Convenience API: builds an Op and applies it. ---------------------
    InternalId insert(std::string ext_id, std::span<const float> vec, Metadata meta = {});
    /// Update an existing **live** entry's vector. `meta` controls metadata:
    ///   - nullopt (default): leave the existing metadata unchanged;
    ///   - engaged (even an empty map): replace it (an empty map clears metadata).
    /// No-op returning false if the id is absent or deleted; only insert() revives
    /// a deleted id.
    bool       update(const std::string& ext_id, std::span<const float> vec,
                      std::optional<Metadata> meta = std::nullopt);
    bool       erase(const std::string& ext_id);

    /// The single mutation seam. Returns the affected internal id: the new/revived
    /// id for Insert; the entry's id for a successful Update/Delete; nullopt if an
    /// Update/Delete targeted an absent (or already-deleted) id. Insert always
    /// yields an id.
    std::optional<InternalId> apply(const Op& op);

    // --- Read API ----------------------------------------------------------
    /// O(1) view into the internal arena for a slot's vector.
    ///
    /// Precondition: `id < slot_count()` (debug-asserted). The slot need NOT be
    /// live: get() intentionally returns the (stale) bytes of a tombstoned slot,
    /// because HNSW routes *through* deleted nodes and must read their vectors to
    /// compute distances. Callers that build results filter on is_live(id)
    /// separately. This is the trust-the-caller fast path used in the distance
    /// inner loop, so it performs no liveness check.
    ///
    /// Lifetime: the returned span does NOT own memory. It points into `arena_`
    /// and stays valid only until the next operation that grows or rebuilds the
    /// arena -- i.e. an insert() of a *new* external id (which may reallocate the
    /// arena) or Engine::compact() -- or until the VectorStore is destroyed.
    /// update() and erase() do not invalidate it (they overwrite in place /
    /// tombstone). Copy the data out if you need to retain it across a mutation;
    /// do not hold on to the span.
    [[nodiscard]] std::span<const float> get(InternalId id) const;

    /// Resolve an external id to an internal id (nullopt if absent or tombstoned).
    [[nodiscard]] std::optional<InternalId> resolve(const std::string& ext_id) const;

    [[nodiscard]] const Metadata* metadata(InternalId id) const;
    [[nodiscard]] bool            is_live(InternalId id) const;

    /// External id for an internal id (used when snapshotting the store).
    [[nodiscard]] const std::string& external_id(InternalId id) const;

    [[nodiscard]] Dim         dim() const noexcept { return dim_; }
    [[nodiscard]] std::size_t size() const noexcept { return live_count_; }
    /// Number of allocated slots (live + tombstoned); internal ids are < slot_count().
    /// NOTE: it is the count of slots ever allocated. Tombstoned slots are reclaimed by Engine::compact().
    [[nodiscard]] std::size_t slot_count() const noexcept { return ext_ids_.size(); }

private:
    InternalId                do_insert(const std::string& ext_id, std::span<const float> vec,
                                        const Metadata& meta);
    std::optional<InternalId> do_update(const std::string& ext_id, std::span<const float> vec,
                                        const Metadata* meta);
    std::optional<InternalId> do_erase(const std::string& ext_id);
    void                      validate_dim(std::span<const float> vec) const;

    Dim                       dim_;
    std::vector<float>        arena_;    ///< slot_count()*dim_, row-major; vector id at [id*dim_ ..]
    std::vector<std::uint8_t> live_;     ///< live_[id]: 1 = live, 0 = tombstoned
    std::vector<std::string>  ext_ids_;  ///< ext_ids_[id]: internal id -> external id
    std::vector<Metadata>     meta_;     ///< meta_[id]: internal id -> metadata
    std::unordered_map<std::string, InternalId> ext_to_int_;  ///< external id -> internal id
    std::size_t               live_count_ = 0;                ///< number of live (non-tombstoned) slots
};

}  // namespace toyvdb
