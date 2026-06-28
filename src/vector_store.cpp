#include "toyvdb/vector_store.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace toyvdb {

VectorStore::VectorStore(Dim dim, std::size_t reserve) : dim_(dim) {
    if (dim_ == 0) throw std::invalid_argument("VectorStore: dim must be > 0");
    if (reserve > 0) {
        arena_.reserve(reserve * dim_);
        live_.reserve(reserve);
        ext_ids_.reserve(reserve);
        meta_.reserve(reserve);
        ext_to_int_.reserve(reserve);
    }
}

void VectorStore::validate_dim(std::span<const float> vec) const {
    if (static_cast<Dim>(vec.size()) != dim_) {
        throw std::invalid_argument("VectorStore: vector dimensionality mismatch");
    }
}

InternalId VectorStore::insert(std::string ext_id, std::span<const float> vec, Metadata meta) {
    Op op{OpType::Insert, std::move(ext_id), std::vector<float>(vec.begin(), vec.end()),
          std::move(meta)};
    return apply(op).value();  // Insert always yields an id
}

bool VectorStore::update(const std::string& ext_id, std::span<const float> vec,
                         std::optional<Metadata> meta) {
    Op op{OpType::Update, ext_id, std::vector<float>(vec.begin(), vec.end()), std::move(meta)};
    return apply(op).has_value();
}

bool VectorStore::erase(const std::string& ext_id) {
    Op op{OpType::Delete, ext_id, {}, std::nullopt};
    return apply(op).has_value();
}

std::optional<InternalId> VectorStore::apply(const Op& op) {
    switch (op.type) {
        case OpType::Insert: {
            const Metadata empty;  // Insert always sets metadata (nullopt -> empty)
            return do_insert(op.ext_id, op.vec, op.meta ? *op.meta : empty);
        }
        case OpType::Update:
            // Pass a pointer only when the caller supplied metadata; nullptr means
            // "leave metadata unchanged" (an engaged-but-empty map clears it).
            return do_update(op.ext_id, op.vec, op.meta ? &*op.meta : nullptr);
        case OpType::Delete:
            return do_erase(op.ext_id);
    }
    return std::nullopt;
}

InternalId VectorStore::do_insert(const std::string& ext_id, std::span<const float> vec,
                                  const Metadata& meta) {
    validate_dim(vec);

    // Re-inserting an existing external id revives/overwrites its slot.
    if (const auto it = ext_to_int_.find(ext_id); it != ext_to_int_.end()) {
        const InternalId id = it->second;
        std::copy(vec.begin(), vec.end(), arena_.begin() + static_cast<std::ptrdiff_t>(id) * dim_);
        meta_[id] = meta;
        if (!live_[id]) {
            live_[id] = 1;
            ++live_count_;
        }
        return id;
    }

    // Brand-new external id: append a slot. The next dense internal id is simply
    // the current number of slots (0 for the first insert, 1 for the next, ...).
    const InternalId id = static_cast<InternalId>(ext_ids_.size());
    arena_.insert(arena_.end(), vec.begin(), vec.end());
    live_.push_back(1);
    ext_ids_.push_back(ext_id);
    meta_.push_back(meta);
    ext_to_int_.emplace(ext_id, id);  // record external -> internal mapping
    ++live_count_;
    return id;
}

std::optional<InternalId> VectorStore::do_update(const std::string& ext_id,
                                                 std::span<const float> vec, const Metadata* meta) {
    const auto it = ext_to_int_.find(ext_id);
    // Update affects only a *live* entry. A deleted (tombstoned) or absent id is a
    // no-op -- like SQL UPDATE on a missing row -- rather than a resurrection.
    // Use insert() to revive a deleted id.
    if (it == ext_to_int_.end() || live_[it->second] == 0) return std::nullopt;
    validate_dim(vec);

    const InternalId id = it->second;
    std::copy(vec.begin(), vec.end(), arena_.begin() + static_cast<std::ptrdiff_t>(id) * dim_);
    if (meta != nullptr) meta_[id] = *meta;
    return id;
}

std::optional<InternalId> VectorStore::do_erase(const std::string& ext_id) {
    const auto it = ext_to_int_.find(ext_id);
    if (it == ext_to_int_.end() || live_[it->second] == 0) return std::nullopt;
    live_[it->second] = 0;
    --live_count_;
    return it->second;
}

std::span<const float> VectorStore::get(InternalId id) const {
    assert(id < ext_ids_.size() && "VectorStore::get: id out of range");
    const auto offset = static_cast<std::size_t>(id) * dim_;
    return std::span<const float>(arena_.data() + offset, dim_);
}

std::optional<InternalId> VectorStore::resolve(const std::string& ext_id) const {
    const auto it = ext_to_int_.find(ext_id);
    if (it == ext_to_int_.end()) return std::nullopt;
    if (!live_[it->second]) return std::nullopt;
    return it->second;
}

const Metadata* VectorStore::metadata(InternalId id) const {
    if (id >= meta_.size()) return nullptr;
    return &meta_[id];
}

bool VectorStore::is_live(InternalId id) const {
    return id < live_.size() && live_[id] != 0;
}

const std::string& VectorStore::external_id(InternalId id) const {
    return ext_ids_[id];
}

}  // namespace toyvdb
