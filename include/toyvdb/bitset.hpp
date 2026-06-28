#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace toyvdb {

/// Minimal fixed-size bitset over 64-bit words.
///
/// Used as the "allowed set" for metadata-filtered search: a set bit at index
/// `i` means internal id `i` passed the filter and may be returned. A simple
/// hand-rolled bitset is plenty for a single-node educational store; a roaring
/// bitmap would be the production upgrade for sparse, high-cardinality filters.
class Bitset {
public:
    Bitset() = default;
    explicit Bitset(std::size_t n) { resize(n); }

    void resize(std::size_t n) {
        size_ = n;
        words_.assign((n + 63) / 64, 0ULL);
    }

    void clear() { std::fill(words_.begin(), words_.end(), 0ULL); }

    void set(std::size_t i) { words_[i >> 6] |= (1ULL << (i & 63)); }
    void reset(std::size_t i) { words_[i >> 6] &= ~(1ULL << (i & 63)); }

    [[nodiscard]] bool test(std::size_t i) const {
        if (i >= size_) return false;
        return (words_[i >> 6] >> (i & 63)) & 1ULL;
    }

    /// In-place intersection (AND). Used to combine multiple filter clauses.
    void intersect(const Bitset& other) {
        const std::size_t n = std::min(words_.size(), other.words_.size());
        for (std::size_t i = 0; i < n; ++i) words_[i] &= other.words_[i];
        for (std::size_t i = n; i < words_.size(); ++i) words_[i] = 0ULL;
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    [[nodiscard]] std::size_t count() const noexcept {
        std::size_t c = 0;
        for (std::uint64_t w : words_) c += static_cast<std::size_t>(std::popcount(w));
        return c;
    }

private:
    std::vector<std::uint64_t> words_;
    std::size_t                size_ = 0;
};

}  // namespace toyvdb
