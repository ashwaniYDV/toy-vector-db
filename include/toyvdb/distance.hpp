#pragma once

#include <cmath>
#include <concepts>
#include <span>
#include <string_view>

#include "types.hpp"

namespace toyvdb {

/// Distance convention for the whole codebase: **smaller == closer**.
///
/// This single rule removes the most common vector-search bug (mixing
/// similarity and distance orderings). Top-K is therefore always "keep the k
/// smallest scores", regardless of metric.
///
///   - L2:     squared Euclidean distance.
///   - Cosine: 1 - cosine_similarity        (0 == identical direction).
///   - Dot:    -dot_product                 (negated so larger similarity sorts first).
///
/// Metrics are compile-time policies: `Metric::distance(...)` inlines into the
/// search loop with zero virtual-dispatch overhead. A runtime `MetricKind` enum
/// + factory bridges to the polymorphic Index when the metric is chosen at runtime.

template <class M>
concept DistanceMetric = requires(const float* a, const float* b, Dim d) {
    { M::distance(a, b, d) } -> std::same_as<Score>;
};

struct L2 {
    static Score distance(const float* a, const float* b, Dim d) {
        Score sum = 0;
        for (Dim i = 0; i < d; ++i) {
            const Score diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;  // squared; monotonic with true L2, avoids a sqrt
    }
    static constexpr std::string_view name() { return "l2"; }
};

struct Dot {
    static Score distance(const float* a, const float* b, Dim d) {
        Score dot = 0;
        for (Dim i = 0; i < d; ++i) dot += a[i] * b[i];
        return -dot;
    }
    static constexpr std::string_view name() { return "dot"; }
};

struct Cosine {
    static Score distance(const float* a, const float* b, Dim d) {
        Score dot = 0, na = 0, nb = 0;
        for (Dim i = 0; i < d; ++i) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        const Score denom = std::sqrt(na) * std::sqrt(nb);
        if (denom == 0) return 1.0F;  // undefined direction -> maximally far
        Score sim = dot / denom;      // mathematically in [-1, 1]
        if (sim > 1.0F) sim = 1.0F;   // clamp float rounding so distance stays in [0, 2]
        if (sim < -1.0F) sim = -1.0F;
        return 1.0F - sim;
    }
    static constexpr std::string_view name() { return "cosine"; }
};

static_assert(DistanceMetric<L2>);
static_assert(DistanceMetric<Dot>);
static_assert(DistanceMetric<Cosine>);

/// Runtime metric selector, bridged to compile-time policies by the index factory.
enum class MetricKind { L2, Cosine, Dot };

/// Convenience span overloads (bounds come from the spans; sizes must match dim).
template <DistanceMetric M>
Score distance(std::span<const float> a, std::span<const float> b) {
    return M::distance(a.data(), b.data(), static_cast<Dim>(a.size()));
}

}  // namespace toyvdb
