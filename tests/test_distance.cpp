#include "toyvdb/distance.hpp"

#include <gtest/gtest.h>

#include <array>

using namespace toyvdb;

namespace {
constexpr float kEps = 1e-5F;
}

TEST(Distance, L2SquaredIsCorrect) {
    const std::array<float, 3> a{1.0F, 2.0F, 3.0F};
    const std::array<float, 3> b{4.0F, 6.0F, 3.0F};
    // (3)^2 + (4)^2 + (0)^2 = 25
    EXPECT_NEAR(L2::distance(a.data(), b.data(), 3), 25.0F, kEps);
}

TEST(Distance, L2OfIdenticalIsZero) {
    const std::array<float, 4> a{1.0F, -2.0F, 0.5F, 7.0F};
    EXPECT_NEAR(L2::distance(a.data(), a.data(), 4), 0.0F, kEps);
}

TEST(Distance, CosineIdenticalDirectionIsZero) {
    const std::array<float, 3> a{1.0F, 2.0F, 2.0F};
    const std::array<float, 3> b{2.0F, 4.0F, 4.0F};  // same direction, scaled
    EXPECT_NEAR(Cosine::distance(a.data(), b.data(), 3), 0.0F, kEps);
}

TEST(Distance, CosineOrthogonalIsOne) {
    const std::array<float, 2> a{1.0F, 0.0F};
    const std::array<float, 2> b{0.0F, 1.0F};
    EXPECT_NEAR(Cosine::distance(a.data(), b.data(), 2), 1.0F, kEps);
}

TEST(Distance, CosineOppositeIsTwo) {
    const std::array<float, 2> a{1.0F, 0.0F};
    const std::array<float, 2> b{-1.0F, 0.0F};
    EXPECT_NEAR(Cosine::distance(a.data(), b.data(), 2), 2.0F, kEps);
}

TEST(Distance, CosineZeroVectorIsMaxDistance) {
    const std::array<float, 2> a{0.0F, 0.0F};
    const std::array<float, 2> b{1.0F, 1.0F};
    EXPECT_NEAR(Cosine::distance(a.data(), b.data(), 2), 1.0F, kEps);
}

TEST(Distance, CosineIsNeverNegative) {
    // A near-perfect match can round just past similarity 1.0; distance must
    // still clamp to >= 0 rather than going slightly negative.
    const std::array<float, 5> a{0.3F, 0.6F, 0.1F, 0.7F, 0.2F};
    EXPECT_GE(Cosine::distance(a.data(), a.data(), 5), 0.0F);
    EXPECT_NEAR(Cosine::distance(a.data(), a.data(), 5), 0.0F, kEps);
}

TEST(Distance, DotIsNegatedSimilarity) {
    const std::array<float, 3> a{1.0F, 2.0F, 3.0F};
    const std::array<float, 3> b{1.0F, 1.0F, 1.0F};
    // dot = 6, negated -> -6 (smaller == more similar)
    EXPECT_NEAR(Dot::distance(a.data(), b.data(), 3), -6.0F, kEps);
}

TEST(Distance, MetricsAreSymmetric) {
    const std::array<float, 4> a{0.1F, 0.7F, -0.3F, 0.9F};
    const std::array<float, 4> b{-0.5F, 0.2F, 0.8F, 0.4F};
    EXPECT_NEAR(L2::distance(a.data(), b.data(), 4), L2::distance(b.data(), a.data(), 4), kEps);
    EXPECT_NEAR(Cosine::distance(a.data(), b.data(), 4), Cosine::distance(b.data(), a.data(), 4),
                kEps);
    EXPECT_NEAR(Dot::distance(a.data(), b.data(), 4), Dot::distance(b.data(), a.data(), 4), kEps);
}
