#include "toyvdb/engine.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace toyvdb;

namespace {
std::vector<float> v2(float a, float b) { return {a, b}; }
}

// Use L2 so geometric distances are easy to reason about.
TEST(FlatIndex, ReturnsNearestInAscendingOrder) {
    Engine eng(EngineConfig{/*dim=*/2, MetricKind::L2});
    eng.insert("a", v2(0.0F, 0.0F));
    eng.insert("b", v2(1.0F, 0.0F));
    eng.insert("c", v2(5.0F, 0.0F));
    eng.insert("d", v2(10.0F, 0.0F));

    const auto query = v2(0.0F, 0.0F);
    const auto hits = eng.search(query, 3);

    ASSERT_EQ(hits.size(), 3U);
    // Closest -> farthest: a(0), b(1), c(25)
    EXPECT_EQ(*eng.store().resolve("a"), hits[0].id);
    EXPECT_EQ(*eng.store().resolve("b"), hits[1].id);
    EXPECT_EQ(*eng.store().resolve("c"), hits[2].id);

    // Scores are non-decreasing (ascending distance).
    EXPECT_LE(hits[0].score, hits[1].score);
    EXPECT_LE(hits[1].score, hits[2].score);
}

TEST(FlatIndex, KLargerThanCorpusReturnsAll) {
    Engine eng(EngineConfig{2, MetricKind::L2});
    eng.insert("a", v2(0, 0));
    eng.insert("b", v2(1, 1));

    const auto hits = eng.search(v2(0, 0), 10);
    EXPECT_EQ(hits.size(), 2U);
}

TEST(FlatIndex, KZeroReturnsEmpty) {
    Engine eng(EngineConfig{2, MetricKind::L2});
    eng.insert("a", v2(0, 0));
    EXPECT_TRUE(eng.search(v2(0, 0), 0).empty());
}

TEST(FlatIndex, ErasedVectorsAreNotReturned) {
    Engine eng(EngineConfig{2, MetricKind::L2});
    eng.insert("a", v2(0, 0));
    eng.insert("b", v2(1, 0));
    eng.erase("a");

    const auto hits = eng.search(v2(0, 0), 5);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(*eng.store().resolve("b"), hits[0].id);
}

TEST(FlatIndex, MetadataFilterRestrictsCandidates) {
    Engine eng(EngineConfig{2, MetricKind::L2});

    Metadata en;
    en["lang"] = std::string("en");
    Metadata fr;
    fr["lang"] = std::string("fr");

    eng.insert("a", v2(0.0F, 0.0F), en);  // closest, but English
    eng.insert("b", v2(1.0F, 0.0F), fr);  // farther, French
    eng.insert("c", v2(2.0F, 0.0F), fr);  // farthest, French

    Filter only_fr = Filter::eq("lang", std::string("fr"));
    const auto hits = eng.search(v2(0.0F, 0.0F), 5, &only_fr);

    ASSERT_EQ(hits.size(), 2U);  // "a" filtered out despite being nearest
    EXPECT_EQ(*eng.store().resolve("b"), hits[0].id);
    EXPECT_EQ(*eng.store().resolve("c"), hits[1].id);
}

TEST(FlatIndex, NumericRangeFilter) {
    Engine eng(EngineConfig{2, MetricKind::L2});
    for (int i = 0; i < 5; ++i) {
        Metadata m;
        m["year"] = std::int64_t{2020 + i};
        eng.insert("doc-" + std::to_string(i), v2(static_cast<float>(i), 0.0F), m);
    }

    Filter recent = Filter::ge("year", std::int64_t{2023});
    const auto hits = eng.search(v2(0.0F, 0.0F), 10, &recent);

    // years 2023, 2024 -> docs 3 and 4
    ASSERT_EQ(hits.size(), 2U);
}

TEST(FlatIndex, ReinsertSameIdDoesNotDuplicateResults) {
    Engine eng(EngineConfig{2, MetricKind::L2});
    eng.insert("a", v2(0.0F, 0.0F));
    eng.insert("a", v2(0.0F, 0.0F));  // re-insert same id (update) must not duplicate
    eng.insert("a", v2(0.0F, 0.0F));
    eng.insert("b", v2(1.0F, 0.0F));

    const auto hits = eng.search(v2(0.0F, 0.0F), 10);
    ASSERT_EQ(hits.size(), 2U);  // {a, b}, not {a, a, a, b}
}

TEST(FlatIndex, NameReflectsMetric) {
    Engine cos(EngineConfig{2, MetricKind::Cosine});
    Engine l2(EngineConfig{2, MetricKind::L2});
    EXPECT_EQ(cos.index().name(), "cosine");
    EXPECT_EQ(l2.index().name(), "l2");
}
