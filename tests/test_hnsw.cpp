#include "toyvdb/engine.hpp"

#include <gtest/gtest.h>

#include <random>
#include <unordered_set>
#include <vector>

using namespace toyvdb;

namespace {

std::vector<float> random_vec(std::mt19937& rng, Dim dim) {
    std::uniform_real_distribution<float> d(-1.0F, 1.0F);
    std::vector<float> v(dim);
    for (auto& x : v) x = d(rng);
    return v;
}

std::vector<float> v2(float a, float b) { return {a, b}; }

}  // namespace

// The headline property: HNSW must closely match the exact FlatIndex oracle.
TEST(Hnsw, RecallMatchesFlatOracle) {
    constexpr Dim         dim = 16;
    constexpr std::size_t n = 2000;
    constexpr std::size_t q = 100;
    constexpr std::size_t k = 10;
    constexpr std::size_t ef = 200;

    Engine flat(EngineConfig{dim, MetricKind::L2, IndexKind::Flat});
    Engine hnsw(EngineConfig{dim, MetricKind::L2, IndexKind::Hnsw});

    std::mt19937 rng(123);
    for (std::size_t i = 0; i < n; ++i) {
        const auto vec = random_vec(rng, dim);
        const auto id = "v" + std::to_string(i);
        flat.insert(id, vec);
        hnsw.insert(id, vec);
    }

    double recall_sum = 0.0;
    for (std::size_t i = 0; i < q; ++i) {
        const auto query = random_vec(rng, dim);

        const auto truth = flat.search(query, k);
        const auto got = hnsw.search(query, k, nullptr, ef);
        ASSERT_EQ(got.size(), k);

        std::unordered_set<InternalId> want;
        for (const auto& r : truth) want.insert(r.id);
        std::size_t hit = 0;
        for (const auto& r : got) {
            if (want.contains(r.id)) ++hit;
        }
        recall_sum += static_cast<double>(hit) / static_cast<double>(k);
    }

    const double recall = recall_sum / static_cast<double>(q);
    EXPECT_GE(recall, 0.90) << "HNSW recall@" << k << " = " << recall;
}

TEST(Hnsw, ResultsAreAscendingByDistance) {
    Engine hnsw(EngineConfig{8, MetricKind::L2, IndexKind::Hnsw});
    std::mt19937 rng(7);
    for (std::size_t i = 0; i < 200; ++i) hnsw.insert("v" + std::to_string(i), random_vec(rng, 8));

    const auto hits = hnsw.search(random_vec(rng, 8), 10, nullptr, 64);
    ASSERT_FALSE(hits.empty());
    for (std::size_t i = 1; i < hits.size(); ++i) {
        EXPECT_LE(hits[i - 1].score, hits[i].score);
    }
}

TEST(Hnsw, EmptyIndexReturnsEmpty) {
    Engine hnsw(EngineConfig{4, MetricKind::L2, IndexKind::Hnsw});
    const std::vector<float> query{0.0F, 0.0F, 0.0F, 0.0F};
    EXPECT_TRUE(hnsw.search(query, 5).empty());
}

TEST(Hnsw, SingleNode) {
    Engine hnsw(EngineConfig{3, MetricKind::L2, IndexKind::Hnsw});
    hnsw.insert("only", std::vector<float>{1.0F, 2.0F, 3.0F});
    const auto hits = hnsw.search(std::vector<float>{1.0F, 2.0F, 3.0F}, 5);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(*hnsw.store().resolve("only"), hits[0].id);
}

TEST(Hnsw, ErasedVectorsAreNotReturned) {
    Engine hnsw(EngineConfig{2, MetricKind::L2, IndexKind::Hnsw});
    hnsw.insert("a", v2(0.0F, 0.0F));
    hnsw.insert("b", v2(1.0F, 0.0F));
    hnsw.insert("c", v2(2.0F, 0.0F));
    hnsw.erase("a");

    const auto hits = hnsw.search(v2(0.0F, 0.0F), 5, nullptr, 32);
    for (const auto& h : hits) {
        EXPECT_NE(h.id, 0U) << "erased node should not appear";  // 'a' had internal id 0
    }
}

TEST(Hnsw, MetadataFilteredSearch) {
    constexpr Dim dim = 8;
    Engine        hnsw(EngineConfig{dim, MetricKind::L2, IndexKind::Hnsw});

    std::mt19937 rng(99);
    for (std::size_t i = 0; i < 500; ++i) {
        Metadata m;
        m["even"] = static_cast<bool>(i % 2 == 0);
        hnsw.insert("v" + std::to_string(i), random_vec(rng, dim), m);
    }

    const auto query = random_vec(rng, dim);
    const auto only_even = Filter::eq("even", true);
    const auto hits = hnsw.search(query, 10, &only_even, 200);

    ASSERT_FALSE(hits.empty());
    for (const auto& h : hits) {
        const Metadata* m = hnsw.store().metadata(h.id);
        ASSERT_NE(m, nullptr);
        EXPECT_TRUE(std::get<bool>(m->at("even")));
    }
}

TEST(Hnsw, GraphInvariantsHold) {
    constexpr Dim dim = 8;
    Engine        eng(EngineConfig{dim, MetricKind::L2, IndexKind::Hnsw});
    std::mt19937  rng(5);
    for (std::size_t i = 0; i < 300; ++i) eng.insert("v" + std::to_string(i), random_vec(rng, dim));

    // Downcast through the polymorphic interface for white-box invariant checks.
    const auto& idx = dynamic_cast<const HnswIndex<L2>&>(eng.index());
    ASSERT_EQ(idx.node_count(), 300U);
    ASSERT_NE(idx.entry_point(), kInvalidId);

    // Every node's layer-0 neighbour list must respect the cap (2*M) and be non-empty
    // for a connected graph of this size.
    const std::size_t cap0 = 2U * 16U;  // 2*M with default M=16
    for (InternalId id = 0; id < idx.node_count(); ++id) {
        const std::size_t deg = idx.neighbor_count(id, 0);
        EXPECT_LE(deg, cap0);
        EXPECT_GT(deg, 0U);
    }
}
