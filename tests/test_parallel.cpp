#include "toyvdb/engine.hpp"

#include <gtest/gtest.h>

#include <random>
#include <vector>

using namespace toyvdb;

namespace {

std::vector<float> random_vec(std::mt19937& rng, Dim dim) {
    std::uniform_real_distribution<float> d(-1.0F, 1.0F);
    std::vector<float> v(dim);
    for (auto& x : v) x = d(rng);
    return v;
}

// Parallel batch results must be identical to running each query serially.
void expect_same(const std::vector<SearchResult>& a, const std::vector<SearchResult>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].id, b[i].id);
        EXPECT_FLOAT_EQ(a[i].score, b[i].score);
    }
}

}  // namespace

TEST(Parallel, BatchMatchesSerialFlat) {
    constexpr Dim dim = 16;
    Engine        eng(EngineConfig{dim, MetricKind::L2, IndexKind::Flat});

    std::mt19937 rng(11);
    for (std::size_t i = 0; i < 1000; ++i) eng.insert("v" + std::to_string(i), random_vec(rng, dim));

    std::vector<std::vector<float>> queries;
    for (std::size_t i = 0; i < 64; ++i) queries.push_back(random_vec(rng, dim));

    const auto batch = eng.search_batch(queries, 10, nullptr, 0, /*threads=*/4);
    ASSERT_EQ(batch.size(), queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const auto serial = eng.search(queries[i], 10);
        expect_same(batch[i], serial);
    }
}

TEST(Parallel, BatchMatchesSerialHnsw) {
    constexpr Dim dim = 16;
    Engine        eng(EngineConfig{dim, MetricKind::L2, IndexKind::Hnsw});

    std::mt19937 rng(22);
    for (std::size_t i = 0; i < 1000; ++i) eng.insert("v" + std::to_string(i), random_vec(rng, dim));

    std::vector<std::vector<float>> queries;
    for (std::size_t i = 0; i < 64; ++i) queries.push_back(random_vec(rng, dim));

    // HNSW search is deterministic for a fixed graph, so parallel == serial exactly.
    const auto batch = eng.search_batch(queries, 10, nullptr, 80, /*threads=*/8);
    ASSERT_EQ(batch.size(), queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const auto serial = eng.search(queries[i], 10, nullptr, 80);
        expect_same(batch[i], serial);
    }
}

TEST(Parallel, FilteredBatchMatchesSerial) {
    constexpr Dim dim = 8;
    Engine        eng(EngineConfig{dim, MetricKind::L2, IndexKind::Flat});

    std::mt19937 rng(33);
    for (std::size_t i = 0; i < 500; ++i) {
        Metadata m;
        m["even"] = static_cast<bool>(i % 2 == 0);
        eng.insert("v" + std::to_string(i), random_vec(rng, dim), m);
    }

    std::vector<std::vector<float>> queries;
    for (std::size_t i = 0; i < 32; ++i) queries.push_back(random_vec(rng, dim));

    const auto only_even = Filter::eq("even", true);
    const auto batch = eng.search_batch(queries, 5, &only_even, 0, /*threads=*/4);
    ASSERT_EQ(batch.size(), queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const auto serial = eng.search(queries[i], 5, &only_even);
        expect_same(batch[i], serial);
        for (const auto& h : batch[i]) {
            EXPECT_TRUE(std::get<bool>(eng.store().metadata(h.id)->at("even")));
        }
    }
}

TEST(Parallel, EmptyQueriesReturnsEmpty) {
    Engine eng(EngineConfig{4, MetricKind::L2, IndexKind::Flat});
    eng.insert("a", std::vector<float>{0, 0, 0, 0});
    const std::vector<std::vector<float>> none;
    EXPECT_TRUE(eng.search_batch(none, 5).empty());
}

TEST(Parallel, MoreThreadsThanQueriesIsSafe) {
    Engine eng(EngineConfig{4, MetricKind::L2, IndexKind::Flat});
    std::mt19937 rng(44);
    for (std::size_t i = 0; i < 50; ++i) eng.insert("v" + std::to_string(i), random_vec(rng, 4));

    std::vector<std::vector<float>> queries{random_vec(rng, 4), random_vec(rng, 4)};
    const auto batch = eng.search_batch(queries, 3, nullptr, 0, /*threads=*/16);
    ASSERT_EQ(batch.size(), 2U);
    EXPECT_EQ(batch[0].size(), 3U);
}
