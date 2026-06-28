// Behavioral edge cases across the stack -- the kinds of things that surface
// when actually driving the system, not just happy-path unit tests.

#include "toyvdb/engine.hpp"
#include "toyvdb/rag.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace toyvdb;

namespace {
std::vector<float> v2(float a, float b) { return {a, b}; }
}  // namespace

// ---- Engine / index behaviour --------------------------------------------

TEST(Edge, WrongDimQueryThrows) {
    Engine flat(EngineConfig{4, MetricKind::L2, IndexKind::Flat});
    flat.insert("a", std::vector<float>{0, 0, 0, 0});
    EXPECT_THROW((void)flat.search(std::vector<float>{1, 2}, 5), std::invalid_argument);

    Engine hnsw(EngineConfig{4, MetricKind::L2, IndexKind::Hnsw});
    hnsw.insert("a", std::vector<float>{0, 0, 0, 0});
    EXPECT_THROW((void)hnsw.search(std::vector<float>{1, 2}, 5, nullptr, 16), std::invalid_argument);
}

TEST(Edge, WrongDimBatchQueryThrows) {
    Engine eng(EngineConfig{4, MetricKind::L2});
    eng.insert("a", std::vector<float>{0, 0, 0, 0});
    const std::vector<std::vector<float>> qs{{0, 0, 0, 0}, {1, 2}};  // 2nd query has wrong dim
    EXPECT_THROW((void)eng.search_batch(qs, 5), std::invalid_argument);
}

TEST(Edge, CompactReclaimsTombstonesAndKeepsData) {
    Engine eng(EngineConfig{2, MetricKind::L2, IndexKind::Flat});
    eng.insert("a", v2(0, 0));
    eng.insert("b", v2(1, 0));
    eng.insert("c", v2(2, 0));
    eng.erase("b");
    EXPECT_EQ(eng.store().size(), 2U);
    EXPECT_EQ(eng.store().slot_count(), 3U);  // tombstone slot still allocated

    eng.compact();
    EXPECT_EQ(eng.store().size(), 2U);
    EXPECT_EQ(eng.store().slot_count(), 2U);  // reclaimed
    EXPECT_TRUE(eng.store().resolve("a").has_value());
    EXPECT_TRUE(eng.store().resolve("c").has_value());
    EXPECT_FALSE(eng.store().resolve("b").has_value());

    const auto hits = eng.search(v2(0, 0), 5);
    ASSERT_EQ(hits.size(), 2U);
    EXPECT_EQ(*eng.store().resolve("a"), hits[0].id);  // still searchable, correct order
}

TEST(Edge, CompactRebuildsHnswSearchable) {
    Engine eng(EngineConfig{4, MetricKind::L2, IndexKind::Hnsw});
    eng.insert("a", std::vector<float>{0, 0, 0, 0});
    eng.insert("b", std::vector<float>{1, 1, 1, 1});
    eng.insert("c", std::vector<float>{5, 5, 5, 5});
    eng.erase("b");

    eng.compact();
    EXPECT_EQ(eng.store().slot_count(), 2U);  // b reclaimed
    const auto hits = eng.search(std::vector<float>{0, 0, 0, 0}, 1, nullptr, 16);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(*eng.store().resolve("a"), hits[0].id);  // graph rebuilt + searchable
}

TEST(Edge, FilterOnMissingKeyReturnsEmpty) {
    Engine eng(EngineConfig{2, MetricKind::L2});
    eng.insert("a", v2(0, 0));
    eng.insert("b", v2(1, 0));
    const auto f = Filter::eq("nonexistent", std::string("x"));
    EXPECT_TRUE(eng.search(v2(0, 0), 5, &f).empty());
}

TEST(Edge, UpdateIsReflectedInSearch) {
    Engine eng(EngineConfig{2, MetricKind::L2});  // Flat = exact
    eng.insert("a", v2(0, 0));
    eng.insert("b", v2(10, 0));

    auto hits = eng.search(v2(0, 0), 1);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(*eng.store().resolve("a"), hits[0].id);  // a is nearest

    eng.update("a", v2(20, 0));  // move a far away
    hits = eng.search(v2(0, 0), 1);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(*eng.store().resolve("b"), hits[0].id);  // now b is nearest
}

TEST(Edge, DeleteAllThenSearchIsEmpty_Flat) {
    Engine eng(EngineConfig{2, MetricKind::L2, IndexKind::Flat});
    eng.insert("a", v2(0, 0));
    eng.insert("b", v2(1, 0));
    eng.erase("a");
    eng.erase("b");
    EXPECT_EQ(eng.store().size(), 0U);
    EXPECT_TRUE(eng.search(v2(0, 0), 5).empty());
}

TEST(Edge, DeleteAllThenSearchIsEmpty_Hnsw) {
    Engine eng(EngineConfig{2, MetricKind::L2, IndexKind::Hnsw});
    eng.insert("a", v2(0, 0));
    eng.insert("b", v2(1, 0));
    eng.erase("a");
    eng.erase("b");
    EXPECT_TRUE(eng.search(v2(0, 0), 5, nullptr, 32).empty());
}

TEST(Edge, ZeroVectorCosineQueryIsSafeAndBounded) {
    Engine eng(EngineConfig{3, MetricKind::Cosine});
    eng.insert("a", std::vector<float>{1, 0, 0});
    eng.insert("b", std::vector<float>{0, 1, 0});

    const auto hits = eng.search(std::vector<float>{0, 0, 0}, 5);  // degenerate query
    ASSERT_EQ(hits.size(), 2U);
    for (const auto& h : hits) {
        EXPECT_GE(h.score, 0.0F);  // never negative
        EXPECT_LE(h.score, 2.0F);  // cosine distance bounded
    }
}

TEST(Edge, KZeroAndKHugeOnBothIndexes) {
    for (IndexKind kind : {IndexKind::Flat, IndexKind::Hnsw}) {
        EngineConfig cfg{2, MetricKind::L2, kind};
        Engine       eng(cfg);
        eng.insert("a", v2(0, 0));
        eng.insert("b", v2(1, 0));
        EXPECT_TRUE(eng.search(v2(0, 0), 0, nullptr, 16).empty());      // k = 0
        EXPECT_EQ(eng.search(v2(0, 0), 1000, nullptr, 16).size(), 2U);  // k >> corpus
    }
}

// ---- RAG behaviour --------------------------------------------------------

TEST(Edge, ReingestReplacesStaleChunks) {
    MockEmbeddingModel model(64);
    RagEngine          rag(model, RagConfig{ChunkConfig{/*window=*/4, /*overlap=*/1}});

    std::string longtext;
    for (int i = 0; i < 20; ++i) longtext += "alpha" + std::to_string(i) + " ";
    rag.ingest(Document{"d", longtext});
    EXPECT_GT(rag.engine().store().size(), 1U);  // several chunks

    rag.ingest(Document{"d", "tiny replacement text"});  // 3 tokens -> 1 chunk
    EXPECT_EQ(rag.document_count(), 1U);
    EXPECT_EQ(rag.engine().store().size(), 1U);  // stale chunks gone, not accumulated

    // The old chunks must not be retrievable any more.
    const auto hits = rag.retrieve("alpha5", 5);
    for (const auto& h : hits) EXPECT_EQ(h.text.find("alpha"), std::string::npos);
}

TEST(Edge, DocumentCountTracksDistinctDocs) {
    MockEmbeddingModel model(32);
    RagEngine          rag(model);
    rag.ingest(Document{"d1", "first document about vectors"});
    rag.ingest(Document{"d2", "second document about graphs"});
    rag.ingest(Document{"d1", "first document re-ingested"});  // same id again
    EXPECT_EQ(rag.document_count(), 2U);
}

TEST(Edge, IngestEmptyTextIsNoop) {
    MockEmbeddingModel model(32);
    RagEngine          rag(model);
    EXPECT_EQ(rag.ingest(Document{"empty", "   "}), 0U);
    EXPECT_EQ(rag.document_count(), 0U);
    EXPECT_EQ(rag.engine().store().size(), 0U);
}

TEST(Edge, RemoveDocumentErasesChunks) {
    MockEmbeddingModel model(64);
    RagEngine          rag(model, RagConfig{ChunkConfig{4, 1}});

    rag.ingest(Document{"d1", "the quick brown fox jumps over the lazy dog again"});
    rag.ingest(Document{"d2", "vector databases use nearest neighbour search"});
    const std::size_t before = rag.engine().store().size();
    ASSERT_GT(before, 0U);
    ASSERT_EQ(rag.document_count(), 2U);

    EXPECT_TRUE(rag.remove("d1"));
    EXPECT_EQ(rag.document_count(), 1U);
    EXPECT_LT(rag.engine().store().size(), before);  // d1's chunks gone

    // d1 content no longer retrievable; d2 still is.
    for (const auto& h : rag.retrieve("quick brown fox", 5)) EXPECT_NE(h.doc_id, "d1");
    EXPECT_FALSE(rag.retrieve("vector databases", 5).empty());
}

TEST(Edge, RemoveMissingDocumentReturnsFalse) {
    MockEmbeddingModel model(32);
    RagEngine          rag(model);
    EXPECT_FALSE(rag.remove("ghost"));
}

TEST(Edge, ReingestAfterRemoveStartsFresh) {
    MockEmbeddingModel model(32);
    RagEngine          rag(model);
    rag.ingest(Document{"d", "alpha beta gamma"});
    EXPECT_TRUE(rag.remove("d"));
    EXPECT_EQ(rag.document_count(), 0U);
    EXPECT_EQ(rag.ingest(Document{"d", "delta epsilon"}), 1U);  // clean re-add
    EXPECT_EQ(rag.document_count(), 1U);
}

TEST(Edge, RetrieveOnEmptyCorpusIsEmpty) {
    MockEmbeddingModel model(32);
    RagEngine          rag(model);
    EXPECT_TRUE(rag.retrieve("anything", 5).empty());
}
