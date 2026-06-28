#include "toyvdb/rag.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <string>

using namespace toyvdb;

// ---- Chunker --------------------------------------------------------------

TEST(Chunker, ShortDocIsSingleChunk) {
    const auto chunks = chunk_text("hello world", ChunkConfig{64, 16});
    ASSERT_EQ(chunks.size(), 1U);
    EXPECT_EQ(chunks[0].text, "hello world");
    EXPECT_EQ(chunks[0].start_token, 0U);
    EXPECT_EQ(chunks[0].end_token, 2U);
}

TEST(Chunker, EmptyTextYieldsNoChunks) {
    EXPECT_TRUE(chunk_text("   ", ChunkConfig{8, 2}).empty());
}

TEST(Chunker, ConsecutiveChunksOverlap) {
    // 10 tokens, window 4, overlap 2 -> step 2: [0,4) [2,6) [4,8) [6,10) [8,10)
    std::string text;
    for (int i = 0; i < 10; ++i) text += "t" + std::to_string(i) + " ";
    const auto chunks = chunk_text(text, ChunkConfig{4, 2});

    ASSERT_GE(chunks.size(), 2U);
    EXPECT_EQ(chunks[0].start_token, 0U);
    EXPECT_EQ(chunks[0].end_token, 4U);
    EXPECT_EQ(chunks[1].start_token, 2U);  // shares 2 tokens with chunk 0
    EXPECT_EQ(chunks[1].end_token, 6U);
    EXPECT_EQ(chunks.back().end_token, 10U);  // covers the whole document
}

TEST(Chunker, OverlapGteWindowIsClamped) {
    // overlap >= window must not deadlock (step is forced >= 1).
    const auto chunks = chunk_text("a b c d e", ChunkConfig{2, 5});
    ASSERT_FALSE(chunks.empty());
    EXPECT_EQ(chunks.back().end_token, 5U);
}

// ---- Mock embedding -------------------------------------------------------

TEST(MockEmbedding, DeterministicAndNormalized) {
    MockEmbeddingModel model(64);
    const auto a = model.embed("the quick brown fox");
    const auto b = model.embed("the quick brown fox");
    EXPECT_EQ(a, b);  // deterministic
    EXPECT_EQ(a.size(), 64U);

    double norm = 0.0;
    for (float x : a) norm += static_cast<double>(x) * x;
    EXPECT_NEAR(std::sqrt(norm), 1.0, 1e-5);  // L2-normalised
}

TEST(MockEmbedding, SharedTokensAreCloserThanDisjoint) {
    MockEmbeddingModel model(128);
    const auto base = model.embed("machine learning vector database");
    const auto similar = model.embed("vector database machine learning systems");
    const auto unrelated = model.embed("banana apple orange fruit");

    auto cos = [](const std::vector<float>& x, const std::vector<float>& y) {
        double dot = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) dot += static_cast<double>(x[i]) * y[i];
        return dot;  // both normalised => dot == cosine similarity
    };
    EXPECT_GT(cos(base, similar), cos(base, unrelated));
}

// ---- End-to-end RAG -------------------------------------------------------

TEST(RagEngine, IngestReportsChunkCount) {
    MockEmbeddingModel model(64);
    RagEngine          rag(model, RagConfig{ChunkConfig{4, 1}});
    std::string        text;
    for (int i = 0; i < 12; ++i) text += "word" + std::to_string(i) + " ";
    const std::size_t chunks = rag.ingest(Document{"doc1", text});
    EXPECT_GE(chunks, 3U);
}

TEST(RagEngine, RetrievesRelevantChunk) {
    MockEmbeddingModel model(256);
    RagEngine          rag(model, RagConfig{ChunkConfig{6, 2}});

    rag.ingest(Document{"d1", "the mitochondria is the powerhouse of the cell"});
    rag.ingest(Document{"d2", "vector databases use approximate nearest neighbour search"});
    rag.ingest(Document{"d3", "the quick brown fox jumps over the lazy dog"});

    const auto hits = rag.retrieve("nearest neighbour search in vector databases", 1);
    ASSERT_FALSE(hits.empty());
    EXPECT_EQ(hits[0].doc_id, "d2");
    EXPECT_NE(hits[0].text.find("nearest"), std::string::npos);
}

TEST(RagEngine, BuildPromptIncludesQueryAndContext) {
    MockEmbeddingModel model(128);
    RagEngine          rag(model, RagConfig{ChunkConfig{8, 2}});
    rag.ingest(Document{"d1", "paris is the capital of france and a major city"});

    const std::string prompt = rag.build_prompt("what is the capital of france");
    EXPECT_NE(prompt.find("what is the capital of france"), std::string::npos);
    EXPECT_NE(prompt.find("Context:"), std::string::npos);
    EXPECT_NE(prompt.find("capital"), std::string::npos);  // context injected
}

TEST(PromptBuilderBudget, SmallBudgetIncludesFewerChunks) {
    std::vector<RetrievedChunk> chunks;
    for (int i = 0; i < 10; ++i) {
        chunks.push_back(RetrievedChunk{"d", "alpha beta gamma delta epsilon", 0.1F});
    }

    const std::string big = PromptBuilder{512}.build("q", chunks);
    const std::string small = PromptBuilder{40}.build("q", chunks);

    // Fewer "- " context bullets fit under the tighter budget.
    auto count_bullets = [](const std::string& s) {
        std::size_t n = 0, pos = 0;
        while ((pos = s.find("- ", pos)) != std::string::npos) {
            ++n;
            pos += 2;
        }
        return n;
    };
    EXPECT_LT(count_bullets(small), count_bullets(big));
}
