#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "engine.hpp"
#include "types.hpp"

namespace toyvdb {

// ---------------------------------------------------------------------------
// Documents & chunking
// ---------------------------------------------------------------------------

struct Document {
    std::string id;
    std::string text;
};

struct Chunk {
    std::string text;
    std::size_t start_token = 0;  // inclusive
    std::size_t end_token = 0;    // exclusive
};

struct ChunkConfig {
    std::size_t window = 64;   // tokens per chunk
    std::size_t overlap = 16;  // tokens shared with the previous chunk
};

/// Sliding-window chunker over whitespace tokens. Consecutive chunks overlap by
/// `overlap` tokens; a document shorter than `window` yields a single chunk.
[[nodiscard]] std::vector<Chunk> chunk_text(std::string_view text, const ChunkConfig& cfg);

/// Approximate token count (whitespace-delimited words). Good enough for budgeting.
[[nodiscard]] std::size_t approx_tokens(std::string_view text);

// ---------------------------------------------------------------------------
// Embeddings (the one piece intentionally pluggable, not built from scratch)
// ---------------------------------------------------------------------------

/// Adapter interface for an embedding model. Real deployments wrap an external
/// model/API here; tests use the deterministic MockEmbeddingModel below.
class EmbeddingModel {
public:
    virtual ~EmbeddingModel() = default;
    [[nodiscard]] virtual Dim                dim() const = 0;
    [[nodiscard]] virtual std::vector<float> embed(std::string_view text) const = 0;
};

/// Deterministic, network-free embedding via signed feature hashing of tokens
/// (FNV-1a) followed by L2 normalisation. Texts that share tokens land close in
/// cosine space, which is enough to exercise the full RAG path offline.
class MockEmbeddingModel final : public EmbeddingModel {
public:
    explicit MockEmbeddingModel(Dim dim = 128) : dim_(dim) {}
    [[nodiscard]] Dim                dim() const override { return dim_; }
    [[nodiscard]] std::vector<float> embed(std::string_view text) const override;

private:
    Dim dim_;
};

// ---------------------------------------------------------------------------
// Retrieval + prompt building
// ---------------------------------------------------------------------------

struct RetrievedChunk {
    std::string doc_id;
    std::string text;
    Score       score;  // distance: smaller == closer
};

/// Assembles a context-injected prompt under a token budget, reserving room for
/// the query and the template scaffolding so the whole prompt stays within budget.
class PromptBuilder {
public:
    explicit PromptBuilder(std::size_t token_budget = 512) : budget_(token_budget) {}

    [[nodiscard]] std::string build(std::string_view query,
                                    const std::vector<RetrievedChunk>& chunks) const;

private:
    std::size_t budget_;
};

struct RagConfig {
    ChunkConfig chunk{};
    MetricKind  metric = MetricKind::Cosine;
    IndexKind   index = IndexKind::Flat;
    std::size_t retrieve_k = 5;
    std::size_t token_budget = 512;
};

/// End-to-end RAG engine: ingest (chunk -> embed -> store), retrieve (embed query
/// -> vector search -> map back to chunk text), and build a grounded prompt.
class RagEngine {
public:
    RagEngine(const EmbeddingModel& model, RagConfig cfg = {});

    RagEngine(const RagEngine&) = delete;
    RagEngine& operator=(const RagEngine&) = delete;

    /// Chunk, embed, and store a document. Re-ingesting an existing doc id
    /// replaces its previous chunks (so stale chunks never linger). Returns the
    /// number of chunks ingested.
    std::size_t ingest(const Document& doc);

    [[nodiscard]] std::vector<RetrievedChunk> retrieve(std::string_view query, std::size_t k) const;

    /// Remove a document and all of its chunks. Returns true if it existed.
    bool remove(const std::string& doc_id);

    /// retrieve(retrieve_k) + token-budgeted prompt assembly.
    [[nodiscard]] std::string build_prompt(std::string_view query) const;

    /// Number of distinct documents currently ingested.
    [[nodiscard]] std::size_t document_count() const noexcept { return doc_chunks_.size(); }

    [[nodiscard]] const Engine& engine() const noexcept { return engine_; }

private:
    const EmbeddingModel& model_;
    RagConfig             cfg_;
    Engine                engine_;
    std::unordered_map<std::string, std::size_t> doc_chunks_;  // doc id -> chunk count
};

}  // namespace toyvdb
