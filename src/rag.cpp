#include "toyvdb/rag.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace toyvdb {

namespace {

std::vector<std::string> tokenize(std::string_view text) {
    std::vector<std::string> tokens;
    std::string              cur;
    for (const char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!cur.empty()) {
                tokens.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

std::uint64_t fnv1a(std::string_view s) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const char c : s) {
        h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
        h *= 1099511628211ULL;
    }
    return h;
}

}  // namespace

std::size_t approx_tokens(std::string_view text) { return tokenize(text).size(); }

std::vector<Chunk> chunk_text(std::string_view text, const ChunkConfig& cfg) {
    const std::vector<std::string> tokens = tokenize(text);
    std::vector<Chunk>             chunks;
    if (tokens.empty()) return chunks;

    const std::size_t window = (cfg.window == 0) ? 1 : cfg.window;
    const std::size_t overlap = (cfg.overlap >= window) ? (window - 1) : cfg.overlap;
    const std::size_t step = window - overlap;  // >= 1

    for (std::size_t start = 0; start < tokens.size(); start += step) {
        const std::size_t end = std::min(start + window, tokens.size());

        std::string joined;
        for (std::size_t i = start; i < end; ++i) {
            if (i > start) joined.push_back(' ');
            joined += tokens[i];
        }
        chunks.push_back(Chunk{std::move(joined), start, end});

        if (end == tokens.size()) break;  // last chunk reached the end
    }
    return chunks;
}

std::vector<float> MockEmbeddingModel::embed(std::string_view text) const {
    std::vector<float>             v(dim_, 0.0F);
    const std::vector<std::string> tokens = tokenize(text);

    for (const std::string& tok : tokens) {
        if (tok.size() < 3) continue;  // drop very short stopwords (a, is, of, to, ...)
        const std::uint64_t h = fnv1a(tok);
        const std::size_t   idx = static_cast<std::size_t>(h % dim_);
        const float         sign = ((h >> 63) & 1ULL) ? 1.0F : -1.0F;  // signed hashing
        v[idx] += sign;
    }

    double norm_sq = 0.0;
    for (const float x : v) norm_sq += static_cast<double>(x) * static_cast<double>(x);
    if (norm_sq > 0.0) {
        const float inv = static_cast<float>(1.0 / std::sqrt(norm_sq));
        for (float& x : v) x *= inv;
    }
    return v;
}

std::string PromptBuilder::build(std::string_view query,
                                 const std::vector<RetrievedChunk>& chunks) const {
    // Reserve budget for the query and the template scaffolding so the assembled
    // prompt stays within budget; greedily include the closest chunks first.
    constexpr std::size_t kScaffoldTokens = 24;
    const std::size_t     reserved = approx_tokens(query) + kScaffoldTokens;
    const std::size_t     available = (budget_ > reserved) ? budget_ - reserved : 0;

    std::string context;
    std::size_t used = 0;
    for (const RetrievedChunk& c : chunks) {
        const std::size_t cost = approx_tokens(c.text);
        if (used + cost > available) break;
        context += "- ";
        context += c.text;
        context.push_back('\n');
        used += cost;
    }

    std::string prompt = "You are a helpful assistant. Answer using only the context below.\n\n";
    prompt += "Context:\n";
    prompt += context;
    prompt += "\nQuestion: ";
    prompt += std::string(query);
    prompt += "\nAnswer:";
    return prompt;
}

RagEngine::RagEngine(const EmbeddingModel& model, RagConfig cfg)
    : model_(model), cfg_(cfg), engine_(EngineConfig{model.dim(), cfg.metric, cfg.index}) {}

std::size_t RagEngine::ingest(const Document& doc) {
    // Replace semantics: drop any chunks from a previous ingest of this doc id so
    // stale chunks never linger (e.g. when re-ingesting shorter text).
    if (const auto it = doc_chunks_.find(doc.id); it != doc_chunks_.end()) {
        for (std::size_t i = 0; i < it->second; ++i) {
            engine_.erase(doc.id + "#" + std::to_string(i));
        }
    }

    const std::vector<Chunk> chunks = chunk_text(doc.text, cfg_.chunk);
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        const std::vector<float> vec = model_.embed(chunks[i].text);

        Metadata meta;
        meta["doc"] = doc.id;
        meta["text"] = chunks[i].text;

        engine_.insert(doc.id + "#" + std::to_string(i), vec, std::move(meta));
    }

    if (chunks.empty()) {
        doc_chunks_.erase(doc.id);
    } else {
        doc_chunks_[doc.id] = chunks.size();
    }
    return chunks.size();
}

std::vector<RetrievedChunk> RagEngine::retrieve(std::string_view query, std::size_t k) const {
    const std::vector<float> qv = model_.embed(query);
    const auto               hits = engine_.search(qv, k);

    std::vector<RetrievedChunk> out;
    out.reserve(hits.size());
    for (const SearchResult& h : hits) {
        const Metadata* m = engine_.store().metadata(h.id);
        std::string     doc_id;
        std::string     text;
        if (m != nullptr) {
            if (const auto it = m->find("doc"); it != m->end()) {
                if (const auto* s = std::get_if<std::string>(&it->second)) doc_id = *s;
            }
            if (const auto it = m->find("text"); it != m->end()) {
                if (const auto* s = std::get_if<std::string>(&it->second)) text = *s;
            }
        }
        out.push_back(RetrievedChunk{std::move(doc_id), std::move(text), h.score});
    }
    return out;
}

std::string RagEngine::build_prompt(std::string_view query) const {
    const std::vector<RetrievedChunk> chunks = retrieve(query, cfg_.retrieve_k);
    return PromptBuilder(cfg_.token_budget).build(query, chunks);
}

bool RagEngine::remove(const std::string& doc_id) {
    const auto it = doc_chunks_.find(doc_id);
    if (it == doc_chunks_.end()) return false;
    for (std::size_t i = 0; i < it->second; ++i) {
        engine_.erase(doc_id + "#" + std::to_string(i));
    }
    doc_chunks_.erase(it);
    return true;
}

}  // namespace toyvdb
