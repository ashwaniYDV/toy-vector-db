// Tiny end-to-end RAG demo: ingest a few documents, retrieve for a query, and
// print the grounded prompt that would be sent to an LLM.
//
// Uses the deterministic MockEmbeddingModel so it runs offline with no API keys.
// In a real deployment you would implement EmbeddingModel against an actual model.

#include <iostream>
#include <string>
#include <vector>

#include "toyvdb/rag.hpp"

int main() {
    using namespace toyvdb;

    MockEmbeddingModel model(256);

    RagConfig cfg;
    cfg.chunk = ChunkConfig{12, 4};  // small chunks so the demo shows several
    cfg.retrieve_k = 3;
    cfg.token_budget = 128;
    RagEngine rag(model, cfg);

    const std::vector<Document> corpus{
        {"hnsw", "HNSW is a graph based approximate nearest neighbour index that uses a "
                 "hierarchy of navigable small world layers to find close vectors quickly."},
        {"wal", "A write ahead log records every mutation before it is applied so the "
                "database can recover its state after a crash by replaying the log."},
        {"cosine", "Cosine similarity measures the angle between two vectors and is a common "
                   "distance metric for comparing text embeddings in a vector database."},
        {"rag", "Retrieval augmented generation retrieves relevant documents and injects them "
                "into the prompt so a language model can answer using grounded context."},
    };

    std::size_t total_chunks = 0;
    for (const auto& doc : corpus) total_chunks += rag.ingest(doc);
    std::cout << "ingested " << corpus.size() << " documents (" << total_chunks << " chunks)\n";
    std::cout << "indexed vectors: " << rag.engine().store().size() << "\n\n";

    const std::string query = "how does a database recover after a crash";
    std::cout << "query: \"" << query << "\"\n\n";

    std::cout << "top retrieved chunks:\n";
    for (const auto& hit : rag.retrieve(query, cfg.retrieve_k)) {
        std::cout << "  [" << hit.doc_id << "] dist=" << hit.score << "  " << hit.text << "\n";
    }

    std::cout << "\n--- assembled prompt ---\n";
    std::cout << rag.build_prompt(query) << "\n";
    std::cout << "------------------------\n";
    return 0;
}
