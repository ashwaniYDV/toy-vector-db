// REST API server for toy-vector-db.
//
// Wraps a vector Engine (raw numeric vectors) and a RagEngine (text) behind a
// small JSON HTTP API, and serves a single-page UI (server/static/index.html).
// Engine access is serialised with one mutex (the single-writer model), so the
// in-memory state stays consistent under cpp-httplib's per-request threads.
//
// Endpoints:
//   GET    /api/stats
//   POST   /api/vectors        {id, vector:[...], metadata?:{}}
//   POST   /api/search         {vector:[...], k?, ef?, filter?:{}}
//   DELETE /api/vectors/:id
//   POST   /api/documents      {id, text}
//   DELETE /api/documents/:id
//   POST   /api/rag/search     {query, k?}
//   POST   /api/rag/prompt     {query}

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "toyvdb/engine.hpp"
#include "toyvdb/rag.hpp"

using json = nlohmann::json;
using namespace toyvdb;

namespace {

MetaValue json_to_meta_value(const json& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number_integer()) return v.get<std::int64_t>();
    if (v.is_number_float()) return v.get<double>();
    return v.dump();  // fallback: keep as string
}

Metadata json_to_metadata(const json& obj) {
    Metadata m;
    if (obj.is_object())
        for (const auto& [k, v] : obj.items()) m[k] = json_to_meta_value(v);
    return m;
}

std::optional<Filter> json_to_filter(const json& obj) {
    if (!obj.is_object() || obj.empty()) return std::nullopt;
    std::vector<Filter> clauses;
    for (const auto& [k, v] : obj.items()) clauses.push_back(Filter::eq(k, json_to_meta_value(v)));
    return Filter::all_of(std::move(clauses));
}

std::vector<float> json_to_vector(const json& arr) {
    std::vector<float> v;
    for (const auto& x : arr) v.push_back(x.get<float>());
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    const int     port = (argc > 1) ? std::atoi(argv[1]) : 8080;
    constexpr Dim kVecDim = 4;  // raw-vector store dimension (small, easy to type)

    Engine             vec_engine(EngineConfig{kVecDim, MetricKind::L2, IndexKind::Hnsw});
    MockEmbeddingModel embed_model(256);
    RagConfig          rag_cfg;
    rag_cfg.chunk = ChunkConfig{40, 8};
    rag_cfg.retrieve_k = 5;
    RagEngine rag(embed_model, rag_cfg);

    std::mutex mu;

    httplib::Server svr;
    svr.set_mount_point("/", TOYVDB_STATIC_DIR);

    // Allow the page to call the API even when opened from file:// or another origin.
    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type"},
    });
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) { res.status = 204; });

    const auto fail = [](httplib::Response& res, int code, const std::string& msg) {
        res.status = code;
        res.set_content(json{{"error", msg}}.dump(), "application/json");
    };

    svr.Get("/api/stats", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(mu);
        const json j = {
            {"vector",
             {{"dim", kVecDim},
              {"metric", "l2"},
              {"index", std::string(vec_engine.index().name())},
              {"size", vec_engine.store().size()}}},
            {"rag",
             {{"dim", embed_model.dim()},
              {"documents", rag.document_count()},
              {"chunks", rag.engine().store().size()}}},
        };
        res.set_content(j.dump(2), "application/json");
    });

    svr.Post("/api/vectors", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const json               body = json::parse(req.body);
            const std::string        id = body.at("id").get<std::string>();
            const std::vector<float> vec = json_to_vector(body.at("vector"));
            Metadata meta = body.contains("metadata") ? json_to_metadata(body["metadata"]) : Metadata{};

            std::lock_guard<std::mutex> lk(mu);
            const InternalId iid = vec_engine.insert(id, vec, std::move(meta));
            res.set_content(json{{"ok", true}, {"id", id}, {"internal_id", iid}}.dump(),
                            "application/json");
        } catch (const std::exception& e) {
            fail(res, 400, e.what());
        }
    });

    svr.Post("/api/search", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const json               body = json::parse(req.body);
            const std::vector<float> q = json_to_vector(body.at("vector"));
            const std::size_t        k = body.value("k", std::size_t{10});
            const std::size_t        ef = body.value("ef", std::size_t{0});
            std::optional<Filter>    filter =
                body.contains("filter") ? json_to_filter(body["filter"]) : std::nullopt;

            std::lock_guard<std::mutex> lk(mu);
            const auto hits = vec_engine.search(q, k, filter ? &*filter : nullptr, ef);
            json       arr = json::array();
            for (const auto& h : hits)
                arr.push_back({{"id", vec_engine.store().external_id(h.id)}, {"score", h.score}});
            res.set_content(json{{"results", arr}}.dump(), "application/json");
        } catch (const std::exception& e) {
            fail(res, 400, e.what());
        }
    });

    svr.Delete(R"(/api/vectors/(.+))", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const std::string           id = req.matches[1].str();
            std::lock_guard<std::mutex> lk(mu);
            const bool                  existed = vec_engine.erase(id);
            res.set_content(json{{"ok", true}, {"existed", existed}}.dump(), "application/json");
        } catch (const std::exception& e) {
            fail(res, 400, e.what());
        }
    });

    svr.Post("/api/documents", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const json     body = json::parse(req.body);
            const Document doc{body.at("id").get<std::string>(), body.at("text").get<std::string>()};

            std::lock_guard<std::mutex> lk(mu);
            const std::size_t           chunks = rag.ingest(doc);
            res.set_content(json{{"ok", true}, {"chunks", chunks}}.dump(), "application/json");
        } catch (const std::exception& e) {
            fail(res, 400, e.what());
        }
    });

    svr.Delete(R"(/api/documents/(.+))", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const std::string           id = req.matches[1].str();
            std::lock_guard<std::mutex> lk(mu);
            const bool                  existed = rag.remove(id);
            res.set_content(json{{"ok", true}, {"existed", existed}}.dump(), "application/json");
        } catch (const std::exception& e) {
            fail(res, 400, e.what());
        }
    });

    svr.Post("/api/rag/search", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const json        body = json::parse(req.body);
            const std::string query = body.at("query").get<std::string>();
            const std::size_t k = body.value("k", std::size_t{5});

            std::lock_guard<std::mutex> lk(mu);
            const auto                  hits = rag.retrieve(query, k);
            json                        arr = json::array();
            for (const auto& h : hits)
                arr.push_back({{"doc_id", h.doc_id}, {"text", h.text}, {"score", h.score}});
            res.set_content(json{{"results", arr}}.dump(), "application/json");
        } catch (const std::exception& e) {
            fail(res, 400, e.what());
        }
    });

    svr.Post("/api/rag/prompt", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const json        body = json::parse(req.body);
            const std::string query = body.at("query").get<std::string>();

            std::lock_guard<std::mutex> lk(mu);
            res.set_content(json{{"prompt", rag.build_prompt(query)}}.dump(), "application/json");
        } catch (const std::exception& e) {
            fail(res, 400, e.what());
        }
    });

    std::cout << "toy-vector-db server listening on http://localhost:" << port << "\n";
    if (!svr.listen("0.0.0.0", port)) {
        std::cerr << "failed to bind port " << port << "\n";
        return 1;
    }
    return 0;
}
