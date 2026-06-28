// Recall@K + latency benchmark harness: HNSW vs the exact Flat oracle.
//
// Builds the same data into a FlatIndex (exact ground truth) and an HNSW index,
// then sweeps efSearch to trace the recall-vs-latency / recall-vs-QPS Pareto
// curve -- the canonical way ANN indexes are evaluated. Emits a human-readable
// table and a CSV block (paste into a plotting script).
//
// Usage: recall_harness [N] [dim] [Q] [k]

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "toyvdb/engine.hpp"

namespace {

using toyvdb::Dim;
using toyvdb::Engine;
using toyvdb::EngineConfig;
using toyvdb::IndexKind;
using toyvdb::InternalId;
using toyvdb::MetricKind;
using toyvdb::SearchResult;

using Clock = std::chrono::steady_clock;

std::vector<float> random_vector(std::mt19937& rng, Dim dim) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

// Clustered synthetic data: a point near one of `num_clusters` random centroids.
// Real embeddings have cluster structure / low intrinsic dimension; uniform-random
// vectors are a pathological worst case for graph ANN and understate recall.
std::vector<float> clustered_vector(std::mt19937& rng, const std::vector<std::vector<float>>& centroids,
                                    std::size_t which, float sigma) {
    std::normal_distribution<float> noise(0.0F, sigma);
    const auto& c = centroids[which];
    std::vector<float> v(c.size());
    for (std::size_t d = 0; d < c.size(); ++d) v[d] = c[d] + noise(rng);
    return v;
}

double recall_at_k(const std::vector<std::unordered_set<InternalId>>& truth,
                   const std::vector<std::vector<SearchResult>>& got, std::size_t k) {
    double sum = 0.0;
    for (std::size_t q = 0; q < truth.size(); ++q) {
        std::size_t hit = 0;
        for (const auto& r : got[q]) {
            if (truth[q].contains(r.id)) ++hit;
        }
        sum += static_cast<double>(hit) / static_cast<double>(k);
    }
    return truth.empty() ? 0.0 : sum / static_cast<double>(truth.size());
}

double percentile(std::vector<double> xs, double p) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    const auto idx = static_cast<std::size_t>(p * static_cast<double>(xs.size() - 1));
    return xs[idx];
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 20000;
    const Dim         dim = (argc > 2) ? static_cast<Dim>(std::strtoul(argv[2], nullptr, 10)) : 128;
    const std::size_t q = (argc > 3) ? std::strtoul(argv[3], nullptr, 10) : 200;
    const std::size_t k = (argc > 4) ? std::strtoul(argv[4], nullptr, 10) : 10;

    std::cout << "toy-vector-db recall harness (HNSW vs Flat oracle)\n"
              << "  N=" << n << " dim=" << dim << " queries=" << q << " k=" << k
              << "  (clustered synthetic data)\n\n";

    std::mt19937 rng(42);

    // Build a clustered dataset in memory, then index identical vectors in both.
    constexpr std::size_t num_clusters = 100;
    constexpr float       sigma = 0.10F;
    std::vector<std::vector<float>> centroids(num_clusters);
    for (auto& c : centroids) c = random_vector(rng, dim);

    std::vector<std::vector<float>> data(n);
    for (std::size_t i = 0; i < n; ++i) data[i] = clustered_vector(rng, centroids, i % num_clusters, sigma);

    Engine flat(EngineConfig{dim, MetricKind::L2, IndexKind::Flat});
    Engine hnsw(EngineConfig{dim, MetricKind::L2, IndexKind::Hnsw});

    // ---- Build both indexes on identical data ----------------------------
    const auto flat_build_start = Clock::now();
    for (std::size_t i = 0; i < n; ++i) flat.insert("v" + std::to_string(i), data[i]);
    const double flat_build_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - flat_build_start).count();

    const auto hnsw_build_start = Clock::now();
    for (std::size_t i = 0; i < n; ++i) hnsw.insert("v" + std::to_string(i), data[i]);
    const double hnsw_build_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - hnsw_build_start).count();

    // ---- Queries (near random clusters) + exact ground truth -------------
    std::vector<std::vector<float>> queries;
    queries.reserve(q);
    {
        std::uniform_int_distribution<std::size_t> pick(0, num_clusters - 1);
        for (std::size_t i = 0; i < q; ++i)
            queries.push_back(clustered_vector(rng, centroids, pick(rng), sigma));
    }

    std::vector<std::unordered_set<InternalId>> truth(q);
    std::vector<double>                         flat_latencies;
    flat_latencies.reserve(q);
    for (std::size_t i = 0; i < q; ++i) {
        const auto t0 = Clock::now();
        const auto hits = flat.search(queries[i], k);
        flat_latencies.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
        for (const auto& h : hits) truth[i].insert(h.id);
    }

    std::cout << "build:  flat=" << flat_build_ms << "ms  hnsw=" << hnsw_build_ms << "ms\n";
    std::cout << "flat (exact) p50 latency: " << percentile(flat_latencies, 0.50) << "us\n\n";

    // ---- efSearch sweep: the recall-vs-latency Pareto curve --------------
    const std::vector<std::size_t> ef_grid{10, 20, 40, 80, 160, 320};

    std::cout << "  efSearch | recall@" << k << " |   p50(us) |   p95(us) |     QPS\n";
    std::cout << "  ---------+----------+-----------+-----------+--------\n";

    std::cout.setf(std::ios::fixed);
    for (const std::size_t ef : ef_grid) {
        std::vector<std::vector<SearchResult>> got(q);
        std::vector<double>                    lat;
        lat.reserve(q);
        for (std::size_t i = 0; i < q; ++i) {
            const auto t0 = Clock::now();
            got[i] = hnsw.search(queries[i], k, nullptr, ef);
            lat.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
        }
        const double recall = recall_at_k(truth, got, k);
        double       mean = 0.0;
        for (double x : lat) mean += x;
        mean /= static_cast<double>(lat.empty() ? 1 : lat.size());
        const double qps = (mean > 0.0) ? 1e6 / mean : 0.0;

        std::cout.precision(4);
        std::cout << "  " << std::setw(8) << ef << " | " << std::setw(8) << recall << " | "
                  << std::setw(9) << std::setprecision(2) << percentile(lat, 0.50) << " | "
                  << std::setw(9) << percentile(lat, 0.95) << " | " << std::setw(7)
                  << std::setprecision(0) << qps << "\n";
    }

    // ---- Concurrency sweep: QPS vs threads (the multi-threaded read path) -
    // Replicate the query set to a larger batch for stable wall-clock timing.
    std::vector<std::vector<float>> batch;
    const std::size_t               batch_target = std::max<std::size_t>(q, 4000);
    batch.reserve(batch_target);
    for (std::size_t i = 0; i < batch_target; ++i) batch.push_back(queries[i % q]);

    const unsigned                 hw = std::max(1U, std::thread::hardware_concurrency());
    const std::vector<unsigned>    thread_grid{1, 2, 4, 8};
    const std::size_t              bench_ef = 80;

    std::cout << "\nconcurrency sweep (hardware_concurrency=" << hw << ", batch=" << batch_target
              << ", efSearch=" << bench_ef << ")\n";
    std::cout << "  threads |    QPS    | speedup\n";
    std::cout << "  --------+-----------+--------\n";

    double single_thread_qps = 0.0;
    for (const unsigned t : thread_grid) {
        const auto t0 = Clock::now();
        const auto out = hnsw.search_batch(batch, k, nullptr, bench_ef, t);
        const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        const double qps = (secs > 0.0) ? static_cast<double>(out.size()) / secs : 0.0;
        if (t == 1) single_thread_qps = qps;
        const double speedup = (single_thread_qps > 0.0) ? qps / single_thread_qps : 1.0;

        std::cout.precision(0);
        std::cout << "  " << std::setw(7) << t << " | " << std::setw(9) << qps << " | "
                  << std::setprecision(2) << std::setw(5) << speedup << "x\n";
        std::cout.precision(0);
    }

    return 0;
}
