#include "toyvdb/hnsw_index.hpp"

#include <stdexcept>

namespace toyvdb {

std::unique_ptr<Index> make_hnsw_index(const VectorStore& store, MetricKind metric,
                                       HnswParams params) {
    switch (metric) {
        case MetricKind::L2:
            return std::make_unique<HnswIndex<L2>>(store, params);
        case MetricKind::Cosine:
            return std::make_unique<HnswIndex<Cosine>>(store, params);
        case MetricKind::Dot:
            return std::make_unique<HnswIndex<Dot>>(store, params);
    }
    throw std::invalid_argument("make_hnsw_index: unknown metric");
}

}  // namespace toyvdb
