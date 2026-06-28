#include "toyvdb/flat_index.hpp"

#include <stdexcept>

namespace toyvdb {

std::unique_ptr<Index> make_flat_index(const VectorStore& store, MetricKind metric) {
    switch (metric) {
        case MetricKind::L2:
            return std::make_unique<FlatIndex<L2>>(store);
        case MetricKind::Cosine:
            return std::make_unique<FlatIndex<Cosine>>(store);
        case MetricKind::Dot:
            return std::make_unique<FlatIndex<Dot>>(store);
    }
    throw std::invalid_argument("make_flat_index: unknown metric");
}

}  // namespace toyvdb
