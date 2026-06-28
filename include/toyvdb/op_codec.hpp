#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "vector_store.hpp"  // Op, OpType, Metadata

namespace toyvdb {

/// Serialize a logical Op to a byte record (for the WAL / snapshots).
///
/// Layout (little-endian, native -- single-machine durability, not a portable
/// wire format):
///   [u8 type][u32 ext_len][ext bytes]
///   if type != Delete:
///     [u32 dim][dim * f32 vector]
///     [u32 meta_count]  then per entry: [u32 key_len][key][u8 val_type][value]
///       val_type 0=i64(8B) 1=f64(8B) 2=string(u32 len + bytes) 3=bool(1B)
[[nodiscard]] std::vector<std::byte> encode_op(const Op& op);

/// Inverse of encode_op. Throws std::runtime_error on a truncated/malformed record.
[[nodiscard]] Op decode_op(std::span<const std::byte> bytes);

}  // namespace toyvdb
