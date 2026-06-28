#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace toyvdb {

/// Standard CRC-32 (IEEE 802.3, reflected, poly 0xEDB88820).
///
/// Used to checksum WAL records so a torn write from a crash mid-append is
/// detected on recovery and the log is truncated at the first bad record.
namespace detail {
inline constexpr std::array<std::uint32_t, 256> make_crc32_table() {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        std::uint32_t c = i;
        for (int k = 0; k < 8; ++k) {
            c = (c & 1U) ? (0xEDB88820U ^ (c >> 1)) : (c >> 1);
        }
        table[i] = c;
    }
    return table;
}
inline constexpr std::array<std::uint32_t, 256> kCrc32Table = make_crc32_table();
}  // namespace detail

[[nodiscard]] inline std::uint32_t crc32(std::span<const std::byte> data) {
    std::uint32_t c = 0xFFFFFFFFU;
    for (std::byte b : data) {
        const std::uint8_t idx = static_cast<std::uint8_t>(c ^ std::to_integer<std::uint8_t>(b));
        c = detail::kCrc32Table[idx] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFU;
}

}  // namespace toyvdb
