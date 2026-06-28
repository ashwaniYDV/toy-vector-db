#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace toyvdb {

/// Keyed blob storage abstraction, used for snapshots (week 7).
///
/// A second storage seam: snapshot bytes go through here, so the snapshot target
/// (local file now, object store later) is pluggable without touching the
/// snapshot/recovery logic.
class BlobStore {
public:
    virtual ~BlobStore() = default;

    virtual void put(std::string_view key, std::span<const std::byte> data) = 0;

    [[nodiscard]] virtual std::vector<std::byte> get(std::string_view key) = 0;

    [[nodiscard]] virtual bool exists(std::string_view key) = 0;
};

/// Local-directory implementation: each key maps to a file under `dir`.
class FileBlobStore final : public BlobStore {
public:
    explicit FileBlobStore(std::filesystem::path dir);

    void put(std::string_view key, std::span<const std::byte> data) override;
    [[nodiscard]] std::vector<std::byte> get(std::string_view key) override;
    [[nodiscard]] bool exists(std::string_view key) override;

private:
    [[nodiscard]] std::filesystem::path path_for(std::string_view key) const;
    std::filesystem::path dir_;
};

}  // namespace toyvdb
