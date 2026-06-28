#include "toyvdb/blob_store.hpp"

#include <fstream>
#include <stdexcept>

namespace toyvdb {

FileBlobStore::FileBlobStore(std::filesystem::path dir) : dir_(std::move(dir)) {
    std::filesystem::create_directories(dir_);
}

std::filesystem::path FileBlobStore::path_for(std::string_view key) const {
    return dir_ / std::filesystem::path(std::string(key));
}

void FileBlobStore::put(std::string_view key, std::span<const std::byte> data) {
    const auto p = path_for(key);
    std::filesystem::create_directories(p.parent_path());

    // Write to a temp file then rename for atomic publish (snapshot durability).
    const auto tmp = p.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("FileBlobStore: cannot open " + tmp);
        if (!data.empty()) {
            out.write(reinterpret_cast<const char*>(data.data()),
                      static_cast<std::streamsize>(data.size()));
        }
        if (!out) throw std::runtime_error("FileBlobStore: write failed for " + tmp);
    }
    std::filesystem::rename(tmp, p);
}

std::vector<std::byte> FileBlobStore::get(std::string_view key) {
    const auto p = path_for(key);
    std::ifstream in(p, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("FileBlobStore: no such key: " + std::string(key));

    const auto size = in.tellg();
    if (size < 0) throw std::runtime_error("FileBlobStore: tellg failed for " + p.string());
    in.seekg(0);

    std::vector<std::byte> buf(static_cast<std::size_t>(size));
    if (!buf.empty()) {
        in.read(reinterpret_cast<char*>(buf.data()), size);
        if (!in) throw std::runtime_error("FileBlobStore: read failed for " + p.string());
    }
    return buf;
}

bool FileBlobStore::exists(std::string_view key) {
    return std::filesystem::exists(path_for(key));
}

}  // namespace toyvdb
