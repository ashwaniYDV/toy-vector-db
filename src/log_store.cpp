#include "toyvdb/log_store.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <system_error>

#include "toyvdb/crc32.hpp"

namespace toyvdb {

namespace {

void write_all(int fd, const std::byte* data, std::size_t n) {
    std::size_t off = 0;
    while (off < n) {
        const ssize_t w = ::write(fd, data + off, n - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "FileLogStore: write failed");
        }
        off += static_cast<std::size_t>(w);
    }
}

}  // namespace

FileLogStore::FileLogStore(std::filesystem::path path, SyncPolicy policy)
    : path_(std::move(path)), policy_(policy) {
    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_APPEND, 0644);
    if (fd_ < 0) {
        throw std::system_error(errno, std::generic_category(),
                                "FileLogStore: cannot open " + path_.string());
    }
}

FileLogStore::~FileLogStore() {
    if (fd_ >= 0) ::close(fd_);
}

void FileLogStore::append(std::span<const std::byte> record) {
    const std::uint32_t len = static_cast<std::uint32_t>(record.size());
    const std::uint32_t crc = crc32(record);

    // Header is [crc][len], then the payload. Single contiguous buffer so the
    // three pieces hit the fd together.
    std::array<std::byte, 8> header{};
    std::memcpy(header.data(), &crc, 4);
    std::memcpy(header.data() + 4, &len, 4);

    write_all(fd_, header.data(), header.size());
    write_all(fd_, record.data(), record.size());

    if (policy_ == SyncPolicy::Always) sync();
}

void FileLogStore::sync() {
    if (policy_ == SyncPolicy::Off) return;
    if (::fsync(fd_) < 0) {
        throw std::system_error(errno, std::generic_category(), "FileLogStore: fsync failed");
    }
}

void FileLogStore::replay(const std::function<void(std::span<const std::byte>)>& on_record) {
    // Read from the start with a fresh fd so the append offset is untouched.
    const int rfd = ::open(path_.c_str(), O_RDONLY);
    if (rfd < 0) {
        throw std::system_error(errno, std::generic_category(),
                                "FileLogStore: cannot reopen for replay");
    }

    auto read_exact = [&](std::byte* dst, std::size_t n) -> bool {
        std::size_t off = 0;
        while (off < n) {
            const ssize_t r = ::read(rfd, dst + off, n - off);
            if (r < 0) {
                if (errno == EINTR) continue;
                ::close(rfd);
                throw std::system_error(errno, std::generic_category(),
                                        "FileLogStore: read failed");
            }
            if (r == 0) return false;  // EOF (short read => torn tail)
            off += static_cast<std::size_t>(r);
        }
        return true;
    };

    std::vector<std::byte> payload;
    for (;;) {
        std::array<std::byte, 8> header{};
        if (!read_exact(header.data(), header.size())) break;  // clean EOF or torn header

        std::uint32_t crc = 0, len = 0;
        std::memcpy(&crc, header.data(), 4);
        std::memcpy(&len, header.data() + 4, 4);

        payload.resize(len);
        if (len > 0 && !read_exact(payload.data(), len)) break;  // torn payload at tail

        if (crc32(std::span<const std::byte>(payload.data(), len)) != crc) {
            break;  // corruption: stop here, everything before is durable
        }
        on_record(std::span<const std::byte>(payload.data(), len));
    }

    ::close(rfd);
}

void FileLogStore::truncate() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_APPEND, 0644);
    if (fd_ < 0) {
        throw std::system_error(errno, std::generic_category(),
                                "FileLogStore: cannot truncate " + path_.string());
    }
}

}  // namespace toyvdb
