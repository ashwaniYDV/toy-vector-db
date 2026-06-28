#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <vector>

namespace toyvdb {

/// Durability policy for the append-only log.
enum class SyncPolicy {
    Always,       ///< fsync after every append. Durable, slow. Baseline.
    GroupCommit,  ///< caller batches appends, then calls sync() once. Fast + durable.
    Off,          ///< rely on the OS page cache. Fastest, not crash-durable.
};

/// Append-only byte log abstraction.
///
/// This is a storage *seam*: the WAL framing/recovery logic (week 7) lives above
/// this interface and does not care whether bytes land on local disk, a memory
/// buffer, or (later) an object store. `FileLogStore` is the only implementation
/// for now. Records are length-prefixed and CRC-checked by the implementation,
/// so `replay` only ever yields whole, intact records.
class LogStore {
public:
    virtual ~LogStore() = default;

    /// Append one record (frames + checksums it).
    virtual void append(std::span<const std::byte> record) = 0;

    /// Flush buffered/OS-cached data to stable storage.
    virtual void sync() = 0;

    /// Replay all intact records from the start, in append order. Stops at the
    /// first torn/corrupt record (expected only at the tail after a crash).
    virtual void replay(const std::function<void(std::span<const std::byte>)>& on_record) = 0;

    /// Discard all records (used after a snapshot captures the state).
    virtual void truncate() = 0;
};

/// Local-file implementation using POSIX fds so fsync semantics are explicit.
///
/// On-disk record layout (little-endian, native): [u32 crc][u32 len][len bytes].
/// crc covers the `len` payload bytes.
class FileLogStore final : public LogStore {
public:
    explicit FileLogStore(std::filesystem::path path, SyncPolicy policy = SyncPolicy::GroupCommit);
    ~FileLogStore() override;

    FileLogStore(const FileLogStore&) = delete;
    FileLogStore& operator=(const FileLogStore&) = delete;

    void append(std::span<const std::byte> record) override;
    void sync() override;
    void replay(const std::function<void(std::span<const std::byte>)>& on_record) override;
    void truncate() override;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
    SyncPolicy            policy_;
    int                   fd_ = -1;
};

}  // namespace toyvdb
