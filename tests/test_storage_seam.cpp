#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "toyvdb/blob_store.hpp"
#include "toyvdb/log_store.hpp"

using namespace toyvdb;

namespace {

std::span<const std::byte> as_bytes(const std::string& s) {
    return std::span<const std::byte>(reinterpret_cast<const std::byte*>(s.data()), s.size());
}

std::string to_string(std::span<const std::byte> b) {
    return std::string(reinterpret_cast<const char*>(b.data()), b.size());
}

}  // namespace

class StorageSeam : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("toyvdb_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
                "_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        std::filesystem::create_directories(dir_);
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }
    std::filesystem::path dir_;
};

// ---- BlobStore ------------------------------------------------------------

TEST_F(StorageSeam, BlobRoundTrip) {
    FileBlobStore blobs(dir_ / "blobs");
    const std::string payload("snapshot\0binary\1bytes", 21);  // embedded NUL + binary

    EXPECT_FALSE(blobs.exists("snap.0"));
    blobs.put("snap.0", as_bytes(payload));
    EXPECT_TRUE(blobs.exists("snap.0"));

    const auto got = blobs.get("snap.0");
    EXPECT_EQ(to_string(std::span<const std::byte>(got)), payload);
}

TEST_F(StorageSeam, BlobGetMissingThrows) {
    FileBlobStore blobs(dir_ / "blobs");
    EXPECT_THROW((void)blobs.get("nope"), std::runtime_error);
}

// ---- LogStore -------------------------------------------------------------

TEST_F(StorageSeam, LogAppendReplayPreservesRecords) {
    const auto path = dir_ / "wal.log";
    {
        FileLogStore log(path, SyncPolicy::Always);
        log.append(as_bytes("alpha"));
        log.append(as_bytes("beta"));
        log.append(as_bytes("gamma"));
    }

    std::vector<std::string> recovered;
    {
        FileLogStore log(path, SyncPolicy::Off);
        log.replay([&](std::span<const std::byte> r) { recovered.push_back(to_string(r)); });
    }

    ASSERT_EQ(recovered.size(), 3U);
    EXPECT_EQ(recovered[0], "alpha");
    EXPECT_EQ(recovered[1], "beta");
    EXPECT_EQ(recovered[2], "gamma");
}

TEST_F(StorageSeam, LogReplayStopsAtTornTail) {
    const auto path = dir_ / "wal.log";
    {
        FileLogStore log(path, SyncPolicy::Always);
        log.append(as_bytes("one"));
        log.append(as_bytes("two"));
    }

    // Simulate a crash mid-append: a header claiming a payload that never arrived.
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        const std::uint32_t crc = 0;
        const std::uint32_t len = 1000;  // lie: no payload follows
        out.write(reinterpret_cast<const char*>(&crc), 4);
        out.write(reinterpret_cast<const char*>(&len), 4);
    }

    std::vector<std::string> recovered;
    FileLogStore log(path, SyncPolicy::Off);
    log.replay([&](std::span<const std::byte> r) { recovered.push_back(to_string(r)); });

    ASSERT_EQ(recovered.size(), 2U);  // torn tail dropped, durable prefix intact
    EXPECT_EQ(recovered[1], "two");
}

TEST_F(StorageSeam, LogReplayStopsAtCrcMismatch) {
    const auto path = dir_ / "wal.log";
    {
        FileLogStore log(path, SyncPolicy::Always);
        log.append(as_bytes("hello"));  // record 0: [8B header][5B payload] => bytes 0..12
        log.append(as_bytes("world"));  // record 1: header 13..20, payload 21..25
    }

    // Corrupt the first payload byte of record 1.
    {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(21);
        const char flipped = 'X';
        f.write(&flipped, 1);
    }

    std::vector<std::string> recovered;
    FileLogStore log(path, SyncPolicy::Off);
    log.replay([&](std::span<const std::byte> r) { recovered.push_back(to_string(r)); });

    ASSERT_EQ(recovered.size(), 1U);  // record 1 fails CRC, replay stops
    EXPECT_EQ(recovered[0], "hello");
}
