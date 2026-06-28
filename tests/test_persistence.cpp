#include "toyvdb/engine.hpp"
#include "toyvdb/op_codec.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace toyvdb;

namespace {
std::vector<float> v3(float a, float b, float c) { return {a, b, c}; }
}  // namespace

// ---- Op codec round-trips -------------------------------------------------

TEST(OpCodec, InsertRoundTrip) {
    Op op;
    op.type = OpType::Insert;
    op.ext_id = "doc-42";
    op.vec = {1.5F, -2.0F, 3.25F};
    Metadata m;
    m["lang"] = std::string("en");
    m["year"] = std::int64_t{2024};
    m["score"] = 0.75;
    m["ok"] = true;
    op.meta = m;

    const Op got = decode_op(encode_op(op));
    EXPECT_EQ(got.type, OpType::Insert);
    EXPECT_EQ(got.ext_id, "doc-42");
    EXPECT_EQ(got.vec, op.vec);
    ASSERT_TRUE(got.meta.has_value());
    EXPECT_EQ(std::get<std::string>(got.meta->at("lang")), "en");
    EXPECT_EQ(std::get<std::int64_t>(got.meta->at("year")), 2024);
    EXPECT_DOUBLE_EQ(std::get<double>(got.meta->at("score")), 0.75);
    EXPECT_TRUE(std::get<bool>(got.meta->at("ok")));
}

TEST(OpCodec, UpdateMetadataPresenceRoundTrips) {
    // Update with no metadata change -> meta stays nullopt across the codec.
    Op skip;
    skip.type = OpType::Update;
    skip.ext_id = "a";
    skip.vec = {1.0F, 2.0F};
    EXPECT_FALSE(decode_op(encode_op(skip)).meta.has_value());

    // Update that sets metadata to an empty map -> engaged-but-empty survives.
    Op clear = skip;
    clear.meta = Metadata{};
    const Op got = decode_op(encode_op(clear));
    ASSERT_TRUE(got.meta.has_value());
    EXPECT_TRUE(got.meta->empty());
}

TEST(OpCodec, DeleteRoundTrip) {
    Op op;
    op.type = OpType::Delete;
    op.ext_id = "gone";

    const Op got = decode_op(encode_op(op));
    EXPECT_EQ(got.type, OpType::Delete);
    EXPECT_EQ(got.ext_id, "gone");
    EXPECT_TRUE(got.vec.empty());
}

TEST(OpCodec, TruncatedRecordThrows) {
    Op op;
    op.type = OpType::Insert;
    op.ext_id = "x";
    op.vec = {1.0F, 2.0F};
    auto bytes = encode_op(op);
    bytes.resize(bytes.size() / 2);  // chop it in half
    EXPECT_THROW((void)decode_op(bytes), std::runtime_error);
}

TEST(OpCodec, UnknownOpTypeThrows) {
    std::vector<std::byte> bytes{std::byte{99}};  // 99 is not a valid OpType
    EXPECT_THROW((void)decode_op(bytes), std::runtime_error);
}

// ---- Persistence / recovery ----------------------------------------------

class Persistence : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("toyvdb_persist_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        std::filesystem::remove_all(dir_);
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }
    EngineConfig config() const {
        EngineConfig cfg{3, MetricKind::L2, IndexKind::Flat};
        cfg.persistence = PersistenceConfig{dir_, SyncPolicy::Always};
        return cfg;
    }
    std::filesystem::path dir_;
};

TEST_F(Persistence, RecoversInsertsAfterReopen) {
    {
        Engine e(config());
        EXPECT_TRUE(e.durable());
        e.insert("a", v3(1, 0, 0));
        e.insert("b", v3(0, 1, 0));
        e.insert("c", v3(0, 0, 1));
    }  // engine destroyed -> WAL closed

    Engine e2(config());  // recovers from WAL
    EXPECT_EQ(e2.store().size(), 3U);
    EXPECT_TRUE(e2.store().resolve("a").has_value());
    EXPECT_TRUE(e2.store().resolve("b").has_value());

    const auto hits = e2.search(v3(1, 0, 0), 1);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(*e2.store().resolve("a"), hits[0].id);
}

TEST_F(Persistence, RecoversMetadataAndSupportsFilteredSearch) {
    {
        Engine e(config());
        Metadata m;
        m["lang"] = std::string("fr");
        e.insert("a", v3(1, 0, 0));
        e.insert("b", v3(0, 1, 0), m);
    }

    Engine     e2(config());
    const auto only_fr = Filter::eq("lang", std::string("fr"));
    const auto hits = e2.search(v3(1, 0, 0), 5, &only_fr);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(*e2.store().resolve("b"), hits[0].id);
}

TEST_F(Persistence, DeletesArePersisted) {
    {
        Engine e(config());
        e.insert("a", v3(1, 0, 0));
        e.insert("b", v3(0, 1, 0));
        e.erase("a");
    }

    Engine e2(config());
    EXPECT_EQ(e2.store().size(), 1U);
    EXPECT_FALSE(e2.store().resolve("a").has_value());
    EXPECT_TRUE(e2.store().resolve("b").has_value());
}

TEST_F(Persistence, UpdatesArePersisted) {
    {
        Engine e(config());
        e.insert("a", v3(1, 1, 1));
        e.update("a", v3(9, 9, 9));
    }

    Engine     e2(config());
    const auto id = e2.store().resolve("a");
    ASSERT_TRUE(id.has_value());
    EXPECT_FLOAT_EQ(e2.store().get(*id)[0], 9.0F);
}

TEST_F(Persistence, SnapshotTruncatesWalAndRecovers) {
    const auto wal_path = dir_ / "wal.log";
    {
        Engine e(config());
        for (int i = 0; i < 10; ++i) e.insert("v" + std::to_string(i), v3(static_cast<float>(i), 0, 0));
        e.snapshot();  // fold state into snapshot blob, truncate WAL
        const auto after_snapshot = std::filesystem::file_size(wal_path);
        EXPECT_EQ(after_snapshot, 0U);
        e.insert("v10", v3(10, 0, 0));  // post-snapshot op goes to the fresh WAL
    }

    Engine e2(config());
    EXPECT_EQ(e2.store().size(), 11U);
    EXPECT_TRUE(e2.store().resolve("v0").has_value());   // from snapshot
    EXPECT_TRUE(e2.store().resolve("v10").has_value());  // from post-snapshot WAL
}

TEST_F(Persistence, RecoversDurablePrefixAfterTornTail) {
    {
        Engine e(config());
        e.insert("a", v3(1, 0, 0));
        e.insert("b", v3(0, 1, 0));
    }

    // Simulate a crash mid-append: a header promising bytes that never arrived.
    {
        std::ofstream out(dir_ / "wal.log", std::ios::binary | std::ios::app);
        const std::uint32_t crc = 0;
        const std::uint32_t len = 9999;
        out.write(reinterpret_cast<const char*>(&crc), 4);
        out.write(reinterpret_cast<const char*>(&len), 4);
    }

    Engine e2(config());  // replay stops at torn tail; durable prefix intact
    EXPECT_EQ(e2.store().size(), 2U);
    EXPECT_TRUE(e2.store().resolve("a").has_value());
    EXPECT_TRUE(e2.store().resolve("b").has_value());
}

TEST_F(Persistence, NoOpDeleteOrUpdateIsNotLogged) {
    const auto wal = dir_ / "wal.log";
    Engine     e(config());
    e.insert("a", v3(1, 0, 0));
    const auto size_after_insert = std::filesystem::file_size(wal);

    EXPECT_FALSE(e.erase("ghost"));                // id does not exist
    EXPECT_FALSE(e.update("ghost", v3(2, 2, 2)));  // id does not exist
    e.flush();

    // No record should have been appended for the two no-ops.
    EXPECT_EQ(std::filesystem::file_size(wal), size_after_insert);
}

TEST_F(Persistence, UpdatedMetadataIsPersisted) {
    {
        Engine   e(config());
        Metadata m;
        m["lang"] = std::string("en");
        e.insert("a", v3(1, 1, 1), m);

        Metadata m2;
        m2["lang"] = std::string("fr");
        e.update("a", v3(2, 2, 2), m2);  // replace vector + metadata
    }

    Engine     e2(config());
    const auto id = e2.store().resolve("a");
    ASSERT_TRUE(id.has_value());
    EXPECT_FLOAT_EQ(e2.store().get(*id)[0], 2.0F);
    EXPECT_EQ(std::get<std::string>(e2.store().metadata(*id)->at("lang")), "fr");
}

TEST_F(Persistence, CompactPersistsAndRecovers) {
    {
        Engine e(config());
        e.insert("a", v3(1, 0, 0));
        e.insert("b", v3(0, 1, 0));
        e.insert("c", v3(0, 0, 1));
        e.erase("b");
        e.compact();
        EXPECT_EQ(e.store().slot_count(), 2U);  // reclaimed in memory
    }

    Engine e2(config());  // recovers from the post-compaction snapshot
    EXPECT_EQ(e2.store().size(), 2U);
    EXPECT_EQ(e2.store().slot_count(), 2U);
    EXPECT_TRUE(e2.store().resolve("a").has_value());
    EXPECT_FALSE(e2.store().resolve("b").has_value());
    EXPECT_TRUE(e2.store().resolve("c").has_value());
}

TEST_F(Persistence, RecoveryRebuildsSearchableHnswGraph) {
    EngineConfig cfg{4, MetricKind::L2, IndexKind::Hnsw};
    cfg.persistence = PersistenceConfig{dir_, SyncPolicy::Always};
    {
        Engine e(cfg);
        e.insert("a", std::vector<float>{0, 0, 0, 0});
        e.insert("b", std::vector<float>{1, 1, 1, 1});
        e.insert("c", std::vector<float>{5, 5, 5, 5});
    }

    Engine     e2(cfg);  // HNSW graph rebuilt by replaying inserts
    const auto hits = e2.search(std::vector<float>{0, 0, 0, 0}, 1, nullptr, 32);
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(*e2.store().resolve("a"), hits[0].id);
}
