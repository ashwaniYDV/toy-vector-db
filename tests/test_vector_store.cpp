#include "toyvdb/vector_store.hpp"

#include <gtest/gtest.h>

#include <array>
#include <vector>

using namespace toyvdb;

namespace {
std::vector<float> vec3(float a, float b, float c) { return {a, b, c}; }
}

TEST(VectorStore, InsertGetRoundTrip) {
    VectorStore s(3);
    const auto v = vec3(1.0F, 2.0F, 3.0F);
    const InternalId id = s.insert("doc-1", v);

    EXPECT_EQ(s.size(), 1U);
    EXPECT_EQ(s.dim(), 3U);
    const auto got = s.get(id);
    ASSERT_EQ(got.size(), 3U);
    EXPECT_FLOAT_EQ(got[0], 1.0F);
    EXPECT_FLOAT_EQ(got[2], 3.0F);
}

TEST(VectorStore, ResolveMapsExternalToInternal) {
    VectorStore s(3);
    const InternalId id = s.insert("doc-1", vec3(1, 1, 1));
    const auto resolved = s.resolve("doc-1");
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(*resolved, id);
    EXPECT_FALSE(s.resolve("missing").has_value());
}

TEST(VectorStore, UpdateChangesVectorInPlace) {
    VectorStore s(3);
    const InternalId id = s.insert("doc-1", vec3(1, 1, 1));
    EXPECT_TRUE(s.update("doc-1", vec3(9, 8, 7)));

    const auto got = s.get(id);
    EXPECT_FLOAT_EQ(got[0], 9.0F);
    EXPECT_FLOAT_EQ(got[1], 8.0F);
    EXPECT_FLOAT_EQ(got[2], 7.0F);
    EXPECT_EQ(s.size(), 1U);  // update does not grow the store
}

TEST(VectorStore, UpdateMissingReturnsFalse) {
    VectorStore s(3);
    EXPECT_FALSE(s.update("nope", vec3(1, 2, 3)));
}

TEST(VectorStore, EraseTombstones) {
    VectorStore s(3);
    const InternalId id = s.insert("doc-1", vec3(1, 1, 1));
    EXPECT_TRUE(s.is_live(id));

    EXPECT_TRUE(s.erase("doc-1"));
    EXPECT_FALSE(s.is_live(id));
    EXPECT_EQ(s.size(), 0U);
    EXPECT_FALSE(s.resolve("doc-1").has_value());
    EXPECT_EQ(s.slot_count(), 1U);  // slot retained (tombstone, not freed)
}

TEST(VectorStore, EraseMissingReturnsFalse) {
    VectorStore s(3);
    EXPECT_FALSE(s.erase("ghost"));
}

TEST(VectorStore, ReinsertRevivesSameSlot) {
    VectorStore s(3);
    const InternalId id1 = s.insert("doc-1", vec3(1, 1, 1));
    s.erase("doc-1");
    const InternalId id2 = s.insert("doc-1", vec3(2, 2, 2));

    EXPECT_EQ(id1, id2);          // same internal slot reused
    EXPECT_EQ(s.slot_count(), 1U);  // no new slot allocated
    EXPECT_EQ(s.size(), 1U);
    EXPECT_TRUE(s.is_live(id2));
    EXPECT_FLOAT_EQ(s.get(id2)[0], 2.0F);
}

TEST(VectorStore, MetadataStoredAndRetrieved) {
    VectorStore s(3);
    Metadata m;
    m["lang"] = std::string("en");
    m["year"] = std::int64_t{2024};
    const InternalId id = s.insert("doc-1", vec3(1, 1, 1), m);

    const Metadata* got = s.metadata(id);
    ASSERT_NE(got, nullptr);
    ASSERT_TRUE(got->contains("lang"));
    EXPECT_EQ(std::get<std::string>(got->at("lang")), "en");
    EXPECT_EQ(std::get<std::int64_t>(got->at("year")), 2024);
}

TEST(VectorStore, DimMismatchThrows) {
    VectorStore s(3);
    const std::array<float, 2> bad{1.0F, 2.0F};
    EXPECT_THROW(s.insert("doc-1", bad), std::invalid_argument);
}

TEST(VectorStore, ZeroDimThrows) {
    EXPECT_THROW(VectorStore(0), std::invalid_argument);
}

TEST(VectorStore, UpdateReplacesMetadataWhenProvided) {
    VectorStore s(3);
    Metadata    m1;
    m1["lang"] = std::string("en");
    const InternalId id = s.insert("doc-1", vec3(1, 1, 1), m1);

    Metadata m2;
    m2["lang"] = std::string("fr");
    m2["year"] = std::int64_t{2024};
    EXPECT_TRUE(s.update("doc-1", vec3(2, 2, 2), m2));

    const Metadata* m = s.metadata(id);
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(std::get<std::string>(m->at("lang")), "fr");
    EXPECT_EQ(std::get<std::int64_t>(m->at("year")), 2024);
    EXPECT_FLOAT_EQ(s.get(id)[0], 2.0F);  // vector also updated
}

TEST(VectorStore, UpdateWithoutMetadataKeepsExisting) {
    VectorStore s(3);
    Metadata    m1;
    m1["lang"] = std::string("en");
    const InternalId id = s.insert("doc-1", vec3(1, 1, 1), m1);

    EXPECT_TRUE(s.update("doc-1", vec3(9, 9, 9)));  // vector-only update
    EXPECT_FLOAT_EQ(s.get(id)[0], 9.0F);
    EXPECT_EQ(std::get<std::string>(s.metadata(id)->at("lang")), "en");  // metadata unchanged
}

TEST(VectorStore, UpdateCanClearMetadataWithEmptyMap) {
    VectorStore s(3);
    Metadata    m1;
    m1["lang"] = std::string("en");
    const InternalId id = s.insert("doc-1", vec3(1, 1, 1), m1);

    // An engaged-but-empty map clears metadata (distinct from nullopt = unchanged).
    EXPECT_TRUE(s.update("doc-1", vec3(2, 2, 2), Metadata{}));
    ASSERT_NE(s.metadata(id), nullptr);
    EXPECT_TRUE(s.metadata(id)->empty());
}

TEST(VectorStore, UpdateDoesNotReviveTombstone) {
    VectorStore      s(3);
    const InternalId id = s.insert("doc-1", vec3(1, 1, 1));
    EXPECT_TRUE(s.erase("doc-1"));

    EXPECT_FALSE(s.update("doc-1", vec3(2, 2, 2)));  // UPDATE on deleted == no-op
    EXPECT_FALSE(s.is_live(id));                     // still dead, not resurrected
    EXPECT_EQ(s.size(), 0U);
    EXPECT_FALSE(s.resolve("doc-1").has_value());

    // insert() is the way to bring it back.
    EXPECT_EQ(s.insert("doc-1", vec3(7, 7, 7)), id);
    EXPECT_TRUE(s.is_live(id));
}

TEST(VectorStore, ApplyOpModelMatchesConvenienceApi) {
    VectorStore s(3);
    Op insert{OpType::Insert, "doc-1", vec3(5, 5, 5), {}};
    const InternalId id = s.apply(insert).value();
    EXPECT_EQ(s.size(), 1U);

    Op del{OpType::Delete, "doc-1", {}, {}};
    EXPECT_TRUE(s.apply(del).has_value());  // returns the deleted id
    EXPECT_FALSE(s.is_live(id));
    EXPECT_EQ(s.size(), 0U);

    EXPECT_FALSE(s.apply(del).has_value());  // second delete is a no-op
}
