#include "toyvdb/metadata.hpp"

#include <gtest/gtest.h>

#include <string>

using namespace toyvdb;

namespace {
Metadata doc(const std::string& lang, std::int64_t year, bool published) {
    Metadata m;
    m["lang"] = lang;
    m["year"] = year;
    m["published"] = published;
    return m;
}
}  // namespace

TEST(MetadataFilter, EqualityLeaf) {
    const auto m = doc("en", 2024, true);
    EXPECT_TRUE(Filter::eq("lang", std::string("en")).matches(m));
    EXPECT_FALSE(Filter::eq("lang", std::string("fr")).matches(m));
}

TEST(MetadataFilter, MissingKeyNeverMatchesLeaf) {
    const auto m = doc("en", 2024, true);
    EXPECT_FALSE(Filter::eq("author", std::string("x")).matches(m));
    // NOT of a missing-key leaf is therefore true.
    EXPECT_TRUE(Filter::negate(Filter::eq("author", std::string("x"))).matches(m));
}

TEST(MetadataFilter, NumericRangeAcrossIntAndDouble) {
    const auto m = doc("en", 2024, true);
    EXPECT_TRUE(Filter::ge("year", std::int64_t{2024}).matches(m));
    EXPECT_TRUE(Filter::ge("year", 2023.5).matches(m));   // int compared to double
    EXPECT_FALSE(Filter::gt("year", std::int64_t{2024}).matches(m));
    EXPECT_TRUE(Filter::lt("year", std::int64_t{2025}).matches(m));
}

TEST(MetadataFilter, BoolEquality) {
    const auto m = doc("en", 2024, true);
    EXPECT_TRUE(Filter::eq("published", true).matches(m));
    EXPECT_FALSE(Filter::eq("published", false).matches(m));
}

TEST(MetadataFilter, IncomparableTypesDoNotMatch) {
    const auto m = doc("en", 2024, true);
    // "lang" is a string; comparing against a number is incomparable -> false.
    EXPECT_FALSE(Filter::eq("lang", std::int64_t{1}).matches(m));
}

TEST(MetadataFilter, InMembership) {
    const auto m = doc("de", 2024, true);
    const auto langs = Filter::in("lang", {std::string("en"), std::string("de"), std::string("fr")});
    EXPECT_TRUE(langs.matches(m));
    EXPECT_FALSE(Filter::in("lang", {std::string("en"), std::string("fr")}).matches(m));
}

TEST(MetadataFilter, AndCombines) {
    const auto m = doc("en", 2024, true);
    const auto f = Filter::all_of({
        Filter::eq("lang", std::string("en")),
        Filter::ge("year", std::int64_t{2020}),
        Filter::eq("published", true),
    });
    EXPECT_TRUE(f.matches(m));

    const auto f2 = Filter::all_of({
        Filter::eq("lang", std::string("en")),
        Filter::ge("year", std::int64_t{2030}),  // fails
    });
    EXPECT_FALSE(f2.matches(m));
}

TEST(MetadataFilter, OrCombines) {
    const auto m = doc("en", 2024, true);
    const auto f = Filter::any_of({
        Filter::eq("lang", std::string("fr")),  // false
        Filter::eq("lang", std::string("en")),  // true
    });
    EXPECT_TRUE(f.matches(m));

    const auto none = Filter::any_of({
        Filter::eq("lang", std::string("fr")),
        Filter::eq("lang", std::string("de")),
    });
    EXPECT_FALSE(none.matches(m));
}

TEST(MetadataFilter, NestedComposition) {
    const auto m = doc("en", 2024, false);
    // (lang in {en,de}) AND NOT published AND year >= 2020
    const auto f = Filter::all_of({
        Filter::in("lang", {std::string("en"), std::string("de")}),
        Filter::negate(Filter::eq("published", true)),
        Filter::ge("year", std::int64_t{2020}),
    });
    EXPECT_TRUE(f.matches(m));
}
