#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace toyvdb {

/// A metadata value. Kept deliberately small: integers, reals, strings, bools.
using MetaValue = std::variant<std::int64_t, double, std::string, bool>;

/// Per-vector metadata: a flat key -> value map.
using Metadata = std::unordered_map<std::string, MetaValue>;

/// Three-way compare for two metadata values.
/// Returns <0, 0, >0, or nullopt if the values are not comparable.
/// Integers and doubles compare numerically across types; strings and bools
/// compare only against their own type.
inline std::optional<int> compare(const MetaValue& a, const MetaValue& b) {
    auto as_number = [](const MetaValue& v) -> std::optional<double> {
        if (auto p = std::get_if<std::int64_t>(&v)) return static_cast<double>(*p);
        if (auto p = std::get_if<double>(&v)) return *p;
        return std::nullopt;
    };

    const auto na = as_number(a);
    const auto nb = as_number(b);
    if (na && nb) {
        if (*na < *nb) return -1;
        if (*na > *nb) return 1;
        return 0;
    }

    if (a.index() == b.index()) {
        if (auto p = std::get_if<std::string>(&a)) {
            const auto& rhs = std::get<std::string>(b);
            return (*p < rhs) ? -1 : (*p > rhs ? 1 : 0);
        }
        if (auto p = std::get_if<bool>(&a)) {
            const bool rhs = std::get<bool>(b);
            if (*p == rhs) return 0;
            return *p ? 1 : -1;
        }
    }
    return std::nullopt;  // incomparable (e.g. string vs number)
}

/// A composable metadata filter: a small expression tree.
///
/// Leaves are comparisons (`eq`/`ne`/`lt`/`le`/`gt`/`ge`) or set membership
/// (`in`); internal nodes combine children with `all_of` (AND), `any_of` (OR),
/// or `negate` (NOT). Build them with the named factories, e.g.
///
///   auto f = Filter::all_of({Filter::eq("lang", std::string("en")),
///                            Filter::ge("year", std::int64_t{2023})});
///
/// A missing key never matches a leaf (so NOT of a missing-key leaf is true).
class Filter {
public:
    enum class Cmp { Eq, Ne, Lt, Le, Gt, Ge };

    // --- Leaf factories ----------------------------------------------------
    static Filter eq(std::string key, MetaValue v) { return cmp(std::move(key), Cmp::Eq, std::move(v)); }
    static Filter ne(std::string key, MetaValue v) { return cmp(std::move(key), Cmp::Ne, std::move(v)); }
    static Filter lt(std::string key, MetaValue v) { return cmp(std::move(key), Cmp::Lt, std::move(v)); }
    static Filter le(std::string key, MetaValue v) { return cmp(std::move(key), Cmp::Le, std::move(v)); }
    static Filter gt(std::string key, MetaValue v) { return cmp(std::move(key), Cmp::Gt, std::move(v)); }
    static Filter ge(std::string key, MetaValue v) { return cmp(std::move(key), Cmp::Ge, std::move(v)); }

    static Filter in(std::string key, std::vector<MetaValue> values) {
        Filter f;
        f.kind_ = Kind::In;
        f.key_ = std::move(key);
        f.set_ = std::move(values);
        return f;
    }

    // --- Composite factories ----------------------------------------------
    static Filter all_of(std::vector<Filter> children) {
        Filter f;
        f.kind_ = Kind::And;
        f.children_ = std::move(children);
        return f;
    }
    static Filter any_of(std::vector<Filter> children) {
        Filter f;
        f.kind_ = Kind::Or;
        f.children_ = std::move(children);
        return f;
    }
    static Filter negate(Filter child) {
        Filter f;
        f.kind_ = Kind::Not;
        f.children_.push_back(std::move(child));
        return f;
    }

    [[nodiscard]] bool matches(const Metadata& m) const {
        switch (kind_) {
            case Kind::Cmp: {
                const auto it = m.find(key_);
                if (it == m.end()) return false;
                const auto c = compare(it->second, value_);
                if (!c) return false;
                return eval_cmp(*c, cmp_);
            }
            case Kind::In: {
                const auto it = m.find(key_);
                if (it == m.end()) return false;
                for (const auto& v : set_) {
                    const auto c = compare(it->second, v);
                    if (c && *c == 0) return true;
                }
                return false;
            }
            case Kind::And:
                for (const auto& ch : children_) {
                    if (!ch.matches(m)) return false;
                }
                return true;
            case Kind::Or:
                for (const auto& ch : children_) {
                    if (ch.matches(m)) return true;
                }
                return false;
            case Kind::Not:
                return !children_.front().matches(m);
        }
        return false;
    }

private:
    enum class Kind { Cmp, In, And, Or, Not };

    static Filter cmp(std::string key, Cmp op, MetaValue v) {
        Filter f;
        f.kind_ = Kind::Cmp;
        f.key_ = std::move(key);
        f.cmp_ = op;
        f.value_ = std::move(v);
        return f;
    }

    static bool eval_cmp(int c, Cmp op) {
        switch (op) {
            case Cmp::Eq: return c == 0;
            case Cmp::Ne: return c != 0;
            case Cmp::Lt: return c < 0;
            case Cmp::Le: return c <= 0;
            case Cmp::Gt: return c > 0;
            case Cmp::Ge: return c >= 0;
        }
        return false;
    }

    Kind                   kind_ = Kind::Cmp;
    std::string            key_;
    Cmp                    cmp_ = Cmp::Eq;
    MetaValue              value_;
    std::vector<MetaValue> set_;       // for In
    std::vector<Filter>    children_;  // for And / Or / Not
};

}  // namespace toyvdb
