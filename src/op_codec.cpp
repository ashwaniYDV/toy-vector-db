#include "toyvdb/op_codec.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace toyvdb {

namespace {

class Writer {
public:
    void raw(const void* p, std::size_t n) {
        const auto* b = static_cast<const std::byte*>(p);
        buf_.insert(buf_.end(), b, b + n);
    }
    void u8(std::uint8_t v) { raw(&v, 1); }
    void u32(std::uint32_t v) { raw(&v, 4); }
    void i64(std::int64_t v) { raw(&v, 8); }
    void f64(double v) { raw(&v, 8); }
    void str(const std::string& s) {
        u32(static_cast<std::uint32_t>(s.size()));
        raw(s.data(), s.size());
    }
    std::vector<std::byte> take() { return std::move(buf_); }

private:
    std::vector<std::byte> buf_;
};

class Reader {
public:
    explicit Reader(std::span<const std::byte> s) : s_(s) {}

    void raw(void* p, std::size_t n) {
        if (off_ + n > s_.size()) throw std::runtime_error("decode_op: truncated record");
        std::memcpy(p, s_.data() + off_, n);
        off_ += n;
    }
    std::uint8_t  u8() { std::uint8_t v = 0; raw(&v, 1); return v; }
    std::uint32_t u32() { std::uint32_t v = 0; raw(&v, 4); return v; }
    std::int64_t  i64() { std::int64_t v = 0; raw(&v, 8); return v; }
    double        f64() { double v = 0; raw(&v, 8); return v; }
    std::string   str() {
        const std::uint32_t n = u32();
        std::string out;
        out.resize(n);
        if (n > 0) raw(out.data(), n);
        return out;
    }

private:
    std::span<const std::byte> s_;
    std::size_t                off_ = 0;
};

void encode_metadata(Writer& w, const Metadata& meta) {
    w.u32(static_cast<std::uint32_t>(meta.size()));
    for (const auto& [key, val] : meta) {
        w.str(key);
        switch (val.index()) {
            case 0:  // int64
                w.u8(0);
                w.i64(std::get<std::int64_t>(val));
                break;
            case 1:  // double
                w.u8(1);
                w.f64(std::get<double>(val));
                break;
            case 2:  // string
                w.u8(2);
                w.str(std::get<std::string>(val));
                break;
            case 3:  // bool
                w.u8(3);
                w.u8(std::get<bool>(val) ? 1 : 0);
                break;
            default:
                throw std::runtime_error("encode_op: unknown metadata value type");
        }
    }
}

void decode_metadata(Reader& r, Metadata& meta) {
    const std::uint32_t count = r.u32();
    for (std::uint32_t i = 0; i < count; ++i) {
        std::string        key = r.str();
        const std::uint8_t t = r.u8();
        switch (t) {
            case 0: meta[key] = r.i64(); break;
            case 1: meta[key] = r.f64(); break;
            case 2: meta[key] = r.str(); break;
            case 3: meta[key] = (r.u8() != 0); break;
            default: throw std::runtime_error("decode_op: unknown metadata value type");
        }
    }
}

}  // namespace

std::vector<std::byte> encode_op(const Op& op) {
    Writer w;
    w.u8(static_cast<std::uint8_t>(op.type));
    w.str(op.ext_id);
    if (op.type != OpType::Delete) {
        w.u32(static_cast<std::uint32_t>(op.vec.size()));
        w.raw(op.vec.data(), op.vec.size() * sizeof(float));
        // Metadata is optional: a presence flag distinguishes "no metadata change"
        // (flag 0) from "set metadata" (flag 1, even to an empty map).
        if (op.meta.has_value()) {
            w.u8(1);
            encode_metadata(w, *op.meta);
        } else {
            w.u8(0);
        }
    }
    return w.take();
}

Op decode_op(std::span<const std::byte> bytes) {
    Reader r(bytes);
    Op     op;

    // Validate the op type at this trust boundary: bytes may be corrupt or forged,
    // so reject anything outside the known range rather than producing a bogus Op.
    const std::uint8_t type_byte = r.u8();
    if (type_byte < static_cast<std::uint8_t>(OpType::Insert) ||
        type_byte > static_cast<std::uint8_t>(OpType::Delete)) {
        throw std::runtime_error("decode_op: unknown op type");
    }
    op.type = static_cast<OpType>(type_byte);
    op.ext_id = r.str();
    if (op.type != OpType::Delete) {
        const std::uint32_t dim = r.u32();
        op.vec.resize(dim);
        if (dim > 0) r.raw(op.vec.data(), static_cast<std::size_t>(dim) * sizeof(float));
        if (r.u8() != 0) {  // metadata presence flag
            Metadata m;
            decode_metadata(r, m);
            op.meta = std::move(m);
        }  // else op.meta stays nullopt
    }
    return op;
}

}  // namespace toyvdb
