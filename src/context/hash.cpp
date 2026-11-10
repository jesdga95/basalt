// src/context/hash.cpp - see include/basalt/context/hash.hpp. FIPS 180-4, plain scalar.
#include "basalt/context/hash.hpp"

namespace basalt::context {
namespace {
constexpr uint32_t kK[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

inline uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

inline uint32_t be32(const uint8_t* p) { return (uint32_t) p[0] << 24 | (uint32_t) p[1] << 16 | (uint32_t) p[2] << 8 | p[3]; }

inline void be32_put(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t) (v >> 24); p[1] = (uint8_t) (v >> 16); p[2] = (uint8_t) (v >> 8); p[3] = (uint8_t) v;
}
}  // namespace

Sha256::Sha256() { reset(); }

void Sha256::reset() {
    h_[0] = 0x6a09e667u; h_[1] = 0xbb67ae85u; h_[2] = 0x3c6ef372u; h_[3] = 0xa54ff53au;
    h_[4] = 0x510e527fu; h_[5] = 0x9b05688cu; h_[6] = 0x1f83d9abu; h_[7] = 0x5be0cd19u;
    len_bits_ = 0;
    buf_len_ = 0;
}

void Sha256::update_be32(uint32_t v) {
    const uint8_t b[4] = {(uint8_t) (v >> 24), (uint8_t) (v >> 16), (uint8_t) (v >> 8), (uint8_t) v};
    update(b, 4);
}

void Sha256::update_be64(uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = (uint8_t) (v >> (8 * (7 - i)));
    update(b, 8);
}

void Sha256::update(const void* data, size_t bytes) {
    const uint8_t* p = (const uint8_t*) data;
    len_bits_ += (uint64_t) bytes * 8;
    if (buf_len_) {
        const size_t take = bytes < 64 - buf_len_ ? bytes : 64 - buf_len_;
        for (size_t i = 0; i < take; ++i) buf_[buf_len_ + i] = p[i];
        buf_len_ += take; p += take; bytes -= take;
        if (buf_len_ < 64) return;
        compress(buf_);
        buf_len_ = 0;
    }
    while (bytes >= 64) {
        compress(p);
        p += 64; bytes -= 64;
    }
    for (size_t i = 0; i < bytes; ++i) buf_[i] = p[i];
    buf_len_ = bytes;
}

Digest Sha256::finish() {
    const uint64_t bits = len_bits_;
    const uint8_t pad = 0x80;
    update(&pad, 1);           // update() counts the pad's bits too; corrected below
    const uint8_t zero = 0x00;
    while (buf_len_ != 56) update(&zero, 1);
    len_bits_ = bits;          // the length field holds the message length, not the padded length
    uint8_t lenb[8];
    for (int i = 0; i < 8; ++i) lenb[i] = (uint8_t) (bits >> (8 * (7 - i)));
    update(lenb, 8);           // fills the block and compresses
    Digest d{};
    for (int i = 0; i < 8; ++i) be32_put(&d[i * 4], h_[i]);
    reset();                   // the hasher starts over, so one object serves many digests
    return d;
}

Digest Sha256::hash(const void* data, size_t bytes) {
    Sha256 s;
    s.update(data, bytes);
    return s.finish();
}

void Sha256::compress(const uint8_t* block) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = be32(block + 4 * i);
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + S1 + ch + kK[i] + w[i];
        const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
}

std::string to_hex(const Digest& d) {
    static const char* kHex = "0123456789abcdef";
    std::string s;
    s.reserve(64);
    for (uint8_t b : d) { s.push_back(kHex[b >> 4]); s.push_back(kHex[b & 15]); }
    return s;
}

bool from_hex(const std::string& hex, Digest& out) {
    if (hex.size() != 64) return false;
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    Digest d{};
    for (size_t i = 0; i < 32; ++i) {
        const int hi = nib(hex[2 * i]), lo = nib(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        d[i] = (uint8_t) (hi << 4 | lo);
    }
    out = d;
    return true;
}
}  // namespace basalt::context
