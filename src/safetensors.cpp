#include "safetensors.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// File layout: 8 byte little-endian header length, JSON header, raw tensor bytes.
// The header only ever holds a fixed shape of JSON, so this parser handles just that:
// {"name": {"dtype": "BF16", "shape": [a, b], "data_offsets": [start, end]}, "__metadata__": {...}}

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("safetensors: " + msg);
}

struct Header {
    const char* p;
    const char* end;

    void ws() {
        while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    }

    bool peek(char c) {
        ws();
        return p < end && *p == c;
    }

    void expect(char c) {
        if (!peek(c)) fail(std::string("expected '") + c + "' in header");
        ++p;
    }

    bool comma() {
        if (!peek(',')) return false;
        ++p;
        return true;
    }

    std::string str() {
        expect('"');
        std::string s;
        while (p < end && *p != '"') {
            if (*p == '\\') ++p;
            if (p < end) s += *p++;
        }
        expect('"');
        return s;
    }

    int64_t integer() {
        ws();
        char* stop = nullptr;
        long long v = std::strtoll(p, &stop, 10);
        if (stop == p) fail("expected integer in header");
        p = stop;
        return v;
    }

    std::vector<int64_t> ints() {
        std::vector<int64_t> v;
        expect('[');
        if (peek(']')) {
            ++p;
            return v;
        }
        do {
            v.push_back(integer());
        } while (comma());
        expect(']');
        return v;
    }

    // Skip any JSON value. Used for __metadata__ and unknown keys.
    void skip() {
        ws();
        if (p >= end) fail("header ended early");
        if (*p == '"') {
            str();
            return;
        }
        if (*p == '{' || *p == '[') {
            char close = (*p == '{') ? '}' : ']';
            ++p;
            if (peek(close)) {
                ++p;
                return;
            }
            do {
                if (close == '}') {
                    str();
                    expect(':');
                }
                skip();
            } while (comma());
            expect(close);
            return;
        }
        while (p < end && *p != ',' && *p != '}' && *p != ']') ++p;
    }
};

DType parse_dtype(const std::string& s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "I64") return DType::I64;
    if (s == "I32") return DType::I32;
    if (s == "U8") return DType::U8;
    fail("unsupported dtype " + s);
}

float bf16_to_f32(uint16_t h) {
    uint32_t bits = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

float f16_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // subnormal: shift until the hidden bit shows up
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                --exp;
            }
            bits = sign | (exp << 23) | ((mant & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

} // namespace

const char* dtype_name(DType d) {
    switch (d) {
    case DType::F32: return "F32";
    case DType::F16: return "F16";
    case DType::BF16: return "BF16";
    case DType::I64: return "I64";
    case DType::I32: return "I32";
    case DType::U8: return "U8";
    }
    return "?";
}

size_t dtype_size(DType d) {
    switch (d) {
    case DType::F32: return 4;
    case DType::F16: return 2;
    case DType::BF16: return 2;
    case DType::I64: return 8;
    case DType::I32: return 4;
    case DType::U8: return 1;
    }
    return 0;
}

int64_t Tensor::numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

SafeTensors::SafeTensors(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) fail("can't open " + path);

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        fail("can't stat " + path);
    }
    size_ = size_t(st.st_size);

    map_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map_ == MAP_FAILED) {
        map_ = nullptr;
        fail("mmap failed for " + path);
    }

    try {
        parse(static_cast<const uint8_t*>(map_));
    } catch (...) {
        munmap(map_, size_);
        map_ = nullptr;
        throw;
    }
}

SafeTensors::~SafeTensors() {
    if (map_) munmap(map_, size_);
}

void SafeTensors::parse(const uint8_t* base) {
    if (size_ < 8) fail("file too small");

    uint64_t header_len;
    std::memcpy(&header_len, base, 8);
    if (header_len > size_ - 8) fail("header length past end of file");

    const uint8_t* data = base + 8 + header_len;
    size_t data_len = size_ - 8 - header_len;

    Header h{reinterpret_cast<const char*>(base + 8), reinterpret_cast<const char*>(data)};
    h.expect('{');
    if (h.peek('}')) return;

    do {
        std::string name = h.str();
        h.expect(':');
        if (name == "__metadata__") {
            h.skip();
            continue;
        }

        Tensor t;
        std::vector<int64_t> offsets;
        bool have_dtype = false;

        h.expect('{');
        do {
            std::string key = h.str();
            h.expect(':');
            if (key == "dtype") {
                t.dtype = parse_dtype(h.str());
                have_dtype = true;
            } else if (key == "shape") {
                t.shape = h.ints();
            } else if (key == "data_offsets") {
                offsets = h.ints();
            } else {
                h.skip();
            }
        } while (h.comma());
        h.expect('}');

        if (!have_dtype || offsets.size() != 2) fail("bad entry for " + name);
        size_t begin = size_t(offsets[0]);
        size_t stop = size_t(offsets[1]);
        if (begin > stop || stop > data_len) fail("offsets out of range for " + name);

        t.data = data + begin;
        t.nbytes = stop - begin;
        if (t.nbytes != size_t(t.numel()) * dtype_size(t.dtype)) fail("size mismatch for " + name);

        tensors_.emplace(std::move(name), std::move(t));
    } while (h.comma());

    h.expect('}');
}

bool SafeTensors::has(const std::string& name) const {
    return tensors_.count(name) > 0;
}

const Tensor& SafeTensors::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) fail("no tensor named " + name);
    return it->second;
}

void to_f32(const Tensor& t, size_t start, size_t count, float* out) {
    if (start + count > size_t(t.numel())) fail("to_f32 range out of bounds");

    const uint8_t* src = t.data + start * dtype_size(t.dtype);
    for (size_t i = 0; i < count; ++i) {
        switch (t.dtype) {
        case DType::F32: {
            std::memcpy(&out[i], src + i * 4, 4);
            break;
        }
        case DType::F16: {
            uint16_t v;
            std::memcpy(&v, src + i * 2, 2);
            out[i] = f16_to_f32(v);
            break;
        }
        case DType::BF16: {
            uint16_t v;
            std::memcpy(&v, src + i * 2, 2);
            out[i] = bf16_to_f32(v);
            break;
        }
        case DType::I64: {
            int64_t v;
            std::memcpy(&v, src + i * 8, 8);
            out[i] = float(v);
            break;
        }
        case DType::I32: {
            int32_t v;
            std::memcpy(&v, src + i * 4, 4);
            out[i] = float(v);
            break;
        }
        case DType::U8:
            out[i] = float(src[i]);
            break;
        }
    }
}

std::vector<float> to_f32(const Tensor& t) {
    std::vector<float> out(size_t(t.numel()));
    to_f32(t, 0, out.size(), out.data());
    return out;
}

WeightFiles::WeightFiles(const std::string& model_dir) {
    std::vector<std::string> names;

    std::ifstream idx(model_dir + "/model.safetensors.index.json");
    if (idx) {
        // every shard name appears as a "....safetensors" string in the weight map
        std::stringstream ss;
        ss << idx.rdbuf();
        std::string text = ss.str();
        const std::string ext = ".safetensors\"";
        size_t at = 0;
        while ((at = text.find(ext, at)) != std::string::npos) {
            size_t start = text.rfind('"', at);
            std::string file = text.substr(start + 1, at + ext.size() - 2 - start);
            if (file.find('/') == std::string::npos &&
                std::find(names.begin(), names.end(), file) == names.end())
                names.push_back(file);
            at += ext.size();
        }
        if (names.empty()) fail("no shard names in " + model_dir + "/model.safetensors.index.json");
    } else {
        names.push_back("model.safetensors");
    }

    for (const auto& n : names) {
        files_.push_back(std::make_unique<SafeTensors>(model_dir + "/" + n));
        for (const auto& kv : files_.back()->tensors()) index_[kv.first] = &kv.second;
    }
}

bool WeightFiles::has(const std::string& name) const {
    return index_.count(name) > 0;
}

const Tensor& WeightFiles::get(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) fail("no tensor named " + name);
    return *it->second;
}
