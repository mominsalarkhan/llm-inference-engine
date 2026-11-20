#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

inline float bf16(uint16_t h) {
    uint32_t bits = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

template <typename F>
void parallel_rows(int rows, F&& fn) {
    static const int n_threads = std::max(1u, std::thread::hardware_concurrency());
    int n = std::min(n_threads, std::max(1, rows / 64));
    if (n == 1) {
        fn(0, rows);
        return;
    }

    std::vector<std::thread> pool;
    int chunk = (rows + n - 1) / n;
    for (int t = 0; t < n; ++t) {
        int r0 = t * chunk;
        int r1 = std::min(rows, r0 + chunk);
        if (r0 >= r1) break;
        pool.emplace_back([&fn, r0, r1] { fn(r0, r1); });
    }
    for (auto& th : pool) th.join();
}

} // namespace

void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float scale = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

void matvec(float* out, const Tensor& w, const float* x) {
    if (w.shape.size() != 2) throw std::runtime_error("matvec: weight is not 2D");
    int rows = int(w.shape[0]);
    int cols = int(w.shape[1]);

    if (w.dtype == DType::BF16) {
        const uint16_t* W = reinterpret_cast<const uint16_t*>(w.data);
        parallel_rows(rows, [&](int r0, int r1) {
            for (int r = r0; r < r1; ++r) {
                const uint16_t* row = W + size_t(r) * cols;
                float sum = 0.0f;
                for (int c = 0; c < cols; ++c) sum += bf16(row[c]) * x[c];
                out[r] = sum;
            }
        });
    } else if (w.dtype == DType::F32) {
        const float* W = reinterpret_cast<const float*>(w.data);
        parallel_rows(rows, [&](int r0, int r1) {
            for (int r = r0; r < r1; ++r) {
                const float* row = W + size_t(r) * cols;
                float sum = 0.0f;
                for (int c = 0; c < cols; ++c) sum += row[c] * x[c];
                out[r] = sum;
            }
        });
    } else {
        throw std::runtime_error(std::string("matvec: unsupported dtype ") + dtype_name(w.dtype));
    }
}

void rope(float* vec, int n_heads, int head_dim, int pos, float theta) {
    int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        // same float math as HF: inv_freq in fp32, then angle = pos * inv_freq
        float inv_freq = 1.0f / std::pow(theta, float(2 * i) / float(head_dim));
        float angle = float(pos) * inv_freq;
        float c = std::cos(angle);
        float s = std::sin(angle);
        for (int h = 0; h < n_heads; ++h) {
            float* v = vec + h * head_dim;
            float a = v[i];
            float b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = b * c + a * s;
        }
    }
}

void softmax(float* x, int n) {
    float mx = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    for (int i = 0; i < n; ++i) x[i] /= sum;
}
