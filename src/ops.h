#pragma once

#include <cmath>

#include "safetensors.h"

// out[i] = x[i] / rms(x) * w[i]
void rmsnorm(float* out, const float* x, const float* w, int n, float eps);

// out = W @ x, W is [rows, cols] in BF16 or F32. Split across threads by rows.
void matvec(float* out, const Tensor& w, const float* x);

// Rotary position embedding, HF "rotate_half" layout:
// dim i pairs with dim i + head_dim/2 inside each head.
void rope(float* vec, int n_heads, int head_dim, int pos, float theta);

void softmax(float* x, int n);

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}
