#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "model.h"

// How the big weight matrices are stored on the GPU.
//   BF16: as shipped, 2 bytes per weight
//   Q8:   int8 per weight + one fp32 scale per 128 weights (~1.03 bytes per weight)
//   Q4:   4 bits per weight + fp16 (scale, min) per 64 weights (~0.56 bytes per weight),
//         output layer kept at Q8
enum class Quant { BF16, Q8, Q4 };

// Same interface as Model, but weights, KV cache and all the math live on the GPU.
// This header stays plain C++ so main.cpp doesn't need nvcc.
class CudaModel {
public:
    CudaModel(const std::string& model_dir, int max_seq, Quant quant = Quant::BF16);
    ~CudaModel();

    const Config& config() const { return cfg_; }
    int max_seq() const { return max_seq_; }
    std::string device_name() const;
    size_t weight_bytes() const;

    const std::vector<float>& forward(int token, int pos, const Tap& tap = nullptr);

private:
    struct Impl;
    void enqueue(const Tap* tap);

    Config cfg_;
    int max_seq_;
    std::unique_ptr<Impl> impl_;
    std::vector<float> logits_;
};
