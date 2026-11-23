#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model.h"

// Same interface as Model, but weights, KV cache and all the math live on the GPU.
// This header stays plain C++ so main.cpp doesn't need nvcc.
class CudaModel {
public:
    CudaModel(const std::string& model_dir, int max_seq);
    ~CudaModel();

    const Config& config() const { return cfg_; }
    int max_seq() const { return max_seq_; }
    std::string device_name() const;

    const std::vector<float>& forward(int token, int pos, const Tap& tap = nullptr);

private:
    struct Impl;
    void enqueue(const Tap* tap);
    Config cfg_;
    int max_seq_;
    std::unique_ptr<Impl> impl_;
    std::vector<float> logits_;
};
