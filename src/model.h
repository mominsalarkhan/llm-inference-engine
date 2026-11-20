#pragma once

#include <functional>
#include <string>
#include <vector>

#include "safetensors.h"

struct Config {
    int dim = 0;
    int hidden_dim = 0;
    int n_layers = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int vocab = 0;
    float rope_theta = 10000.0f;
    float norm_eps = 1e-5f;
};

// Reads the handful of numbers we need out of a HF config.json.
Config load_config(const std::string& path);

// Called with every intermediate result during forward(). Used by the golden checks.
using Tap = std::function<void(const std::string& name, const float* data, int n)>;

class Model {
public:
    // model_dir holds config.json and model.safetensors (a HF snapshot folder works).
    Model(const std::string& model_dir, int max_seq);

    const Config& config() const { return cfg_; }
    int max_seq() const { return max_seq_; }

    // Run one token at position pos. Uses and fills the KV cache. Returns logits.
    const std::vector<float>& forward(int token, int pos, const Tap& tap = nullptr);

private:
    struct Layer {
        std::vector<float> attn_norm;
        std::vector<float> mlp_norm;
        const Tensor* wq;
        const Tensor* wk;
        const Tensor* wv;
        const Tensor* wo;
        const Tensor* gate;
        const Tensor* up;
        const Tensor* down;
    };

    Config cfg_;
    SafeTensors weights_;
    int max_seq_;

    const Tensor* embed_;
    const Tensor* lm_head_;
    std::vector<float> final_norm_;
    std::vector<Layer> layers_;

    // KV cache: [layer][pos][kv_dim]
    std::vector<float> k_cache_;
    std::vector<float> v_cache_;

    // scratch buffers, allocated once
    std::vector<float> x_, xb_, q_, attn_, scores_, h1_, h2_, logits_;
};
