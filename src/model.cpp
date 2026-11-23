#include "model.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "ops.h"

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("can't open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Finds "key": <number> in a flat-ish JSON file. Good enough for HF config.json.
bool json_number(const std::string& text, const std::string& key, double& out) {
    size_t at = text.find("\"" + key + "\"");
    if (at == std::string::npos) return false;
    at = text.find(':', at);
    if (at == std::string::npos) return false;
    const char* start = text.c_str() + at + 1;
    char* stop = nullptr;
    double v = std::strtod(start, &stop);
    if (stop == start) return false;
    out = v;
    return true;
}

int require_int(const std::string& text, const std::string& key) {
    double v;
    if (!json_number(text, key, v)) throw std::runtime_error("config.json is missing " + key);
    return int(v);
}

std::vector<float> load_vec(const WeightFiles& st, const std::string& name) {
    return to_f32(st.get(name));
}

} // namespace

Config load_config(const std::string& path) {
    std::string text = read_file(path);
    Config c;
    c.dim = require_int(text, "hidden_size");
    c.hidden_dim = require_int(text, "intermediate_size");
    c.n_layers = require_int(text, "num_hidden_layers");
    c.n_heads = require_int(text, "num_attention_heads");
    c.vocab = require_int(text, "vocab_size");

    double v;
    c.n_kv_heads = json_number(text, "num_key_value_heads", v) ? int(v) : c.n_heads;
    c.head_dim = json_number(text, "head_dim", v) ? int(v) : c.dim / c.n_heads;
    if (json_number(text, "rope_theta", v)) c.rope_theta = float(v);
    if (json_number(text, "rms_norm_eps", v)) c.norm_eps = float(v);
    return c;
}

Model::Model(const std::string& model_dir, int max_seq)
    : cfg_(load_config(model_dir + "/config.json")),
      weights_(model_dir),
      max_seq_(max_seq) {
    const Config& c = cfg_;

    embed_ = &weights_.get("model.embed_tokens.weight");
    // some models tie the output head to the embedding table
    lm_head_ = weights_.has("lm_head.weight") ? &weights_.get("lm_head.weight") : embed_;
    final_norm_ = load_vec(weights_, "model.norm.weight");

    for (int i = 0; i < c.n_layers; ++i) {
        std::string p = "model.layers." + std::to_string(i) + ".";
        Layer l;
        l.attn_norm = load_vec(weights_, p + "input_layernorm.weight");
        l.mlp_norm = load_vec(weights_, p + "post_attention_layernorm.weight");
        l.wq = &weights_.get(p + "self_attn.q_proj.weight");
        l.wk = &weights_.get(p + "self_attn.k_proj.weight");
        l.wv = &weights_.get(p + "self_attn.v_proj.weight");
        l.wo = &weights_.get(p + "self_attn.o_proj.weight");
        l.gate = &weights_.get(p + "mlp.gate_proj.weight");
        l.up = &weights_.get(p + "mlp.up_proj.weight");
        l.down = &weights_.get(p + "mlp.down_proj.weight");
        layers_.push_back(std::move(l));
    }

    int q_dim = c.n_heads * c.head_dim;
    int kv_dim = c.n_kv_heads * c.head_dim;

    k_cache_.assign(size_t(c.n_layers) * max_seq_ * kv_dim, 0.0f);
    v_cache_.assign(size_t(c.n_layers) * max_seq_ * kv_dim, 0.0f);

    x_.resize(c.dim);
    xb_.resize(c.dim);
    q_.resize(q_dim);
    attn_.resize(q_dim);
    scores_.resize(max_seq_);
    h1_.resize(c.hidden_dim);
    h2_.resize(c.hidden_dim);
    logits_.resize(c.vocab);
}

const std::vector<float>& Model::forward(int token, int pos, const Tap& tap) {
    const Config& c = cfg_;
    if (pos >= max_seq_) throw std::runtime_error("position past max_seq");
    if (token < 0 || token >= c.vocab) throw std::runtime_error("token id out of range");

    auto emit = [&](const std::string& name, const float* data, int n) {
        if (tap) tap(name, data, n);
    };

    int dim = c.dim;
    int hd = c.head_dim;
    int kv_dim = c.n_kv_heads * hd;
    int group = c.n_heads / c.n_kv_heads; // query heads per kv head
    float* x = x_.data();
    float* xb = xb_.data();

    to_f32(*embed_, size_t(token) * dim, dim, x);
    emit("embed", x, dim);

    for (int li = 0; li < c.n_layers; ++li) {
        const Layer& L = layers_[li];
        std::string p = "layer" + std::to_string(li) + ".";

        // attention block
        rmsnorm(xb, x, L.attn_norm.data(), dim, c.norm_eps);
        emit(p + "attn_norm", xb, dim);

        float* k = k_cache_.data() + (size_t(li) * max_seq_ + pos) * kv_dim;
        float* v = v_cache_.data() + (size_t(li) * max_seq_ + pos) * kv_dim;
        matvec(q_.data(), *L.wq, xb);
        matvec(k, *L.wk, xb);
        matvec(v, *L.wv, xb);

        rope(q_.data(), c.n_heads, hd, pos, c.rope_theta);
        rope(k, c.n_kv_heads, hd, pos, c.rope_theta);

        float scale = 1.0f / std::sqrt(float(hd));
        for (int h = 0; h < c.n_heads; ++h) {
            const float* q = q_.data() + h * hd;
            int kvh = h / group;
            float* sc = scores_.data();

            for (int t = 0; t <= pos; ++t) {
                const float* kt = k_cache_.data() + (size_t(li) * max_seq_ + t) * kv_dim + kvh * hd;
                float dot = 0.0f;
                for (int i = 0; i < hd; ++i) dot += q[i] * kt[i];
                sc[t] = dot * scale;
            }
            softmax(sc, pos + 1);

            float* out = attn_.data() + h * hd;
            for (int i = 0; i < hd; ++i) out[i] = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                const float* vt = v_cache_.data() + (size_t(li) * max_seq_ + t) * kv_dim + kvh * hd;
                for (int i = 0; i < hd; ++i) out[i] += sc[t] * vt[i];
            }
        }

        matvec(xb, *L.wo, attn_.data());
        emit(p + "attn_out", xb, dim);
        for (int i = 0; i < dim; ++i) x[i] += xb[i];

        // MLP block: down(silu(gate(n)) * up(n))
        rmsnorm(xb, x, L.mlp_norm.data(), dim, c.norm_eps);
        emit(p + "mlp_norm", xb, dim);

        matvec(h1_.data(), *L.gate, xb);
        matvec(h2_.data(), *L.up, xb);
        for (int i = 0; i < c.hidden_dim; ++i) h1_[i] = silu(h1_[i]) * h2_[i];
        matvec(xb, *L.down, h1_.data());
        emit(p + "mlp_out", xb, dim);
        for (int i = 0; i < dim; ++i) x[i] += xb[i];

        emit(p + "out", x, dim);
    }

    rmsnorm(xb, x, final_norm_.data(), dim, c.norm_eps);
    emit("final_norm", xb, dim);

    matvec(logits_.data(), *lm_head_, xb);
    emit("logits", logits_.data(), c.vocab);
    return logits_;
}
