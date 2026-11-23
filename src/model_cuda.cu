#include "model_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "safetensors.h"

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err_ = (call);                                                            \
        if (err_ != cudaSuccess)                                                              \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err_) + " at " + \
                                     __FILE__ + ":" + std::to_string(__LINE__));              \
    } while (0)

namespace {

// ---------- device helpers ----------

__device__ inline float bf16_to_f32(uint16_t h) {
    return __uint_as_float(uint32_t(h) << 16);
}

__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ inline float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// blockDim.x must be a multiple of 32. Every thread gets the result.
__device__ float block_sum(float v) {
    __shared__ float tmp[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads(); // tmp may still be in use from a previous call
    if (lane == 0) tmp[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < int(blockDim.x >> 5)) ? tmp[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) tmp[0] = v;
    }
    __syncthreads();
    return tmp[0];
}

__device__ float block_max(float v) {
    __shared__ float tmp[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) tmp[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < int(blockDim.x >> 5)) ? tmp[lane] : -INFINITY;
        v = warp_max(v);
        if (lane == 0) tmp[0] = v;
    }
    __syncthreads();
    return tmp[0];
}

// ---------- kernels ----------

__global__ void embed_kernel(float* out, const uint16_t* table, int token, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = bf16_to_f32(table[size_t(token) * dim + i]);
}

// launched with a single block
__global__ void rmsnorm_kernel(float* out, const float* x, const float* w, int n, float eps) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    float scale = 1.0f / sqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * scale * w[i];
}

// One warp per output row. Naive on purpose: this is the kernel step 6 replaces.
__global__ void matvec_bf16_kernel(float* out, const uint16_t* W, const float* x, int rows, int cols) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return; // whole warp exits together

    const uint16_t* w = W + size_t(row) * cols;
    float sum = 0.0f;
    for (int c = lane; c < cols; c += 32) sum += bf16_to_f32(w[c]) * x[c];
    sum = warp_sum(sum);
    if (lane == 0) out[row] = sum;
}

// HF rotate_half layout: dim i pairs with i + head_dim/2
__global__ void rope_kernel(float* vec, int n_heads, int head_dim, int pos, float theta) {
    int half = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heads * half) return;

    int h = idx / half;
    int i = idx % half;
    float inv_freq = 1.0f / powf(theta, float(2 * i) / float(head_dim));
    float angle = float(pos) * inv_freq;
    float s, c;
    sincosf(angle, &s, &c);

    float* v = vec + h * head_dim;
    float a = v[i];
    float b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = b * c + a * s;
}

// One block per query head. Scores for every past position live in shared memory.
__global__ void attention_kernel(float* out, const float* q, const float* k_cache, const float* v_cache,
                                 int pos, int head_dim, int kv_dim, int group, float scale) {
    extern __shared__ float sc[]; // pos + 1 floats

    int h = blockIdx.x;
    int kvh = h / group;
    const float* qh = q + h * head_dim;

    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        const float* kt = k_cache + size_t(t) * kv_dim + kvh * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) dot += qh[i] * kt[i];
        sc[t] = dot * scale;
    }
    __syncthreads();

    float m = -INFINITY;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) m = fmaxf(m, sc[t]);
    m = block_max(m);

    float s = 0.0f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        float e = expf(sc[t] - m);
        sc[t] = e;
        s += e;
    }
    s = block_sum(s); // also syncs, so every sc[t] is written before we read it below

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t <= pos; ++t) acc += sc[t] * v_cache[size_t(t) * kv_dim + kvh * head_dim + i];
        out[h * head_dim + i] = acc / s;
    }
}

__global__ void silu_mul_kernel(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = a[i];
        a[i] = g / (1.0f + expf(-g)) * b[i];
    }
}

__global__ void add_kernel(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

inline int blocks_for(int n, int threads) {
    return (n + threads - 1) / threads;
}

} // namespace

// ---------- host side ----------

struct CudaModel::Impl {
    struct Layer {
        float* attn_norm;
        float* mlp_norm;
        uint16_t* wq;
        uint16_t* wk;
        uint16_t* wv;
        uint16_t* wo;
        uint16_t* gate;
        uint16_t* up;
        uint16_t* down;
    };

    std::vector<void*> allocations;
    uint16_t* embed = nullptr;
    uint16_t* lm_head = nullptr;
    float* final_norm = nullptr;
    std::vector<Layer> layers;

    float* k_cache = nullptr;
    float* v_cache = nullptr;
    float *x, *xb, *q, *attn, *h1, *h2, *logits;

    std::vector<float> host_tmp; // for taps

    ~Impl() {
        for (void* p : allocations) cudaFree(p);
    }

    template <typename T>
    T* alloc(size_t count) {
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
        allocations.push_back(p);
        return static_cast<T*>(p);
    }

    // Upload a BF16 matrix as-is and check its shape.
    uint16_t* upload_bf16(const SafeTensors& st, const std::string& name, int64_t rows, int64_t cols) {
        const Tensor& t = st.get(name);
        if (t.dtype != DType::BF16) throw std::runtime_error(name + ": CUDA path only handles BF16 weights for now");
        if (t.shape.size() != 2 || t.shape[0] != rows || t.shape[1] != cols)
            throw std::runtime_error(name + ": unexpected shape");
        uint16_t* d = alloc<uint16_t>(size_t(t.numel()));
        CUDA_CHECK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
        return d;
    }

    // Norm weights are tiny, convert to fp32 on the host first.
    float* upload_f32(const SafeTensors& st, const std::string& name) {
        std::vector<float> v = to_f32(st.get(name));
        float* d = alloc<float>(v.size());
        CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
        return d;
    }
};

CudaModel::CudaModel(const std::string& model_dir, int max_seq)
    : cfg_(load_config(model_dir + "/config.json")), max_seq_(max_seq), impl_(new Impl) {
    const Config& c = cfg_;
    Impl& m = *impl_;

    int q_dim = c.n_heads * c.head_dim;
    int kv_dim = c.n_kv_heads * c.head_dim;

    // the mmap only needs to live while we copy to the GPU
    SafeTensors st(model_dir + "/model.safetensors");

    m.embed = m.upload_bf16(st, "model.embed_tokens.weight", c.vocab, c.dim);
    m.lm_head = st.has("lm_head.weight") ? m.upload_bf16(st, "lm_head.weight", c.vocab, c.dim) : m.embed;
    m.final_norm = m.upload_f32(st, "model.norm.weight");

    for (int i = 0; i < c.n_layers; ++i) {
        std::string p = "model.layers." + std::to_string(i) + ".";
        Impl::Layer l;
        l.attn_norm = m.upload_f32(st, p + "input_layernorm.weight");
        l.mlp_norm = m.upload_f32(st, p + "post_attention_layernorm.weight");
        l.wq = m.upload_bf16(st, p + "self_attn.q_proj.weight", q_dim, c.dim);
        l.wk = m.upload_bf16(st, p + "self_attn.k_proj.weight", kv_dim, c.dim);
        l.wv = m.upload_bf16(st, p + "self_attn.v_proj.weight", kv_dim, c.dim);
        l.wo = m.upload_bf16(st, p + "self_attn.o_proj.weight", c.dim, q_dim);
        l.gate = m.upload_bf16(st, p + "mlp.gate_proj.weight", c.hidden_dim, c.dim);
        l.up = m.upload_bf16(st, p + "mlp.up_proj.weight", c.hidden_dim, c.dim);
        l.down = m.upload_bf16(st, p + "mlp.down_proj.weight", c.dim, c.hidden_dim);
        m.layers.push_back(l);
    }

    size_t cache = size_t(c.n_layers) * max_seq_ * kv_dim;
    m.k_cache = m.alloc<float>(cache);
    m.v_cache = m.alloc<float>(cache);
    CUDA_CHECK(cudaMemset(m.k_cache, 0, cache * sizeof(float)));
    CUDA_CHECK(cudaMemset(m.v_cache, 0, cache * sizeof(float)));

    m.x = m.alloc<float>(c.dim);
    m.xb = m.alloc<float>(c.dim);
    m.q = m.alloc<float>(q_dim);
    m.attn = m.alloc<float>(q_dim);
    m.h1 = m.alloc<float>(c.hidden_dim);
    m.h2 = m.alloc<float>(c.hidden_dim);
    m.logits = m.alloc<float>(c.vocab);

    logits_.resize(c.vocab);
}

CudaModel::~CudaModel() = default;

std::string CudaModel::device_name() const {
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return "unknown";
    return prop.name;
}

const std::vector<float>& CudaModel::forward(int token, int pos, const Tap& tap) {
    const Config& c = cfg_;
    Impl& m = *impl_;
    if (pos >= max_seq_) throw std::runtime_error("position past max_seq");
    if (token < 0 || token >= c.vocab) throw std::runtime_error("token id out of range");

    // Only used by the golden check: copy a GPU buffer back and hand it to the tap.
    auto emit = [&](const std::string& name, const float* d, int n) {
        if (!tap) return;
        m.host_tmp.resize(n);
        CUDA_CHECK(cudaMemcpy(m.host_tmp.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
        tap(name, m.host_tmp.data(), n);
    };

    const int T = 256;
    int dim = c.dim;
    int hd = c.head_dim;
    int q_dim = c.n_heads * hd;
    int kv_dim = c.n_kv_heads * hd;
    int group = c.n_heads / c.n_kv_heads;
    float scale = 1.0f / std::sqrt(float(hd));

    auto matvec = [&](float* out, const uint16_t* w, const float* x, int rows, int cols) {
        int warps_per_block = T / 32;
        matvec_bf16_kernel<<<blocks_for(rows, warps_per_block), T>>>(out, w, x, rows, cols);
    };

    embed_kernel<<<blocks_for(dim, T), T>>>(m.x, m.embed, token, dim);
    emit("embed", m.x, dim);

    for (int li = 0; li < c.n_layers; ++li) {
        const Impl::Layer& L = m.layers[li];
        std::string p = "layer" + std::to_string(li) + ".";

        float* k_layer = m.k_cache + size_t(li) * max_seq_ * kv_dim;
        float* v_layer = m.v_cache + size_t(li) * max_seq_ * kv_dim;
        float* k = k_layer + size_t(pos) * kv_dim;
        float* v = v_layer + size_t(pos) * kv_dim;

        rmsnorm_kernel<<<1, 1024>>>(m.xb, m.x, L.attn_norm, dim, c.norm_eps);
        emit(p + "attn_norm", m.xb, dim);

        matvec(m.q, L.wq, m.xb, q_dim, dim);
        matvec(k, L.wk, m.xb, kv_dim, dim);
        matvec(v, L.wv, m.xb, kv_dim, dim);

        rope_kernel<<<blocks_for(c.n_heads * hd / 2, T), T>>>(m.q, c.n_heads, hd, pos, c.rope_theta);
        rope_kernel<<<blocks_for(c.n_kv_heads * hd / 2, T), T>>>(k, c.n_kv_heads, hd, pos, c.rope_theta);

        size_t smem = size_t(pos + 1) * sizeof(float);
        attention_kernel<<<c.n_heads, 128, smem>>>(m.attn, m.q, k_layer, v_layer, pos, hd, kv_dim, group, scale);

        matvec(m.xb, L.wo, m.attn, dim, q_dim);
        emit(p + "attn_out", m.xb, dim);
        add_kernel<<<blocks_for(dim, T), T>>>(m.x, m.xb, dim);

        rmsnorm_kernel<<<1, 1024>>>(m.xb, m.x, L.mlp_norm, dim, c.norm_eps);
        emit(p + "mlp_norm", m.xb, dim);

        matvec(m.h1, L.gate, m.xb, c.hidden_dim, dim);
        matvec(m.h2, L.up, m.xb, c.hidden_dim, dim);
        silu_mul_kernel<<<blocks_for(c.hidden_dim, T), T>>>(m.h1, m.h2, c.hidden_dim);
        matvec(m.xb, L.down, m.h1, dim, c.hidden_dim);
        emit(p + "mlp_out", m.xb, dim);
        add_kernel<<<blocks_for(dim, T), T>>>(m.x, m.xb, dim);

        emit(p + "out", m.x, dim);
    }

    rmsnorm_kernel<<<1, 1024>>>(m.xb, m.x, m.final_norm, dim, c.norm_eps);
    emit("final_norm", m.xb, dim);

    matvec(m.logits, m.lm_head, m.xb, c.vocab, dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(logits_.data(), m.logits, c.vocab * sizeof(float), cudaMemcpyDeviceToHost));
    emit("logits", m.logits, c.vocab);
    return logits_;
}
