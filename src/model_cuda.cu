#include "model_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
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
// token and pos are read from device memory (state[0], state[1]) instead of being
// kernel arguments. That way one recorded CUDA graph works for every token.

__global__ void embed_kernel(float* out, const uint16_t* table, const int* state, int dim) {
    int token = state[0];
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

// One warp per output row.
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

// After the fused q|k|v matvec: apply RoPE to q (in place) and k, then write k and v
// into the KV cache at the current position. HF rotate_half layout.
// qkv layout: [q: n_heads*hd][k: n_kv*hd][v: n_kv*hd]
__global__ void qkv_post_kernel(float* qkv, float* k_layer, float* v_layer, const int* state,
                                int n_heads, int n_kv, int hd, float theta) {
    int pos = state[1];
    int half = hd / 2;
    int q_dim = n_heads * hd;
    int kv_dim = n_kv * hd;
    int n_rot = (n_heads + n_kv) * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_rot) {
        int h = idx / half; // q heads first, then k heads, they sit back to back
        int i = idx % half;
        float inv_freq = 1.0f / powf(theta, float(2 * i) / float(hd));
        float s, c;
        sincosf(float(pos) * inv_freq, &s, &c);

        float* v = qkv + h * hd;
        float a = v[i];
        float b = v[i + half];
        float ra = a * c - b * s;
        float rb = b * c + a * s;

        if (h < n_heads) {
            v[i] = ra;
            v[i + half] = rb;
        } else {
            float* k = k_layer + size_t(pos) * kv_dim + (h - n_heads) * hd;
            k[i] = ra;
            k[i + half] = rb;
        }
    } else if (idx < n_rot + kv_dim) {
        int j = idx - n_rot;
        v_layer[size_t(pos) * kv_dim + j] = qkv[q_dim + kv_dim + j];
    }
}

// One block per query head. Scores for every past position live in shared memory.
__global__ void attention_kernel(float* out, const float* q, const float* k_cache, const float* v_cache,
                                 const int* state, int head_dim, int kv_dim, int group, float scale) {
    extern __shared__ float sc[]; // sized for max_seq at launch
    int pos = state[1];

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

// After the fused gate|up matvec: h[i] = silu(h[i]) * h[n + i], written into the first half.
__global__ void silu_mul_kernel(float* h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = h[i];
        h[i] = g / (1.0f + expf(-g)) * h[n + i];
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
        uint16_t* wqkv; // [q_dim + 2*kv_dim, dim]
        uint16_t* wo;
        uint16_t* wgu; // [2*hidden, dim], gate rows then up rows
        uint16_t* down;
    };

    std::vector<void*> allocations;
    uint16_t* embed = nullptr;
    uint16_t* lm_head = nullptr;
    float* final_norm = nullptr;
    std::vector<Layer> layers;

    float* k_cache = nullptr;
    float* v_cache = nullptr;
    float *x, *xb, *qkv, *attn, *h, *logits;
    int* state = nullptr;      // device: [token, pos]
    int* host_state = nullptr; // pinned host copy
    size_t attn_smem = 0;

    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool use_graph = true;

    std::vector<float> host_tmp; // for taps

    ~Impl() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
        if (host_state) cudaFreeHost(host_state);
        for (void* p : allocations) cudaFree(p);
    }

    template <typename T>
    T* alloc(size_t count) {
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
        allocations.push_back(p);
        return static_cast<T*>(p);
    }

    // Upload one or more BF16 matrices stacked by rows into a single buffer.
    uint16_t* upload_rows(const SafeTensors& st, const std::vector<std::string>& names,
                          const std::vector<int64_t>& rows, int64_t cols) {
        int64_t total = 0;
        for (int64_t r : rows) total += r;
        uint16_t* d = alloc<uint16_t>(size_t(total * cols));

        int64_t offset = 0;
        for (size_t i = 0; i < names.size(); ++i) {
            const Tensor& t = st.get(names[i]);
            if (t.dtype != DType::BF16)
                throw std::runtime_error(names[i] + ": CUDA path only handles BF16 weights for now");
            if (t.shape.size() != 2 || t.shape[0] != rows[i] || t.shape[1] != cols)
                throw std::runtime_error(names[i] + ": unexpected shape");
            CUDA_CHECK(cudaMemcpy(d + offset * cols, t.data, t.nbytes, cudaMemcpyHostToDevice));
            offset += rows[i];
        }
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

    m.embed = m.upload_rows(st, {"model.embed_tokens.weight"}, {c.vocab}, c.dim);
    m.lm_head = st.has("lm_head.weight") ? m.upload_rows(st, {"lm_head.weight"}, {c.vocab}, c.dim) : m.embed;
    m.final_norm = m.upload_f32(st, "model.norm.weight");

    for (int i = 0; i < c.n_layers; ++i) {
        std::string p = "model.layers." + std::to_string(i) + ".";
        Impl::Layer l;
        l.attn_norm = m.upload_f32(st, p + "input_layernorm.weight");
        l.mlp_norm = m.upload_f32(st, p + "post_attention_layernorm.weight");
        l.wqkv = m.upload_rows(st,
                               {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                                p + "self_attn.v_proj.weight"},
                               {q_dim, kv_dim, kv_dim}, c.dim);
        l.wo = m.upload_rows(st, {p + "self_attn.o_proj.weight"}, {c.dim}, q_dim);
        l.wgu = m.upload_rows(st, {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"},
                              {c.hidden_dim, c.hidden_dim}, c.dim);
        l.down = m.upload_rows(st, {p + "mlp.down_proj.weight"}, {c.dim}, c.hidden_dim);
        m.layers.push_back(l);
    }

    size_t cache = size_t(c.n_layers) * max_seq_ * kv_dim;
    m.k_cache = m.alloc<float>(cache);
    m.v_cache = m.alloc<float>(cache);
    CUDA_CHECK(cudaMemset(m.k_cache, 0, cache * sizeof(float)));
    CUDA_CHECK(cudaMemset(m.v_cache, 0, cache * sizeof(float)));

    m.x = m.alloc<float>(c.dim);
    m.xb = m.alloc<float>(c.dim);
    m.qkv = m.alloc<float>(q_dim + 2 * kv_dim);
    m.attn = m.alloc<float>(q_dim);
    m.h = m.alloc<float>(2 * c.hidden_dim);
    m.logits = m.alloc<float>(c.vocab);
    m.state = m.alloc<int>(2);
    CUDA_CHECK(cudaMallocHost(&m.host_state, 2 * sizeof(int)));

    // attention scores need max_seq floats of shared memory; past 48KB we have to ask for it
    m.attn_smem = size_t(max_seq_) * sizeof(float);
    if (m.attn_smem > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        int(m.attn_smem)));

    CUDA_CHECK(cudaStreamCreate(&m.stream));
    m.use_graph = std::getenv("ENGINE_NO_GRAPH") == nullptr;

    logits_.resize(c.vocab);
}

CudaModel::~CudaModel() = default;

std::string CudaModel::device_name() const {
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return "unknown";
    return prop.name;
}

// Queue every kernel for one token on the stream. With a tap, it also copies
// checkpoints back for the golden check (that path can't be recorded into a graph).
void CudaModel::enqueue(const Tap* tap) {
    const Config& c = cfg_;
    Impl& m = *impl_;
    cudaStream_t s = m.stream;

    auto emit = [&](const std::string& name, const float* d, int n) {
        if (!tap || !*tap) return;
        m.host_tmp.resize(n);
        CUDA_CHECK(cudaMemcpyAsync(m.host_tmp.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        (*tap)(name, m.host_tmp.data(), n);
    };

    const int T = 256;
    int dim = c.dim;
    int hd = c.head_dim;
    int q_dim = c.n_heads * hd;
    int kv_dim = c.n_kv_heads * hd;
    int group = c.n_heads / c.n_kv_heads;
    float scale = 1.0f / std::sqrt(float(hd));
    int n_post = (c.n_heads + c.n_kv_heads) * hd / 2 + kv_dim;

    auto matvec = [&](float* out, const uint16_t* w, const float* x, int rows, int cols) {
        int warps_per_block = T / 32;
        matvec_bf16_kernel<<<blocks_for(rows, warps_per_block), T, 0, s>>>(out, w, x, rows, cols);
    };

    embed_kernel<<<blocks_for(dim, T), T, 0, s>>>(m.x, m.embed, m.state, dim);
    emit("embed", m.x, dim);

    for (int li = 0; li < c.n_layers; ++li) {
        const Impl::Layer& L = m.layers[li];
        std::string p = "layer" + std::to_string(li) + ".";
        float* k_layer = m.k_cache + size_t(li) * max_seq_ * kv_dim;
        float* v_layer = m.v_cache + size_t(li) * max_seq_ * kv_dim;

        rmsnorm_kernel<<<1, 1024, 0, s>>>(m.xb, m.x, L.attn_norm, dim, c.norm_eps);
        emit(p + "attn_norm", m.xb, dim);

        matvec(m.qkv, L.wqkv, m.xb, q_dim + 2 * kv_dim, dim);
        qkv_post_kernel<<<blocks_for(n_post, T), T, 0, s>>>(m.qkv, k_layer, v_layer, m.state, c.n_heads,
                                                            c.n_kv_heads, hd, c.rope_theta);
        attention_kernel<<<c.n_heads, 128, m.attn_smem, s>>>(m.attn, m.qkv, k_layer, v_layer, m.state, hd,
                                                             kv_dim, group, scale);

        matvec(m.xb, L.wo, m.attn, dim, q_dim);
        emit(p + "attn_out", m.xb, dim);
        add_kernel<<<blocks_for(dim, T), T, 0, s>>>(m.x, m.xb, dim);

        rmsnorm_kernel<<<1, 1024, 0, s>>>(m.xb, m.x, L.mlp_norm, dim, c.norm_eps);
        emit(p + "mlp_norm", m.xb, dim);

        matvec(m.h, L.wgu, m.xb, 2 * c.hidden_dim, dim);
        silu_mul_kernel<<<blocks_for(c.hidden_dim, T), T, 0, s>>>(m.h, c.hidden_dim);
        matvec(m.xb, L.down, m.h, dim, c.hidden_dim);
        emit(p + "mlp_out", m.xb, dim);
        add_kernel<<<blocks_for(dim, T), T, 0, s>>>(m.x, m.xb, dim);

        emit(p + "out", m.x, dim);
    }

    rmsnorm_kernel<<<1, 1024, 0, s>>>(m.xb, m.x, m.final_norm, dim, c.norm_eps);
    emit("final_norm", m.xb, dim);

    matvec(m.logits, m.lm_head, m.xb, c.vocab, dim);
    emit("logits", m.logits, c.vocab);
}

const std::vector<float>& CudaModel::forward(int token, int pos, const Tap& tap) {
    const Config& c = cfg_;
    Impl& m = *impl_;
    if (pos >= max_seq_) throw std::runtime_error("position past max_seq");
    if (token < 0 || token >= c.vocab) throw std::runtime_error("token id out of range");

    m.host_state[0] = token;
    m.host_state[1] = pos;
    CUDA_CHECK(cudaMemcpyAsync(m.state, m.host_state, 2 * sizeof(int), cudaMemcpyHostToDevice, m.stream));

    if (tap || !m.use_graph) {
        enqueue(tap ? &tap : nullptr);
    } else {
        if (!m.graph_exec) {
            // record the whole token once, then replay it every call
            CUDA_CHECK(cudaStreamBeginCapture(m.stream, cudaStreamCaptureModeThreadLocal));
            enqueue(nullptr);
            CUDA_CHECK(cudaStreamEndCapture(m.stream, &m.graph));
            CUDA_CHECK(cudaGraphInstantiate(&m.graph_exec, m.graph, 0));
        }
        CUDA_CHECK(cudaGraphLaunch(m.graph_exec, m.stream));
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(logits_.data(), m.logits, c.vocab * sizeof(float), cudaMemcpyDeviceToHost,
                               m.stream));
    CUDA_CHECK(cudaStreamSynchronize(m.stream));
    return logits_;
}
