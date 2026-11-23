#include "model_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <thread>
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

constexpr int Q8_GROUP = 128; // weights per scale; one warp covers exactly one group per step

// INT8 weights with one fp32 scale per 128 weights. One warp per row: each lane loads
// 4 int8 weights as a single 32-bit word and the matching 4 inputs as a float4, so one
// step of the loop is one full group and the scale is shared by the whole warp.
__global__ void matvec_q8_kernel(float* out, const int8_t* Wq, const float* scales, const float* x, int rows,
                                 int cols) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;

    int groups = cols / Q8_GROUP;
    const int* w = reinterpret_cast<const int*>(Wq + size_t(row) * cols);
    const float* sc = scales + size_t(row) * groups;
    const float4* x4 = reinterpret_cast<const float4*>(x);

    float sum = 0.0f;
    for (int g = 0; g < groups; ++g) {
        int packed = w[g * 32 + lane];
        float4 xv = x4[g * 32 + lane];
        float d = float(int8_t(packed)) * xv.x + float(int8_t(packed >> 8)) * xv.y +
                  float(int8_t(packed >> 16)) * xv.z + float(int8_t(packed >> 24)) * xv.w;
        sum += d * sc[g];
    }
    sum = warp_sum(sum);
    if (lane == 0) out[row] = sum;
}

constexpr int Q4_GROUP = 64; // weights per (scale, min) pair

// INT4 weights, two per byte (low nibble first), asymmetric: w = q * scale + min, q in 0..15.
// Each group of 64 has its own (scale, min) stored as one half2.
// One warp per row; each lane loads 8 bytes = 16 weights, so a warp covers 512 weights per step.
// Since w = q*scale + min, the dot product splits into scale * sum(q*x) + min * sum(x).
__global__ void matvec_q4_kernel(float* out, const uint8_t* Wq, const __half2* sm, const float* x, int rows,
                                 int cols) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;

    const uint2* w = reinterpret_cast<const uint2*>(Wq + size_t(row) * (cols / 2));
    const __half2* g = sm + size_t(row) * (cols / Q4_GROUP);
    const float4* x4 = reinterpret_cast<const float4*>(x);

    float sum = 0.0f;
    int steps = cols / 512;
    for (int it = 0; it < steps; ++it) {
        uint2 packed = w[it * 32 + lane];
        int base = it * 512 + lane * 16; // index of this lane's first weight
        float2 s = __half22float2(g[base / Q4_GROUP]);

        float4 xa = x4[base / 4];
        float4 xb = x4[base / 4 + 1];
        float4 xc = x4[base / 4 + 2];
        float4 xd = x4[base / 4 + 3];
        float xs[16] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w,
                        xc.x, xc.y, xc.z, xc.w, xd.x, xd.y, xd.z, xd.w};

        float sq = 0.0f;
        float sx = 0.0f;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            uint32_t word = k < 8 ? packed.x : packed.y;
            float q = float((word >> (4 * (k & 7))) & 0xF);
            sq += q * xs[k];
            sx += xs[k];
        }
        sum += s.x * sq + s.y * sx;
    }
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
    // A weight matrix on the GPU, in whichever format we loaded it as.
    struct Mat {
        uint16_t* bf16 = nullptr;
        int8_t* q8 = nullptr;
        float* scales = nullptr;
        uint8_t* q4 = nullptr;
        __half2* q4_sm = nullptr; // (scale, min) per group of 64
        int rows = 0;
        int cols = 0;
    };

    struct Layer {
        float* attn_norm;
        float* mlp_norm;
        Mat wqkv; // [q_dim + 2*kv_dim, dim]
        Mat wo;
        Mat wgu; // [2*hidden, dim], gate rows then up rows
        Mat down;
    };

    Quant quant = Quant::BF16;
    size_t weight_bytes = 0;
    std::vector<void*> allocations;
    uint16_t* embed = nullptr; // embedding stays BF16: we only ever read one row of it
    Mat lm_head;
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

    // Stack one or more BF16 matrices by rows into a single GPU matrix, quantizing on the way if asked.
    Mat upload(const WeightFiles& st, const std::vector<std::string>& names, const std::vector<int64_t>& rows,
               int64_t cols, Quant fmt) {
        Mat m;
        for (int64_t r : rows) m.rows += int(r);
        m.cols = int(cols);

        for (size_t i = 0; i < names.size(); ++i) {
            const Tensor& t = st.get(names[i]);
            if (t.dtype != DType::BF16)
                throw std::runtime_error(names[i] + ": CUDA path only handles BF16 checkpoints for now");
            if (t.shape.size() != 2 || t.shape[0] != rows[i] || t.shape[1] != cols)
                throw std::runtime_error(names[i] + ": unexpected shape");
        }

        if (fmt == Quant::BF16) {
            m.bf16 = alloc<uint16_t>(size_t(m.rows) * cols);
            int64_t offset = 0;
            for (size_t i = 0; i < names.size(); ++i) {
                const Tensor& t = st.get(names[i]);
                CUDA_CHECK(cudaMemcpy(m.bf16 + offset * cols, t.data, t.nbytes, cudaMemcpyHostToDevice));
                offset += rows[i];
            }
            weight_bytes += size_t(m.rows) * cols * 2;
            return m;
        }

        // Everything below quantizes on the CPU once at load, spread over all cores.
        // fn(global_row, weights_as_float) is called for every row of every stacked matrix.
        auto for_each_row = [&](auto&& fn) {
            int64_t offset = 0;
            for (size_t i = 0; i < names.size(); ++i) {
                const Tensor& t = st.get(names[i]);
                int64_t base = offset;
                auto work = [&, base](int64_t r0, int64_t r1) {
                    std::vector<float> row(static_cast<size_t>(cols));
                    for (int64_t r = r0; r < r1; ++r) {
                        to_f32(t, size_t(r * cols), size_t(cols), row.data());
                        fn(base + r, row.data());
                    }
                };
                int n_threads = int(std::max(1u, std::thread::hardware_concurrency()));
                int64_t chunk = (rows[i] + n_threads - 1) / n_threads;
                std::vector<std::thread> pool;
                for (int k = 0; k < n_threads; ++k) {
                    int64_t r0 = k * chunk;
                    int64_t r1 = std::min(rows[i], r0 + chunk);
                    if (r0 < r1) pool.emplace_back(work, r0, r1);
                }
                for (auto& th : pool) th.join();
                offset += rows[i];
            }
        };

        if (fmt == Quant::Q8) {
            // symmetric: scale = max|w| / 127 per group of 128, q = round(w / scale)
            if (cols % Q8_GROUP != 0) throw std::runtime_error(names[0] + ": cols not a multiple of 128");
            int groups = int(cols / Q8_GROUP);
            std::vector<int8_t> hq(size_t(m.rows) * cols);
            std::vector<float> hs(size_t(m.rows) * groups);

            for_each_row([&](int64_t r, const float* row) {
                for (int g = 0; g < groups; ++g) {
                    const float* w = row + g * Q8_GROUP;
                    float amax = 0.0f;
                    for (int k = 0; k < Q8_GROUP; ++k) amax = std::max(amax, std::fabs(w[k]));
                    float scale = amax / 127.0f;
                    float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
                    int8_t* q = hq.data() + size_t(r) * cols + g * Q8_GROUP;
                    for (int k = 0; k < Q8_GROUP; ++k) {
                        float v = std::nearbyint(w[k] * inv);
                        q[k] = int8_t(std::max(-127.0f, std::min(127.0f, v)));
                    }
                    hs[size_t(r) * groups + g] = scale;
                }
            });

            m.q8 = alloc<int8_t>(hq.size());
            m.scales = alloc<float>(hs.size());
            CUDA_CHECK(cudaMemcpy(m.q8, hq.data(), hq.size(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(m.scales, hs.data(), hs.size() * sizeof(float), cudaMemcpyHostToDevice));
            weight_bytes += hq.size() + hs.size() * sizeof(float);
            return m;
        }

        // Q4: asymmetric per group of 64. min and max of the group map to 0 and 15,
        // step = (max - min) / 15. scale and min are rounded to fp16 first and the
        // quantization uses the rounded values, so the GPU sees exactly what we computed with.
        if (cols % 512 != 0) throw std::runtime_error(names[0] + ": cols not a multiple of 512");
        int groups = int(cols / Q4_GROUP);
        std::vector<uint8_t> hq(size_t(m.rows) * cols / 2);
        std::vector<__half2> hsm(size_t(m.rows) * groups);

        bool search = fmt == Quant::Q4_MSE;

        for_each_row([&](int64_t r, const float* row) {
            int q[Q4_GROUP];
            int trial[Q4_GROUP];
            for (int g = 0; g < groups; ++g) {
                const float* w = row + g * Q4_GROUP;
                float lo = w[0];
                float hi = w[0];
                for (int k = 1; k < Q4_GROUP; ++k) {
                    lo = std::min(lo, w[k]);
                    hi = std::max(hi, w[k]);
                }

                auto assign = [&](float sc, float mn, int* out) {
                    float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
                    for (int k = 0; k < Q4_GROUP; ++k)
                        out[k] = std::max(0, std::min(15, int(std::nearbyint((w[k] - mn) * inv))));
                };
                auto sq_error = [&](float sc, float mn, const int* qq) {
                    float e = 0.0f;
                    for (int k = 0; k < Q4_GROUP; ++k) {
                        float d = qq[k] * sc + mn - w[k];
                        e += d * d;
                    }
                    return e;
                };

                // plain round-to-nearest: min..max mapped onto 0..15
                float scale = (hi - lo) / 15.0f;
                float mn = lo;

                if (search) {
                    // Try ranges shrunk by 0..16%, then refit scale and min by least squares on the
                    // chosen levels (fit w ~ q*scale + min), reassign, refit once more. Keep the best.
                    float best = INFINITY;
                    for (int step = 0; step <= 8; ++step) {
                        float shrink = 0.02f * step;
                        float s_try = (hi - lo) * (1.0f - shrink) / 15.0f;
                        float m_try = lo + (hi - lo) * shrink * 0.5f;
                        assign(s_try, m_try, trial);

                        for (int iter = 0; iter < 2; ++iter) {
                            float n = float(Q4_GROUP), sq = 0, sqq = 0, sw = 0, sqw = 0;
                            for (int k = 0; k < Q4_GROUP; ++k) {
                                sq += trial[k];
                                sqq += float(trial[k]) * trial[k];
                                sw += w[k];
                                sqw += trial[k] * w[k];
                            }
                            float det = n * sqq - sq * sq;
                            if (det <= 0.0f) break;
                            float s_fit = (n * sqw - sq * sw) / det;
                            if (s_fit <= 0.0f) break;
                            s_try = s_fit;
                            m_try = (sw - s_fit * sq) / n;
                            assign(s_try, m_try, trial);
                        }

                        float e = sq_error(s_try, m_try, trial);
                        if (e < best) {
                            best = e;
                            scale = s_try;
                            mn = m_try;
                        }
                    }
                }

                // store as fp16 and quantize with the rounded values, so the GPU sees exactly this
                __half h_scale = __float2half(scale);
                __half h_min = __float2half(mn);
                assign(__half2float(h_scale), __half2float(h_min), q);

                uint8_t* out = hq.data() + (size_t(r) * cols + g * Q4_GROUP) / 2;
                for (int k = 0; k < Q4_GROUP; k += 2) out[k / 2] = uint8_t(q[k] | (q[k + 1] << 4));

                __half2 packed_sm;
                packed_sm.x = h_scale;
                packed_sm.y = h_min;
                hsm[size_t(r) * groups + g] = packed_sm;
            }
        });

        m.q4 = alloc<uint8_t>(hq.size());
        m.q4_sm = alloc<__half2>(hsm.size());
        CUDA_CHECK(cudaMemcpy(m.q4, hq.data(), hq.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m.q4_sm, hsm.data(), hsm.size() * sizeof(__half2), cudaMemcpyHostToDevice));
        weight_bytes += hq.size() + hsm.size() * sizeof(__half2);
        return m;
    }

    // Norm weights are tiny, convert to fp32 on the host first.
    float* upload_f32(const WeightFiles& st, const std::string& name) {
        std::vector<float> v = to_f32(st.get(name));
        float* d = alloc<float>(v.size());
        CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
        return d;
    }
};

CudaModel::CudaModel(const std::string& model_dir, int max_seq, Quant quant)
    : cfg_(load_config(model_dir + "/config.json")), max_seq_(max_seq), impl_(new Impl) {
    const Config& c = cfg_;
    Impl& m = *impl_;
    m.quant = quant;

    int q_dim = c.n_heads * c.head_dim;
    int kv_dim = c.n_kv_heads * c.head_dim;

    // the mmap only needs to live while we copy to the GPU
    WeightFiles st(model_dir);

    m.embed = m.upload(st, {"model.embed_tokens.weight"}, {c.vocab}, c.dim, Quant::BF16).bf16;
    std::string head = st.has("lm_head.weight") ? "lm_head.weight" : "model.embed_tokens.weight";
    // the output layer picks the next word, so in 4-bit mode it stays 8-bit
    m.lm_head = m.upload(st, {head}, {c.vocab}, c.dim, (quant == Quant::Q4 || quant == Quant::Q4_MSE) ? Quant::Q8 : quant);
    m.final_norm = m.upload_f32(st, "model.norm.weight");

    for (int i = 0; i < c.n_layers; ++i) {
        std::string p = "model.layers." + std::to_string(i) + ".";
        Impl::Layer l;
        l.attn_norm = m.upload_f32(st, p + "input_layernorm.weight");
        l.mlp_norm = m.upload_f32(st, p + "post_attention_layernorm.weight");
        l.wqkv = m.upload(st,
                          {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                           p + "self_attn.v_proj.weight"},
                          {q_dim, kv_dim, kv_dim}, c.dim, quant);
        l.wo = m.upload(st, {p + "self_attn.o_proj.weight"}, {c.dim}, q_dim, quant);
        l.wgu = m.upload(st, {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"},
                         {c.hidden_dim, c.hidden_dim}, c.dim, quant);
        l.down = m.upload(st, {p + "mlp.down_proj.weight"}, {c.dim}, c.hidden_dim, quant);
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

size_t CudaModel::weight_bytes() const {
    return impl_->weight_bytes;
}

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
    int kv_dim = c.n_kv_heads * hd;
    int group = c.n_heads / c.n_kv_heads;
    float scale = 1.0f / std::sqrt(float(hd));
    int n_post = (c.n_heads + c.n_kv_heads) * hd / 2 + kv_dim;

    auto matvec = [&](float* out, const Impl::Mat& w, const float* x) {
        int grid = blocks_for(w.rows, T / 32);
        if (w.q4)
            matvec_q4_kernel<<<grid, T, 0, s>>>(out, w.q4, w.q4_sm, x, w.rows, w.cols);
        else if (w.q8)
            matvec_q8_kernel<<<grid, T, 0, s>>>(out, w.q8, w.scales, x, w.rows, w.cols);
        else
            matvec_bf16_kernel<<<grid, T, 0, s>>>(out, w.bf16, x, w.rows, w.cols);
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

        matvec(m.qkv, L.wqkv, m.xb);
        qkv_post_kernel<<<blocks_for(n_post, T), T, 0, s>>>(m.qkv, k_layer, v_layer, m.state, c.n_heads,
                                                            c.n_kv_heads, hd, c.rope_theta);
        attention_kernel<<<c.n_heads, 128, m.attn_smem, s>>>(m.attn, m.qkv, k_layer, v_layer, m.state, hd,
                                                             kv_dim, group, scale);

        matvec(m.xb, L.wo, m.attn);
        emit(p + "attn_out", m.xb, dim);
        add_kernel<<<blocks_for(dim, T), T, 0, s>>>(m.x, m.xb, dim);

        rmsnorm_kernel<<<1, 1024, 0, s>>>(m.xb, m.x, L.mlp_norm, dim, c.norm_eps);
        emit(p + "mlp_norm", m.xb, dim);

        matvec(m.h, L.wgu, m.xb);
        silu_mul_kernel<<<blocks_for(c.hidden_dim, T), T, 0, s>>>(m.h, c.hidden_dim);
        matvec(m.xb, L.down, m.h);
        emit(p + "mlp_out", m.xb, dim);
        add_kernel<<<blocks_for(dim, T), T, 0, s>>>(m.x, m.xb, dim);

        emit(p + "out", m.x, dim);
    }

    rmsnorm_kernel<<<1, 1024, 0, s>>>(m.xb, m.x, m.final_norm, dim, c.norm_eps);
    emit("final_norm", m.xb, dim);

    matvec(m.logits, m.lm_head, m.xb);
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
