# LLM Inference Engine

An LLM inference engine written from scratch in C++ and CUDA. No PyTorch, no llama.cpp, no cuBLAS: a hand-written safetensors loader, a CPU reference, CUDA kernels for every operation, INT8 and INT4 weight quantization, and a perplexity harness to measure what compression costs.

Runs Llama-architecture models (TinyLlama 1.1B, Mistral 7B) on a consumer RTX 3060 (12 GB).

## Results

All numbers on an RTX 3060 (12 GB, 360 GB/s), single-stream decode at batch size 1.

**WikiText-2 perplexity** (first 16,384 test tokens, 512-token windows; percentages above each model's reference)

| Model | Reference | Q8 | Q4 round-to-nearest | Q4 activation-weighted | Gap closed |
| --- | --- | --- | --- | --- | --- |
| TinyLlama 1.1B | HF fp32, 10.1184 | 10.1268 (+0.08%) | 11.1180 (+9.88%) | 10.4486 (+3.26%) | 67% |
| Mistral 7B | own Q8, 6.5359 | 6.5359 | 6.8283 (+4.47%) | 6.6332 (+1.49%) | 67% |

The engine's BF16 path scores 10.1185 on the same TinyLlama windows, against 10.1184 for Hugging Face in fp32.

The tables below use a shorter held-out passage (about 460 tokens) from earlier in development.

**Mistral 7B**

| Format | Weights on GPU | Decode | Perplexity |
| --- | --- | --- | --- |
| Q8 | 7.59 GB | 41.7 tok/s | 5.7916 (baseline) |
| Q4, round-to-nearest | 4.32 GB | 72.3 tok/s | 5.9778 (+3.22%) |
| Q4, activation-weighted | 4.32 GB | 72.3 tok/s | 5.8551 (+1.10%) |

BF16 Mistral (14.5 GB) does not fit on the card, which is the point of quantizing it.

**TinyLlama 1.1B**

| Format | Weights on GPU | Decode | Perplexity vs HF fp32 (8.8647) |
| --- | --- | --- | --- |
| BF16 | 2.20 GB | 145 tok/s | 8.8648 (+0.00%) |
| Q8 | 1.20 GB | 250 tok/s | 8.8995 (+0.39%) |
| Q4, round-to-nearest | 0.74 GB | 355 tok/s | 9.2044 (+3.83%) |
| Q4, unweighted error search | 0.74 GB | 355 tok/s | 9.4972 (+7.14%) |
| Q4, activation-weighted | 0.74 GB | 355 tok/s | 8.9749 (+1.24%) |

Calibration for activation-weighted Q4 always uses a separate passage, never the text being scored.

**Reproduce WikiText-2**

```
pip install pyarrow
python tools/wikitext.py models/tinyllama wikitext-tiny --hf-ref
ENGINE_CALIB=golden/calib.safetensors ./build/engine ppl models/tinyllama wikitext-tiny cuda-q4awq
```

**Correctness.** The CPU and CUDA paths match Hugging Face Transformers (fp32) at five checkpoints in each of TinyLlama's 22 layers, within 2e-5 relative error, and reproduce its greedy output token for token.

## How it got fast

| Step | TinyLlama decode |
| --- | --- |
| CPU reference (Ryzen 5 3600) | 4.6 tok/s |
| First CUDA port, BF16 | 126 tok/s |
| Fused QKV and gate/up matrices, whole token recorded as one CUDA graph | 145 tok/s |
| INT8 weights, group 128 | 250 tok/s |
| INT4 weights, group 64, fp32 dequant kernel | 291 tok/s |
| INT4 with a dp4a integer-dot kernel | 355 tok/s |

At batch size 1 every token reads every weight once, so speed is set by bytes per weight. Nsight Compute showed the BF16 matrix kernels already at 91 to 92% of DRAM throughput, so the remaining wins came from launch overhead (CUDA graphs) and from fewer bytes (quantization). The first INT4 kernel reached only 58% of bandwidth on Mistral because unpacking cost more than the memory reads; the dp4a kernel quantizes the input vector to int8 once per matrix and handles four weights per instruction, reaching 82%.

## Features

- safetensors loader with mmap, BF16/F16/F32, multi-file checkpoints via the index file
- CPU forward pass (reference) and CUDA forward pass with a KV cache, grouped-query attention, RoPE, SwiGLU
- CUDA graph decode, fused projections
- Q8 (symmetric, group 128) and Q4 (asymmetric, group 64, fp16 scale and offset) weights, quantized at load time
- dp4a INT4 matrix-vector kernel
- Activation-weighted Q4 rounding from a calibration run (`engine calibrate`)
- Temperature and top-p sampling
- Golden tests against Hugging Face, perplexity evaluation, benchmarking

## Build

Linux or WSL2 with the CUDA toolkit for the GPU build; the CPU build also works on macOS.

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cmake -B build -DENGINE_CUDA=ON && cmake --build build     # GPU
cmake -B build && cmake --build build                      # CPU only
```

The CMake file targets `sm_86` (RTX 30 series); change `CMAKE_CUDA_ARCHITECTURES` for other cards.

## Use

```
# reference activations and perplexity text for TinyLlama (downloads the model)
python tools/golden.py
mkdir -p models
ln -s "$(dirname "$(python -c "from huggingface_hub import hf_hub_download as d; print(d('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'model.safetensors'))")")" models/tinyllama

# correctness, speed, quality
./build/engine check models/tinyllama golden capital cuda
./build/engine bench models/tinyllama cuda-q4 128
./build/engine ppl   models/tinyllama golden cuda-q4

# generate text
python tools/run.py models/tinyllama "The capital of France is" --backend cuda-q8
```

Backends: `cpu`, `cuda` (BF16), `cuda-q8`, `cuda-q4`, `cuda-q4mse`, `cuda-q4awq`.

**Mistral 7B, activation-weighted Q4**

```
hf download mistralai/Mistral-7B-v0.1 --include "*.safetensors" --include "*.json" --local-dir models/mistral-7b
python tools/ppl_tokens.py   models/mistral-7b golden-mistral
python tools/calib_tokens.py models/mistral-7b golden-mistral/calib_tokens.bin
./build/engine calibrate models/mistral-7b cuda-q8 golden-mistral/calib_tokens.bin golden-mistral/calib.safetensors

export ENGINE_CALIB=golden-mistral/calib.safetensors
./build/engine ppl models/mistral-7b golden-mistral cuda-q4awq
python tools/run.py models/mistral-7b "The three most important inventions in human history are" --backend cuda-q4awq
```

Environment switches for A/B tests: `ENGINE_NO_GRAPH=1` disables the CUDA graph, `ENGINE_Q4_FP32=1` uses the older INT4 kernel.

## Layout

```
src/safetensors.*   loader: header parser, mmap, dtype conversion, multi-file models
src/model.*         CPU forward pass (reference)
src/ops.*           CPU math: RMSNorm, matvec, RoPE, softmax
src/model_cuda.*    GPU weights, quantization, kernels, CUDA graph
src/main.cpp        commands: inspect, check, bench, ppl, generate, calibrate
tools/golden.py     Hugging Face reference activations, perplexity and calibration text
tools/wikitext.py   WikiText-2 test tokens and the Hugging Face reference score
tools/run.py        prompt in, text out (tokenizer in Python, all math in the engine)
```

## Status

- [x] safetensors loader
- [x] CPU forward pass with KV cache, matches Hugging Face
- [x] CUDA backend, CUDA graphs, fused projections
- [x] INT8 and INT4 weights, dp4a INT4 kernel
- [x] Mistral 7B in 4.3 GB at 72 tok/s
- [x] Activation-weighted INT4 rounding
- [x] Temperature and top-p sampling
- [x] WikiText-2 evaluation (16k-token subset)
- [ ] Batched prefill, so the full WikiText-2 test set scores in minutes
- [ ] Comparison with AWQ, GPTQ and llama.cpp at equal bits
- [ ] KV cache quantization
- [ ] Paged KV cache and continuous batching
