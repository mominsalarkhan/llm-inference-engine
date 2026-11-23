# LLM Inference Engine

LLM inference engine in C++ and CUDA, built from scratch. Target: TinyLlama 1.1B, then bigger models quantized, on an RTX 3060 (12GB).

## Status

- [x] safetensors loader (mmap, BF16/F16/F32)
- [x] CPU forward pass with KV cache, grouped-query attention, RoPE, SwiGLU
- [x] Matches Hugging Face on all 22 layers of TinyLlama 1.1B (max relative error ~1e-5), identical greedy output
- [x] CUDA backend: 126 tok/s on RTX 3060 (28x over CPU), matches HF on all layers
- [ ] Custom decode kernels (GEMV, fused attention)
- [ ] INT8 / INT4 weight quantization
- [ ] CUDA graphs
- [ ] Paged KV cache and continuous batching

CPU baseline (Apple Silicon, naive scalar matvec): 4.3 tok/s.

## Setup

### Mac (write code, CPU build, reference data)

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tools/golden.py
```

This downloads TinyLlama (~4.4GB in fp32) and writes reference activations to `golden/`.

Build the CPU version:

```
cmake -B build && cmake --build build
./build/engine
```

### PC (CUDA build and testing)

Use WSL2 with Ubuntu, not native Windows.

1. `wsl --install -d Ubuntu` in PowerShell, then reboot.
2. Install the latest NVIDIA Windows driver. Don't install a Linux driver inside WSL.
3. In Ubuntu, install the CUDA toolkit for WSL from NVIDIA's site (pick Linux > x86_64 > WSL-Ubuntu).
4. `nvidia-smi` and `nvcc --version` should both work.

Build with CUDA:

```
cmake -B build -DENGINE_CUDA=ON && cmake --build build
./build/engine
```

### Editing from the Mac

Run `sudo apt install openssh-server` in WSL and start the ssh service. Then use VS Code Remote-SSH from the Mac to edit and build on the PC directly. Git keeps both machines in sync.

## Checking the engine

Point `models/tinyllama` at the HF snapshot folder (it holds `config.json` and `model.safetensors`):

```
ln -s "$(dirname "$(readlink models/tinyllama.safetensors)")" models/tinyllama
```

Then run the full forward pass against golden, layer by layer, plus 20 tokens of greedy generation:

```
./build/engine check models/tinyllama golden capital
```

## Golden data

`golden/<prompt>.safetensors` holds, per prompt:

- `input_ids`, `embed`, `logits`, `final_norm`
- per layer i: `layer{i}.attn_norm`, `layer{i}.attn_out`, `layer{i}.mlp_norm`, `layer{i}.mlp_out`, `layer{i}.out`

`golden/meta.json` has the model config, token ids and 20 greedy tokens per prompt.
