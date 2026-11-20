"""Dump reference activations from Hugging Face so the engine has something to check against.

Run once per model. Writes one .safetensors per prompt plus meta.json into --out.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = {
    "capital": "The capital of France is",
    "count": "1, 2, 3, 4, 5,",
    "story": "Once upon a time, there was a small robot who",
}


def to_np(t):
    if isinstance(t, tuple):
        t = t[0]
    return np.ascontiguousarray(t.detach()[0].float().numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--out", default="golden")
    p.add_argument("--gen", type=int, default=20)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    acts = {}

    def grab(name):
        def hook(module, inputs, output):
            acts[name] = to_np(output)
        return hook

    m = model.model
    m.embed_tokens.register_forward_hook(grab("embed"))
    for i, layer in enumerate(m.layers):
        layer.input_layernorm.register_forward_hook(grab(f"layer{i}.attn_norm"))
        layer.self_attn.register_forward_hook(grab(f"layer{i}.attn_out"))
        layer.post_attention_layernorm.register_forward_hook(grab(f"layer{i}.mlp_norm"))
        layer.mlp.register_forward_hook(grab(f"layer{i}.mlp_out"))
        layer.register_forward_hook(grab(f"layer{i}.out"))
    m.norm.register_forward_hook(grab("final_norm"))

    meta = {"model": args.model, "config": model.config.to_dict(), "prompts": {}}

    for name, text in PROMPTS.items():
        acts.clear()
        ids = tok(text, return_tensors="pt").input_ids

        with torch.no_grad():
            res = model(ids)
        # copy before generate(), which fires the hooks again and overwrites acts
        tensors = dict(acts)

        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=args.gen, do_sample=False)

        tensors["input_ids"] = ids[0].numpy().astype(np.int32)
        tensors["logits"] = np.ascontiguousarray(res.logits[0].float().numpy())
        save_file(tensors, str(out / f"{name}.safetensors"))

        new_ids = gen[0, ids.shape[1]:].tolist()
        meta["prompts"][name] = {
            "text": text,
            "input_ids": ids[0].tolist(),
            "greedy_ids": new_ids,
            "greedy_text": tok.decode(new_ids),
        }
        top = int(res.logits[0, -1].argmax())
        print(f"{name}: {len(tensors)} tensors, next token -> {tok.decode([top])!r}")
        print(f"  greedy: {text}{meta['prompts'][name]['greedy_text']}")

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved to {out.resolve()}")


if __name__ == "__main__":
    main()
