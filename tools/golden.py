"""Dump reference activations from Hugging Face so the engine has something to check against.

Run once per model. Writes one .safetensors per prompt plus meta.json into --out.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# Original text for the perplexity test. Plain, varied English so the score means something.
PPL_TEXT = """The harbor town woke slowly on the first cold morning of the year. Fishing boats sat low in the water, their
nets still wet from the night before, and the gulls circled above them waiting for scraps. Along the main street the
bakery had already opened its doors, and the smell of fresh bread drifted down toward the docks where the workers were
unloading crates of ice.

Maria had lived in the town for most of her life. Her father had been a mechanic who repaired engines for the fleet,
and she had spent long afternoons in his workshop handing him tools and asking questions about how each part worked.
When he retired, she took over the shop, and over the years she added a small office where she kept careful records of
every repair, the cost of each part, and the time each job had taken.

Business was steady but never easy. Fuel prices rose and fell with the seasons, and when the catch was poor the
captains delayed their repairs for as long as they could. Maria learned to plan for those lean months. She saved what
she could in the summer, ordered parts in bulk when suppliers offered discounts, and trained two apprentices so that
she could take on larger jobs without working through the night.

One spring a storm damaged half the fleet. Masts snapped, hulls cracked against the pier, and several engines were
flooded with salt water. For three weeks the shop ran from dawn until well after dark. The apprentices learned more in
that month than in the previous year, and by the time the last boat returned to sea, the town had started to see the
workshop as something more than a place to fix machines. It had become part of how the community survived hard times.

Years later, when people asked Maria what she was most proud of, she did not talk about the engines. She talked about
the records she kept, because they showed exactly what had happened, what had worked, and what had failed. Good notes,
she said, are how you stop making the same mistake twice."""

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

    # perplexity reference: tokens for the engine, plus the fp32 HF score to compare against
    ppl_ids = tok(PPL_TEXT, return_tensors="pt").input_ids[:, :512]
    with torch.no_grad():
        loss = model(ppl_ids, labels=ppl_ids).loss.item()
    ppl_ids[0].numpy().astype(np.int32).tofile(out / "ppl_tokens.bin")
    meta["ppl_hf"] = math.exp(loss)
    meta["ppl_tokens"] = int(ppl_ids.shape[1])
    print(f"perplexity: {meta['ppl_hf']:.4f} over {meta['ppl_tokens']} tokens")

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved to {out.resolve()}")


if __name__ == "__main__":
    main()
