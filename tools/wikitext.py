"""WikiText-2 perplexity set for the engine: the standard benchmark quantization papers report.

Takes the WikiText-2 (raw) test split, joins it the usual way, tokenizes it with the model's
tokenizer, keeps the first --tokens tokens, and writes them for `engine ppl`, which scores them
in non-overlapping windows of --window tokens (the KV cache starts fresh in every window).

    python tools/wikitext.py models/tinyllama wikitext-tiny --hf-ref
    ./build/engine ppl models/tinyllama wikitext-tiny cuda-q4

--hf-ref also scores the same windows with Hugging Face in fp32 on the CPU, as the reference.
That needs about 4 bytes of RAM per parameter, so it is only practical for small models.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("out_dir")
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--hf-ref", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                           repo_type="dataset")
    text = "\n\n".join(pq.read_table(path).column("text").to_pylist())

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tok(text, return_tensors="np").input_ids[0]
    n = min(args.tokens, len(ids)) // args.window * args.window
    ids = ids[:n].astype(np.int32)
    ids.tofile(out / "ppl_tokens.bin")
    meta = {"model": args.model_dir, "dataset": "wikitext-2-raw-v1 test", "ppl_tokens": int(n),
            "ppl_window": args.window, "test_set_tokens": int(len(tok(text).input_ids))}
    print(f"wrote {n} tokens ({n // args.window} windows of {args.window}) of {meta['test_set_tokens']} in the test set")

    if args.hf_ref:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32)
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for w in range(0, n, args.window):
                x = torch.from_numpy(ids[w:w + args.window].astype(np.int64))[None]
                loss = model(x, labels=x).loss.item()  # mean over window-1 predictions
                total += loss * (x.shape[1] - 1)
                count += x.shape[1] - 1
                print(f"\r  hf fp32: window {w // args.window + 1}/{n // args.window}", end="", flush=True)
        meta["ppl_hf"] = math.exp(total / count)
        print(f"\nhf fp32 perplexity: {meta['ppl_hf']:.4f}")

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
