"""Write the perplexity test passage as token ids for any model's tokenizer.

golden.py does this for TinyLlama along with the full HF reference. Bigger models don't
fit in RAM for an fp32 HF reference, so this only writes the tokens:

    python tools/ppl_tokens.py models/mistral-7b golden-mistral
    ./build/engine ppl models/mistral-7b golden-mistral cuda-q8
"""
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from golden import PPL_TEXT  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print("usage: python tools/ppl_tokens.py <model_dir> <out_dir>")
        sys.exit(1)
    model_dir, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(PPL_TEXT).input_ids[:512]
    np.array(ids, dtype=np.int32).tofile(out / "ppl_tokens.bin")
    with open(out / "meta.json", "w") as f:
        json.dump({"model": model_dir, "ppl_tokens": len(ids)}, f, indent=2)
    print(f"wrote {len(ids)} tokens to {out / 'ppl_tokens.bin'}")


if __name__ == "__main__":
    main()
