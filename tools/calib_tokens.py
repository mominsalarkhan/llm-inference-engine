"""Write the calibration passage (golden.CALIB_TEXT) as token ids for a model's tokenizer.

    python tools/calib_tokens.py models/mistral-7b golden-mistral/calib_tokens.bin
    ./build/engine calibrate models/mistral-7b cuda-q8 golden-mistral/calib_tokens.bin golden-mistral/calib.safetensors
"""
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from golden import CALIB_TEXT  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print("usage: python tools/calib_tokens.py <model_dir> <out.bin>")
        sys.exit(1)
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    ids = tok(CALIB_TEXT).input_ids[:512]
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.array(ids, dtype=np.int32).tofile(out)
    print(f"wrote {len(ids)} tokens to {out}")


if __name__ == "__main__":
    main()
