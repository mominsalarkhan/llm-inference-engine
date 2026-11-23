"""Type a prompt, get text back.

Only the tokenizer runs in Python (text <-> token ids). Every bit of model math runs
in the C++/CUDA engine through `engine generate`.

    python tools/run.py models/mistral-7b "The three most important ideas in physics are" --backend cuda-q4
"""
import argparse
import subprocess
import sys

from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("prompt")
    p.add_argument("--backend", default="cuda-q4")
    p.add_argument("--tokens", type=int, default=100)
    p.add_argument("--engine", default="build/engine")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tok(args.prompt).input_ids  # starts with the BOS token

    cmd = [args.engine, "generate", args.model_dir, args.backend, str(args.tokens)] + [str(i) for i in ids]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout, res.stderr, file=sys.stderr)
        sys.exit(res.returncode)

    out_ids, info = [], []
    for line in res.stdout.splitlines():
        if line.startswith("ids:"):
            out_ids = [int(x) for x in line[4:].split()]
        else:
            info.append(line)

    # decode prompt + answer together so spacing at the join comes out right
    print(tok.decode(ids + out_ids, skip_special_tokens=True))
    print()
    for line in info:
        print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
