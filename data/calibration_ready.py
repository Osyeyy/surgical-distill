"""
Builds data/calibration.txt — flat one-prompt-per-line text used for the Phase 0
diagnostic and Phase 1 projection training.

Default source: Malum0x/openhermes2.5-Perplexity_filtered_top30 (sister project).
Falls back to wikitext if the HF dataset isn't accessible.
"""

import argparse
import os

from datasets import load_dataset


def extract_text(sample) -> str:
    if "conversations" in sample:
        parts = []
        for turn in sample["conversations"]:
            role = turn.get("from", "")
            val  = turn.get("value", "")
            parts.append(val)
        return " ".join(parts)
    return sample.get("text") or sample.get("prompt") or sample.get("instruction") or ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="Malum0x/openhermes2.5-Perplexity_filtered_top30")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--max_chars", type=int, default=512)
    p.add_argument("--output", default="data/calibration.txt")
    args = p.parse_args()

    print(f"loading {args.source} [{args.split}]")
    try:
        ds = load_dataset(args.source, split=args.split, streaming=True)
    except Exception as e:
        print(f"failed: {e}")
        print("falling back to wikitext-103")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=args.split, streaming=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n_written = 0
    with open(args.output, "w") as f:
        for sample in ds:
            if n_written >= args.n:
                break
            text = extract_text(sample).strip()
            if not text:
                continue
            text = text.replace("\n", " ").replace("\r", " ")
            text = text[:args.max_chars]
            f.write(text + "\n")
            n_written += 1
    print(f"wrote {n_written} prompts -> {args.output}")


if __name__ == "__main__":
    main()
