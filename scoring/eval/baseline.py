"""
Generic lm-eval wrapper. Used by the three eval arms:

  baseline_a:  restored_top30 alone (no further training)
  baseline_b:  restored_top30 + standard KL distillation (control)
  method:      restored_top30 + this method

Settings match mlp-surgery's eval_all.py for direct comparison:
  GSM8K flexible-extract 5-shot, ARC Challenge acc_norm 0-shot,
  no chat template, no dtype override. batch_size 8 (deterministic for these tasks).
"""

import argparse
import json
import os
import subprocess
import sys
import glob


TASKS = ["gsm8k", "arc_challenge"]
DEFAULT_BATCH = "8"


def run_one(model_arg: str, task: str, output_dir: str, batch_size: str = DEFAULT_BATCH):
    out = os.path.join(output_dir, task)
    if glob.glob(f"{out}/**/results_*.json", recursive=True):
        print(f"  skip {task} (cached)")
        return out
    os.makedirs(out, exist_ok=True)
    cmd = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_arg,
        "--tasks", task,
        "--device", "cuda",
        "--output_path", out,
        "--batch_size", batch_size,
    ]
    print("  $ " + " ".join(cmd))
    r = subprocess.run(cmd)
    return out if r.returncode == 0 else None


def parse_score(out_dir: str, task: str):
    if not out_dir:
        return None
    files = sorted(glob.glob(f"{out_dir}/**/results_*.json", recursive=True))
    if not files:
        return None
    with open(files[-1]) as f:
        data = json.load(f)
    res = data.get("results", {})
    tr = res.get(task) or next((v for k, v in res.items() if task in k), None)
    if not tr: return None
    if task == "gsm8k":
        for k in ("exact_match,flexible-extract", "exact_match,strict-match", "exact_match"):
            if k in tr: return round(tr[k] * 100, 2)
    if task == "arc_challenge":
        for k in ("acc_norm,none", "acc_norm", "acc,none", "acc"):
            if k in tr: return round(tr[k] * 100, 2)
    return None


def evaluate(model_arg: str, output_dir: str, batch_size: str = DEFAULT_BATCH):
    """Runs both tasks, returns dict[task] = score."""
    os.makedirs(output_dir, exist_ok=True)
    scores = {}
    for task in TASKS:
        out = run_one(model_arg, task, output_dir, batch_size)
        scores[task] = parse_score(out, task)
        print(f"  {task}: {scores[task]}")
    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump(scores, f, indent=2)
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_arg", required=True,
                   help="lm-eval --model_args string, e.g. pretrained=Malum0x/mlp-surgery-restored-top30")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", default=DEFAULT_BATCH)
    args = p.parse_args()
    evaluate(args.model_arg, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
