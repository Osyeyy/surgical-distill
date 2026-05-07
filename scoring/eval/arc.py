"""ARC Challenge eval — thin wrapper around baseline.run_one."""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scoring.eval.baseline import run_one, parse_score


def evaluate_arc(model_arg: str, output_dir: str, batch_size: str = "8"):
    out = run_one(model_arg, "arc_challenge", output_dir, batch_size)
    return parse_score(out, "arc_challenge")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_arg", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", default="8")
    a = p.parse_args()
    print("arc_challenge:", evaluate_arc(a.model_arg, a.output_dir, a.batch_size))
