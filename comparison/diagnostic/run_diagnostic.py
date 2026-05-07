"""
Phase 0 — Diagnostic.

Runs both models on the same prompts, captures hidden states at every layer,
projects student → teacher space, computes per-layer cosine similarity.
Saves a baseline divergence map for the Phase 1 / Phase 2 comparison.

Usage:
    python comparison/diagnostic/run_diagnostic.py --prompts data/calibration.txt --n 100
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.loader import load_both, STUDENT_LAYERS, TEACHER_LAYERS, STUDENT_HIDDEN, TEACHER_HIDDEN
from models.projection import HiddenProjection
from distill.hooks import ActivationCapture


def default_layer_map():
    """student layer i -> teacher layer round(i * 28/36).  Endpoints align (0→0, 35→27)."""
    return {
        i: int(round(i * TEACHER_LAYERS / STUDENT_LAYERS))
        for i in range(STUDENT_LAYERS)
    }


@torch.no_grad()
def run_single(model, tokenizer, prompt: str, max_length: int = 256, device: str = "cuda"):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    model(**inputs)
    return inputs


def compute_similarity(student_acts, teacher_acts, projection, layer_map, device):
    """
    Returns dict student_layer_idx -> mean cosine similarity (scalar).
    Both activation lists are mean-pooled per layer: shape (B, D).
    """
    similarities = {}
    for s_idx, t_idx in layer_map.items():
        if s_idx >= len(student_acts) or t_idx >= len(teacher_acts):
            continue
        s_act = student_acts[s_idx].to(device)             # (B, D_s)
        t_act = teacher_acts[t_idx].to(device)             # (B, D_t)
        s_proj = projection(s_act)                          # (B, D_t)
        sim = F.cosine_similarity(s_proj, t_act, dim=-1).mean().item()
        similarities[s_idx] = sim
    return similarities


def run_diagnostic(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher, student, tokenizer = load_both(device=device, student_train=False, use_qlora=True)

    projection = HiddenProjection(student_dim=STUDENT_HIDDEN, teacher_dim=TEACHER_HIDDEN).to(device)
    if args.projection_path and os.path.exists(args.projection_path):
        projection.load(args.projection_path)
        print("using saved projection")
    else:
        print("using untrained projection (random init) — expected for phase 0")

    with open(args.prompts) as f:
        prompts = [line.strip() for line in f if line.strip()]
    prompts = prompts[:args.n]
    print(f"running diagnostic on {len(prompts)} prompts")

    layer_map = default_layer_map()
    print(f"layer map (student → teacher): {layer_map}")

    accumulated = {i: [] for i in range(STUDENT_LAYERS)}
    student_capture = ActivationCapture(student)
    teacher_capture = ActivationCapture(teacher)

    for prompt in tqdm(prompts, desc="diagnostic"):
        student_capture.attach()
        run_single(student, tokenizer, prompt, device=device)
        student_acts = student_capture.get_layer_means()
        student_capture.detach()

        teacher_capture.attach()
        run_single(teacher, tokenizer, prompt, device=device)
        teacher_acts = teacher_capture.get_layer_means()
        teacher_capture.detach()

        sims = compute_similarity(student_acts, teacher_acts, projection, layer_map, device)
        for s_idx, sim in sims.items():
            accumulated[s_idx].append(sim)

    mean_similarities = {
        s_idx: sum(vals) / len(vals)
        for s_idx, vals in accumulated.items()
        if vals
    }

    print("\n--- baseline divergence map ---")
    print(f"{'student layer':<15} {'teacher layer':<15} {'cosine sim':<12} divergence")
    print("-" * 60)
    for s_idx in sorted(mean_similarities.keys()):
        t_idx = layer_map[s_idx]
        sim = mean_similarities[s_idx]
        bar = "█" * max(0, int((1 - sim) * 20))
        print(f"{s_idx:<15} {t_idx:<15} {sim:<12.4f} {bar}")

    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "layer_map": {str(k): v for k, v in layer_map.items()},
        "mean_similarities": {str(k): v for k, v in mean_similarities.items()},
        "n_prompts": len(prompts),
        "projection_trained": args.projection_path is not None,
        "student_layers": STUDENT_LAYERS,
        "teacher_layers": TEACHER_LAYERS,
    }
    out_path = os.path.join(args.output_dir, "diagnostic_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nresults saved → {out_path}")
    print("run comparison/diagnostic/plot.py to visualize")

    return mean_similarities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="logs/diagnostic")
    parser.add_argument("--projection_path", type=str, default=None,
                        help="optional path to saved projection weights")
    args = parser.parse_args()
    run_diagnostic(args)
