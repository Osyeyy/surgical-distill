"""
Phase 1 — projection-only training.

Trains HiddenProjection on activation matching, with both models frozen.
Establishes a reliable bridge before student weights start moving in Phase 2.

Usage:
    python scripts/train_projection.py \\
        --prompts data/calibration.txt \\
        --steps 2000 \\
        --output logs/projection.pt
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loader import load_both, STUDENT_HIDDEN, TEACHER_HIDDEN
from models.projection import HiddenProjection
from distill.hooks import ActivationCapture
from distill.alignment import get_or_make_map


def collate(prompts, tokenizer, max_length, device):
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)


def projection_loss(student_acts, teacher_acts, projection, layer_map):
    """All-layer MSE+cosine loss, trains projection only."""
    losses = []
    for s_idx, t_idx in layer_map.items():
        if s_idx >= len(student_acts) or t_idx >= len(teacher_acts):
            continue
        s = student_acts[s_idx]                              # (B,T,D_s) — no grad (frozen)
        t = teacher_acts[t_idx]                              # (B,T,D_t)
        s_proj = projection(s)                               # (B,T,D_t) — has grad
        cos = F.cosine_similarity(s_proj, t, dim=-1).mean()
        mse = F.mse_loss(s_proj, t)
        losses.append((1.0 - cos) + 0.1 * mse)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--output", default="logs/projection.pt")
    p.add_argument("--alignment_path", default=None,
                   help="optional refined layer map path (json)")
    p.add_argument("--log_every", type=int, default=20)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. load both models (student in eval-only, no LoRA — projection is the only trainable)
    teacher, student, tokenizer = load_both(device=device, student_train=False, use_qlora=True)
    student.eval()

    # 2. projection
    projection = HiddenProjection(student_dim=STUDENT_HIDDEN, teacher_dim=TEACHER_HIDDEN).to(device)
    projection.train()

    # 3. layer map
    layer_map = get_or_make_map(args.alignment_path)

    # 4. data
    with open(args.prompts) as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise SystemExit(f"no prompts in {args.prompts}")
    print(f"loaded {len(prompts)} prompts")

    optim = AdamW(projection.parameters(), lr=args.lr)

    # 5. capture handles
    teacher_capture = ActivationCapture(teacher, detach=True)
    student_capture = ActivationCapture(student, detach=True)   # student frozen → detach OK

    log_path = os.path.join(os.path.dirname(args.output) or ".", "projection_train.log")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    log_f = open(log_path, "w")

    pbar = tqdm(range(args.steps), desc="phase 1: projection")
    rng = torch.Generator().manual_seed(0)
    losses = []
    t0 = time.time()
    for step in pbar:
        # sample a small batch from the prompt pool (with replacement for simplicity)
        idxs = torch.randint(0, len(prompts), (args.batch_size,), generator=rng).tolist()
        batch_prompts = [prompts[i] for i in idxs]
        batch = collate(batch_prompts, tokenizer, args.max_length, device)

        with torch.no_grad():
            teacher_capture.attach(); teacher(**batch);  t_acts = teacher_capture.get_full();  teacher_capture.detach()
            student_capture.attach(); student(**batch);  s_acts = student_capture.get_full();  student_capture.detach()

        loss = projection_loss(s_acts, t_acts, projection, layer_map)
        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{sum(losses[-50:])/min(50,len(losses)):.4f}")

        if step % args.log_every == 0:
            log_f.write(json.dumps({"step": step, "loss": loss.item(),
                                    "elapsed_s": time.time() - t0}) + "\n")
            log_f.flush()

    log_f.close()
    projection.save(args.output)
    print(f"phase 1 done. final avg loss = {sum(losses[-50:])/min(50,len(losses)):.4f}")


if __name__ == "__main__":
    main()
