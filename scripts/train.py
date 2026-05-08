"""
Phase 2 — full online activation-localized distillation.

Loads teacher (frozen 4-bit), student (4-bit + LoRA), trained projection from
phase 1, and runs the divergence-localized loop.

Usage:
    python scripts/train.py \\
        --config configs/default.json \\
        --prompts data/calibration.txt \\
        --projection logs/projection.pt
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loader import load_both, STUDENT_HIDDEN, TEACHER_HIDDEN, STUDENT_LAYERS, VOCAB_SIZE
from models.projection import HiddenProjection
from distill.alignment import get_or_make_map
from scoring.divergence import DivergenceSelector, DivergenceConfig
from scoring.localization.loop import DistillStep


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_dataset_iter(cfg, tokenizer, device):
    """
    Streams a tokenized text dataset. Default: filtered openhermes from sister project.
    """
    name = cfg.get("dataset_name", "Malum0x/openhermes2.5-Perplexity_filtered_top30")
    split = cfg.get("dataset_split", "train")
    max_len = cfg.get("max_length", 384)

    print(f"streaming dataset: {name} [{split}]")
    ds = load_dataset(name, split=split, streaming=True)

    def fmt(sample):
        # OpenHermes-style chat -> flat text
        if "conversations" in sample:
            parts = []
            for turn in sample["conversations"]:
                role = turn.get("from", "")
                val  = turn.get("value", "")
                if role == "human":
                    parts.append(f"### Human: {val}")
                elif role == "gpt":
                    parts.append(f"### Assistant: {val}")
            return "\n".join(parts)
        return sample.get("text") or sample.get("prompt") or ""

    def gen():
        for sample in ds:
            text = fmt(sample)
            if not text or not text.strip():
                continue
            tok = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_len, padding=False)
            yield {k: v.to(device) for k, v in tok.items()}

    return gen()


def curriculum_strategy(step: int, total: int, cfg: dict) -> str:
    """
    Optional curriculum: 'all' for first portion, 'topk_soft' for the middle,
    'argmin' for the final stretch. Disabled by default.
    """
    cur = cfg.get("curriculum", None)
    if not cur:
        return cfg.get("strategy", "topk_soft")
    a = cur.get("all_until_frac", 0.2)
    s = cur.get("argmin_after_frac", 0.8)
    f = step / max(total, 1)
    if f < a: return "all"
    if f >= s: return "argmin"
    return "topk_soft"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--projection", required=True, help="path to phase-1 projection.pt")
    p.add_argument("--alignment_path", default=None)
    p.add_argument("--output_dir", default="logs/phase2")
    args = p.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. models
    teacher, student, tokenizer = load_both(device=device, student_train=True,
                                            use_qlora=cfg.get("use_qlora", True))

    # 2. projection (load phase-1 weights)
    projection = HiddenProjection(student_dim=STUDENT_HIDDEN, teacher_dim=TEACHER_HIDDEN).to(device)
    projection.load(args.projection)
    if cfg.get("freeze_projection", False):
        for p_ in projection.parameters():
            p_.requires_grad = False
    projection.train()

    # 3. layer map (last_n_skip protects the layers that feed the student's lm_head)
    layer_map = get_or_make_map(args.alignment_path, last_n_skip=cfg.get("last_n_skip", 0))

    # 4. selector
    sel_cfg = DivergenceConfig(
        strategy=cfg.get("strategy", "topk_soft"),
        top_k=cfg.get("top_k", 4),
        softmax_temp=cfg.get("softmax_temp", 0.5),
        cosine_weight=cfg.get("cosine_weight", 1.0),
        mse_weight=cfg.get("mse_weight", 0.1),
    )
    selector = DivergenceSelector(sel_cfg)

    step_runner = DistillStep(teacher, student, projection, selector, layer_map, device=device)

    # 5. data
    data_iter = get_dataset_iter(cfg, tokenizer, device)

    # 6. optimizer
    trainable = [p for p in student.parameters() if p.requires_grad]
    if not cfg.get("freeze_projection", False):
        trainable += list(projection.parameters())
    optim = AdamW(trainable, lr=cfg.get("lr", 5e-5))

    n_steps = cfg.get("steps", 2000)
    log_every = cfg.get("log_every", 20)
    save_every = cfg.get("save_every", 500)

    log_path = os.path.join(args.output_dir, "phase2_train.log")
    log_f = open(log_path, "w")
    selected_counter = Counter()

    pbar = tqdm(range(n_steps), desc="phase 2: distill")
    t0 = time.time()
    for step in pbar:
        # curriculum (optional)
        new_strategy = curriculum_strategy(step, n_steps, cfg)
        if new_strategy != selector.cfg.strategy:
            selector.cfg.strategy = new_strategy

        try:
            batch = next(data_iter)
        except StopIteration:
            print("dataset exhausted")
            break

        out = step_runner.step(batch)
        act_loss = out["loss"]

        # Output-level KL co-loss (student || teacher) on shared vocab portion only.
        # Anchors generation coherence so activation pulling can't break the lm_head.
        kl_w = cfg.get("kl_weight", 0.0)
        if kl_w > 0:
            T = cfg.get("kl_temperature", 2.0)
            shared = min(VOCAB_SIZE, out["student_logits"].size(-1), out["teacher_logits"].size(-1))
            s_log_p = F.log_softmax(out["student_logits"][..., :shared].float() / T, dim=-1)
            t_p     = F.softmax(out["teacher_logits"][..., :shared].float() / T, dim=-1)
            kl = F.kl_div(s_log_p, t_p, reduction="batchmean") * (T * T)
            loss = cfg.get("act_weight", 1.0) * act_loss + kl_w * kl
            kl_val = float(kl.item())
        else:
            loss = act_loss
            kl_val = None

        optim.zero_grad()
        loss.backward()
        if cfg.get("grad_clip", 1.0):
            torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
        optim.step()

        for pair in out["selected"]:
            selected_counter[pair] += 1

        if step % log_every == 0:
            top_layer = selected_counter.most_common(1)[0] if selected_counter else None
            log_entry = {
                "step": step,
                "loss": float(loss.item()),
                "act_loss": float(act_loss.item()),
                "kl_loss": kl_val,
                "strategy": selector.cfg.strategy,
                "top_selected": [list(top_layer[0]), top_layer[1]] if top_layer else None,
                "elapsed_s": time.time() - t0,
            }
            log_f.write(json.dumps(log_entry) + "\n")
            log_f.flush()
            pbar.set_postfix(loss=f"{loss.item():.4f}", act=f"{act_loss.item():.3f}",
                             kl=f"{kl_val:.3f}" if kl_val is not None else "off",
                             strat=selector.cfg.strategy)

        if step > 0 and step % save_every == 0:
            student.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{step}"))
            projection.save(os.path.join(args.output_dir, f"checkpoint-{step}", "projection.pt"))

    log_f.close()
    student.save_pretrained(os.path.join(args.output_dir, "final"))
    projection.save(os.path.join(args.output_dir, "final", "projection.pt"))

    # selection histogram
    with open(os.path.join(args.output_dir, "selected_layers.json"), "w") as f:
        json.dump({f"{p[0]}->{p[1]}": c for p, c in selected_counter.items()}, f, indent=2)

    print("phase 2 done.")


if __name__ == "__main__":
    main()
