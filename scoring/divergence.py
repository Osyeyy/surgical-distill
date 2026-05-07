"""
DivergenceSelector — turns per-layer-pair similarities into a loss.

Strategies:
  'argmin'     : original idea. spotlight loss on the single most-divergent layer.
                 fragile (ping-pong, no co-adaptation), kept for ablation.
  'topk_soft'  : recommended. softmax-weighted loss across top-K most-divergent
                 layers (k=4 by default). Avoids ping-pong while preserving
                 the focus-on-weak-spots intuition.
  'all'        : TinyBERT-style baseline. uniform-weighted loss across every
                 aligned layer pair. needed to attribute whether localization
                 is the part that's helping.

Loss combines cosine + scale-aware MSE so magnitude information isn't thrown out.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DivergenceConfig:
    strategy: str = "topk_soft"        # 'argmin' | 'topk_soft' | 'all'
    top_k: int = 4
    softmax_temp: float = 0.5          # lower → sharper, closer to argmin
    cosine_weight: float = 1.0
    mse_weight: float = 0.1            # MSE on projected activations, kept small to avoid scale dominance


def _per_pair_loss(s_proj: torch.Tensor, t_act: torch.Tensor, cfg: DivergenceConfig) -> torch.Tensor:
    """
    s_proj, t_act: (B, T, D_t)  matched shape after projection.
    Returns scalar loss = (1 - cos) * w_cos + MSE * w_mse.
    """
    cos = F.cosine_similarity(s_proj, t_act, dim=-1)         # (B, T)
    cos_loss = (1.0 - cos).mean()
    mse_loss = F.mse_loss(s_proj, t_act) if cfg.mse_weight > 0 else torch.tensor(0.0, device=s_proj.device)
    return cfg.cosine_weight * cos_loss + cfg.mse_weight * mse_loss


def per_pair_similarity(s_proj: torch.Tensor, t_act: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity scalar for one layer pair."""
    return F.cosine_similarity(s_proj, t_act, dim=-1).mean()


class DivergenceSelector(nn.Module):
    """
    Inputs (per forward step):
      student_acts: list[Tensor (B,T,D_s)] of length L_s  (full per-token, not pooled)
      teacher_acts: list[Tensor (B,T,D_t)] of length L_t
      projection:   maps D_s → D_t
      layer_map:    dict[s_idx, t_idx]

    Output: scalar loss + diagnostics dict.
    """

    def __init__(self, cfg: DivergenceConfig | None = None):
        super().__init__()
        self.cfg = cfg or DivergenceConfig()

    def forward(self, student_acts: List[torch.Tensor], teacher_acts: List[torch.Tensor],
                projection: nn.Module, layer_map: Dict[int, int]) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        device = student_acts[0].device

        # 1. compute per-pair losses + similarities (no_grad on similarities for selection)
        per_pair_losses = []
        per_pair_sims = []
        valid_pairs = []
        for s_idx, t_idx in layer_map.items():
            if s_idx >= len(student_acts) or t_idx >= len(teacher_acts):
                continue
            s_full = student_acts[s_idx]                  # (B, T, D_s)
            t_full = teacher_acts[t_idx].to(device)       # (B, T, D_t)
            s_proj = projection(s_full)                   # (B, T, D_t)
            with torch.no_grad():
                sim = per_pair_similarity(s_proj, t_full)
            loss = _per_pair_loss(s_proj, t_full, cfg)
            per_pair_losses.append(loss)
            per_pair_sims.append(sim)
            valid_pairs.append((s_idx, t_idx))

        if not per_pair_losses:
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True), "selected": [],
                    "similarities": {}, "weights": {}}

        sims_t   = torch.stack(per_pair_sims)         # (P,)
        losses_t = torch.stack(per_pair_losses)       # (P,) — has grad

        # 2. selection / weighting
        if cfg.strategy == "argmin":
            sel = int(torch.argmin(sims_t).item())
            loss = per_pair_losses[sel]
            weights = {valid_pairs[sel]: 1.0}
            selected = [valid_pairs[sel]]

        elif cfg.strategy == "topk_soft":
            k = min(cfg.top_k, len(per_pair_losses))
            divergence = 1.0 - sims_t                                    # higher = more divergent
            topk_vals, topk_idx = torch.topk(divergence, k=k)
            # softmax over the top-K divergences with temperature; scale loss by these weights
            w = F.softmax(topk_vals / max(cfg.softmax_temp, 1e-3), dim=0)
            loss = sum(w[i] * per_pair_losses[topk_idx[i].item()] for i in range(k))
            weights = {valid_pairs[topk_idx[i].item()]: float(w[i].item()) for i in range(k)}
            selected = [valid_pairs[i.item()] for i in topk_idx]

        elif cfg.strategy == "all":
            loss = losses_t.mean()
            weights = {p: 1.0 / len(valid_pairs) for p in valid_pairs}
            selected = list(valid_pairs)

        else:
            raise ValueError(f"unknown strategy: {cfg.strategy}")

        sims_dict = {valid_pairs[i]: float(sims_t[i].item()) for i in range(len(valid_pairs))}

        return {
            "loss": loss,
            "selected": selected,
            "similarities": sims_dict,
            "weights": weights,
        }
