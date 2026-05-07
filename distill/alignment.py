"""
Layer correspondence map between student and teacher.

Initial mapping is proportional (student layer i → teacher layer round(i * L_t/L_s)).
After Phase 0 diagnostic, an optional refinement step rewires the mapping to the
teacher layer with highest baseline similarity for each student layer.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Optional

from models.loader import STUDENT_LAYERS, TEACHER_LAYERS


def proportional_map(n_student: int = STUDENT_LAYERS, n_teacher: int = TEACHER_LAYERS) -> Dict[int, int]:
    """student i -> teacher round(i * L_t / L_s).  Endpoints align (0→0, L_s-1 → L_t-1)."""
    return {i: int(round(i * n_teacher / n_student)) for i in range(n_student)}


def refine_from_similarity_matrix(
    sim_matrix: List[List[float]],
    monotonic: bool = True,
) -> Dict[int, int]:
    """
    sim_matrix[s_idx][t_idx] = mean cosine similarity for that pair.
    For each student layer pick its argmax teacher partner.

    If monotonic=True, enforce t(s) >= t(s-1) (no out-of-order rewires).
    Useful because language-model layers do typically follow a coarse depth ordering.
    """
    L_s = len(sim_matrix)
    L_t = len(sim_matrix[0]) if L_s else 0
    mapping: Dict[int, int] = {}
    last = -1
    for s in range(L_s):
        if monotonic:
            candidates = list(range(max(0, last), L_t))
            best = max(candidates, key=lambda t: sim_matrix[s][t])
        else:
            best = max(range(L_t), key=lambda t: sim_matrix[s][t])
        mapping[s] = best
        last = best
    return mapping


def save_map(mapping: Dict[int, int], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)


def load_map(path: str) -> Dict[int, int]:
    with open(path) as f:
        d = json.load(f)
    return {int(k): v for k, v in d.items()}


def get_or_make_map(path: Optional[str]) -> Dict[int, int]:
    if path and os.path.exists(path):
        print(f"alignment: loaded refined map from {path}")
        return load_map(path)
    print("alignment: using proportional initial map")
    return proportional_map()
