"""
DistillStep — one batch of online activation-localized distillation.

Sequence per batch:
  1. teacher forward (no_grad, detached hooks)
  2. student forward (with grad, hooks keep the graph)
  3. selector (scoring/divergence.py) returns a loss + diagnostics
  4. caller does loss.backward() + optimizer.step()

Single source of truth for the forward portion; both Phase 1 (projection-only)
and Phase 2 (full distill) reuse it.
"""

from __future__ import annotations
import os
import sys
from typing import Dict

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from distill.hooks import ActivationCapture
from scoring.divergence import DivergenceSelector


class DistillStep:
    def __init__(self, teacher, student, projection,
                 selector: DivergenceSelector, layer_map: Dict[int, int],
                 device: str = "cuda"):
        self.teacher = teacher
        self.student = student
        self.projection = projection
        self.selector = selector
        self.layer_map = layer_map
        self.device = device

        self._teacher_capture = ActivationCapture(teacher, detach=True)
        self._student_capture = ActivationCapture(student, detach=False)

    @torch.no_grad()
    def _teacher_forward(self, batch):
        self._teacher_capture.attach()
        out = self.teacher(**batch)
        acts = self._teacher_capture.get_full()
        self._teacher_capture.detach()
        return acts, out.logits.detach()

    def _student_forward(self, batch):
        self._student_capture.attach()
        out = self.student(**batch)
        acts = self._student_capture.get_full()
        self._student_capture.detach()
        return acts, out.logits

    def step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        batch: dict of input_ids / attention_mask, already on device.
        Returns: {'loss', 'selected', 'similarities', 'weights',
                  'student_logits', 'teacher_logits'}.
        Caller can add output-level losses (KL etc.) using the logits.
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        teacher_acts, teacher_logits = self._teacher_forward(batch)
        student_acts, student_logits = self._student_forward(batch)
        out = self.selector(student_acts, teacher_acts, self.projection, self.layer_map)
        out["student_logits"] = student_logits
        out["teacher_logits"] = teacher_logits
        return out
