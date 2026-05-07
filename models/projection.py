"""
Learned linear projection from student hidden dim (2048) to teacher hidden dim (3584).

The projection is trained jointly with the distillation loop and encodes structural
correspondence between the two architectures. Not a throwaway component.
"""

import torch
import torch.nn as nn

from .loader import STUDENT_HIDDEN, TEACHER_HIDDEN


class HiddenProjection(nn.Module):
    """Projects student activations into teacher space for activation comparison."""

    def __init__(
        self,
        student_dim: int = STUDENT_HIDDEN,
        teacher_dim: int = TEACHER_HIDDEN,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
        self.norm = nn.LayerNorm(teacher_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, D_s) or (B, T, D_s)
        out: same leading shape, last dim = teacher_dim
        """
        return self.dropout(self.norm(self.proj(x)))

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"projection saved -> {path}")

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"projection loaded <- {path}")
