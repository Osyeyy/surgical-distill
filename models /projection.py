"""
Learned linear projection from student hidden dimension (1536)
to teacher hidden dimension (2560)

The projection encodes the structural correspondence between the two architectures
and is trained jointly with the distillation loop.
"""


import torch
import torch.nn as nn

class HiddenProjection(nn.Module): 
    """
    Projects student activations into teacher space for comparison. 
    """

    def __init__(
        self,
        student_dim: int = 1536,
        teacher_dim: int = 2560,
        dropout: float = 0.0, 
    ):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
        self.norm = nn.LayerNorm(teacher_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # initialize to near-identity behavior (scaled)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, student_hidden_dim) or (batch, seq_len, student_hidden_dim)
        returns: same shape with last dim = teacher_hidden_dim
        """

        return self.dropout(self.norm(self.proj(x)))
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"projection saved -> {path} ")

        def load(self, path: str):
            self.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"projection loaded <- {path}") 
        
