"""
Forward hooks for capturing per-layer hidden state activations on both models.

Provides ActivationCapture: attach to a model, run forward(), read .activations.
Layers are auto-discovered by walking standard wrappers (base_model, model, layers).
"""

import torch
from typing import List


class ActivationCapture:
    """Attaches forward hooks to each transformer layer to record its hidden state output."""

    def __init__(self, model, layer_path: str = "auto",
                 detach: bool = True, cast_to_float: bool = True):
        """
        detach=True:   for teacher / inference. Saves memory, severs grad.
        detach=False:  for student during training — keeps the graph.
        cast_to_float: upcasts hidden states to float32 for stable cosine math.
                       Grad still flows when detach=False.
        """
        self.model = model
        self.activations: List[torch.Tensor] = []
        self._handles = []
        self._layer_path = layer_path
        self._detach = detach
        self._cast = cast_to_float

    def _get_layers(self):
        """Walk PEFT / base_model / model wrappers to find a ModuleList of layers."""
        m = self.model
        for attr in ("base_model", "model"):
            inner = getattr(m, attr, None)
            if inner is not None:
                m = inner
        for attr in ("layers", "h", "blocks"):
            layers = getattr(m, attr, None)
            if layers is not None and isinstance(layers, torch.nn.ModuleList):
                return layers
        raise ValueError(
            "could not auto-detect transformer layers. "
            "pass layer_path='model.layers' explicitly."
        )

    def _hook_fn(self, layer_idx: int):
        def fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self._detach:
                hidden = hidden.detach()
            if self._cast:
                hidden = hidden.float()
            self.activations.append(hidden)
        return fn

    def attach(self):
        self.activations = []
        layers = self._get_layers()
        for idx, layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(self._hook_fn(idx)))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.detach()

    def get_layer_means(self) -> List[torch.Tensor]:
        """Mean-pool over sequence length. Each item: (B, D)."""
        return [act.mean(dim=1) for act in self.activations]

    def get_full(self) -> List[torch.Tensor]:
        """Full per-token activations. Each item: (B, T, D)."""
        return list(self.activations)
