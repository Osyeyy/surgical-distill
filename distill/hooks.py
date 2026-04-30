""" 
Attaches forward hooks to capture hidden state activations at the output of each transformer layer for both models 

"""

import torch
from typing import Dict, List

class ActivationCapture:
    """
    Attaches hooks to amodel and captures the hidden state
    at the output of each transfromer layer

    """

    def __init__(self, model, layer_path: str = "auto"):
        self.model = model
        self.activations: List[torch.Tensor] = []
        self._handles = []
        self._layer_path = layer_path

    def _get_layers(self):
        """
        Attempt to automatically find transformer layers.
        handles commoon naming conventions: model.layers, transformer.h, etc.
        """

        m = self.model
        if hasattr(m, "base_model"):
            m = m.base_model
        if hasattr(m, "model"): 
            m = m.model
        
        for attr in ["layers", "h", "blocks", "transformer"]:
            if hasattr(m, attr):
                layers = getattr(m, attr)
                if isinstance(layers, torch.nn.ModuleList):
                    return layers
                
        raise ValueError(
            "Could not aut-detect transformer layers. "
            "Pass  layer_path='model.layers' or similar explicitly." 
        )
    
    def _hook_fn(self, layer_idx: int):
        def fn(module, input, output):
            # output is typically (hidden_state, ...) or just hidden_state

            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # store mean over sequence length -> shape (batch, hidden_dim)
            self.activations.append(hidden.detach().float())
        return fn
    
    def attach(self):
        self.activations = []
        layers = self._get_layers()
        for idx, layer in enumerate(layers):
            handle = layer.register_forward_hook(self._hook_fn(idx))
            self._handles.append(handle)
    
    def detach(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.detach()

    def get_layer_means(self) -> List[torch.Tensor]:
        """
        Returs mean-pooled activation per layer
        shape per layer: (batch, hidden_dim)
        """

        return [act.mean(dim=1) for act in self.activations]
    
