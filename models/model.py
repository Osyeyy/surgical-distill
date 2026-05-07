"""
Tiny utility module. Re-exports arch constants and provides a layer-finder
that survives PEFT and base_model wrappers.
"""

from .loader import (
    TEACHER_ID, STUDENT_ID,
    TEACHER_HIDDEN, TEACHER_LAYERS,
    STUDENT_HIDDEN, STUDENT_LAYERS,
    VOCAB_SIZE,
)


def find_transformer_layers(model):
    """Walk PEFT/base_model/model wrappers to return the layer ModuleList."""
    m = model
    for attr in ("base_model", "model"):
        inner = getattr(m, attr, None)
        if inner is not None:
            m = inner
    for attr in ("layers", "h", "blocks"):
        layers = getattr(m, attr, None)
        if layers is not None:
            return layers
    raise ValueError("could not locate transformer layers under model")


def count_params(model, only_trainable: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not only_trainable))
