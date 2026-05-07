"""
Loads teacher (Qwen2.5-7B-Instruct, 4-bit, frozen) and student
(mlp-surgery-restored-top30, optionally with LoRA) on a single 4090.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ── ids + arch constants ──────────────────────────────
TEACHER_ID  = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_ID  = "Malum0x/mlp-surgery-restored-top30"

TEACHER_HIDDEN = 3584
TEACHER_LAYERS = 28
STUDENT_HIDDEN = 2048
STUDENT_LAYERS = 36
VOCAB_SIZE     = 151936


def make_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer():
    """Qwen2.5 7B and 3B share the same 151936-token vocab. Use teacher's."""
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_teacher(device: str = "cuda"):
    print(f"loading teacher: {TEACHER_ID} (4-bit nf4)")
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID,
        quantization_config=make_bnb_config(),
        device_map=device,
        attn_implementation="eager",   # hooks behave better with eager attention
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("teacher loaded - frozen")
    return model


def _default_lora_config():
    # same target set used in mlp-surgery / selective-qlora runs so the project
    # stays consistent across phases
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_student(
    device: str = "cuda",
    train: bool = False,
    use_qlora: bool = True,
    lora_config=None,
):
    """
    train=False:                    forward-only, 4-bit
    train=True + use_qlora=True:    4-bit base + bf16 LoRA adapters (default)
    train=True + use_qlora=False:   full bf16 (more memory)
    """
    print(f"loading student: {STUDENT_ID}  (train={train}, qlora={use_qlora})")
    qcfg = make_bnb_config() if (use_qlora or not train) else None
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID,
        quantization_config=qcfg,
        torch_dtype=torch.bfloat16 if not use_qlora else None,
        device_map=device,
        attn_implementation="eager",
    )

    if not train:
        model.eval()
        return model

    if use_qlora:
        model = prepare_model_for_kbit_training(model)
    cfg = lora_config or _default_lora_config()
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def load_both(device: str = "cuda", student_train: bool = False, use_qlora: bool = True):
    teacher = load_teacher(device)
    student = load_student(device, train=student_train, use_qlora=use_qlora)
    tokenizer = load_tokenizer()

    s_vocab = getattr(student.config, "vocab_size", None) if hasattr(student, "config") else None
    t_vocab = getattr(teacher.config, "vocab_size", None) if hasattr(teacher, "config") else None
    if s_vocab and t_vocab and s_vocab != t_vocab:
        print(f"WARNING: vocab mismatch — student={s_vocab} teacher={t_vocab}. "
              "Output-logit comparisons would be invalid.")

    return teacher, student, tokenizer
