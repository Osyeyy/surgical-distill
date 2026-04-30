"""
Loads student and teacher models in 4-bit NF4 quantization.
teacher is always frozen. student is prepared for LoRA.
"""

import torch
from transformers import AutoModelforCausalLM, AutoTokenizer, BitsandBytesConfig
from peft import prepare_model_for_kbit_training

STUDENT_ID = "silx-ai/Quasar-3B-A1B-Preview"
TEACHER_ID = "Qwen/Qwen3_5-4B"

BNB_CONFIG = BitsandBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

def load_tokenizer():
    """
    Both models share vocab (248320 tokens), use teacher tokenizer 
    verify student tokenizer matches before trusting this
    """

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer 

def load_teacher(device: str = "cuda"):
    print(f"loading teacher: {TEACHER_ID}") 
    model = AutoModelforCausalLM.from_pretrained(
        TEACHER_ID,
        quantization_config=BNB_CONFIG,
        device_map=device,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
    )
    model.eval() 
    for param in model.parameters():
        param.requires_grad = False
    print(f"teacher loaded - frozen") 
    return model

def load_student(device: str = "cuda", prepare_for_lora: bool = False):
    print(f"loading student: {STUDENT_ID}")
    model = AutoModelforCausalLM.from_pretrained(
        STUDENT_ID,
        quantization_config=BNB_CONFIG,
        device_map=device, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
    )
    if prepare_for_lora:
        model = prepare_model_for_kbit_training(model)
        print("student prepared for LoRA training")
    else: 
        model.eval() 
    return model

def load_both(device: str = "cuda", student_train: bool = False):
    """
    Loads teacher (frozen) and student on the same device
    on a single 4090 both fit in 4-bit simultaneously
    """

    teacher = load_teacher(device)
    student = load_student(device, prepare_for_lora=student_train)
    tokenizer = load_tokenizer()
    return teacher, student, tokenizer

