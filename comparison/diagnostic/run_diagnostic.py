"""
Runs both models on the same prompts, captures activations at every layer,
projects student -> teacher space, computes cosine similarity per layer pair
Outputs a similarity matrix and saves raw results for plotting

run before any training

Usage: 
python diagnostic/run_diagnostic.py --prompts data/calibration.txt --n 100

"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loader import load_both
from models.hooks import ActivationCapture
from models.projection import HiddenProjection

# layer counts

STUDENT_LAYERS = 24
TEACHER_LAYERS = 32

# proportional layer correspondence: student i -> teacher j
def default_layer_mlp():
    return {
        i: int(i * TEACHER_LAYERS / STUDENT_LAYERS)
        for i in range(STUDENT_LAYERS)
    }

@torch.no_grad()
def run_single(model, tokenizer, prompt: str, max_length: int = 256, device: str = "cuda"):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        model(**inputs)
    return inputs

def compute_similarity(
        student_acts, # list of batch, hidden tensors, len=24
        teacher_acts, # list of batch, hidden tensors, len=32
        projection, # Hidden projection
        layer_map, #dict student_layer  -> teacher_layer
        device, 
):

    """
    for each student layer, project to teacher space,
    compute cosine similarity with the corresponding teacher layer.
    returns dict: student_layer_idx -> cosine_similarity (scalar)
    
    """

    similarities = {}
    for s_idx, t_idx in layer_map.items(): 
        if s_idx >= len(studnet_acts) or t_idx >= len(teacher_acts):
            continue
        s_act = student_acts[s_idx].to(device) # batch, 1536
        t_act = teacher_acts[t_idx].to(device) # batch, 2560
        s_proj = projection(s_act) # batch, 2560
        sim = F.cosine_similarity(s_proj, t_act, dim=-1).mean().item()
        similarities[s_idx] = sim
    return similarities

def run_diagnostic(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # load models
    teacher, student, tokenizer = load_both(device=device, student_train=False)
 
    # load or init projection (untrained at this stage — that's the point)
    projection = HiddenProjection(student_dim=1536, teacher_dim=2560).to(device)
    if args.projection_path and os.path.exists(args.projection_path):
        projection.load(args.projection_path)
        print("using saved projection")
    else:
        print("using untrained projection (random init) — expected for phase 0")
 
    # load prompts
    with open(args.prompts) as f:
        prompts = [line.strip() for line in f if line.strip()]
    prompts = prompts[:args.n]
    print(f"running diagnostic on {len(prompts)} prompts")
 
    layer_map = default_layer_map()
    print(f"layer map (student → teacher): {layer_map}")
 
    # accumulate similarities per layer across all prompts
    accumulated = {i: [] for i in range(STUDENT_LAYERS)}
 
    student_capture = ActivationCapture(student)
    teacher_capture = ActivationCapture(teacher)
 
    for prompt in tqdm(prompts, desc="diagnostic"):
        student_capture.attach()
        run_single(student, tokenizer, prompt, device=device)
        student_acts = student_capture.get_layer_means()
        student_capture.detach()
 
        teacher_capture.attach()
        run_single(teacher, tokenizer, prompt, device=device)
        teacher_acts = teacher_capture.get_layer_means()
        teacher_capture.detach()
 
        sims = compute_similarity(
            student_acts, teacher_acts, projection, layer_map, device
        )
        for s_idx, sim in sims.items():
            accumulated[s_idx].append(sim)
 
    # average across prompts
    mean_similarities = {
        s_idx: sum(vals) / len(vals)
        for s_idx, vals in accumulated.items()
        if vals
    }
 
    # print results
    print("\n--- baseline divergence map ---")
    print(f"{'student layer':<15} {'teacher layer':<15} {'cosine sim':<12} {'divergence'}")
    print("-" * 55)
    for s_idx in sorted(mean_similarities.keys()):
        t_idx = layer_map[s_idx]
        sim = mean_similarities[s_idx]
        bar = "█" * int((1 - sim) * 20)
        print(f"{s_idx:<15} {t_idx:<15} {sim:<12.4f} {bar}")
 
    # save results
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "layer_map": layer_map,
        "mean_similarities": {str(k): v for k, v in mean_similarities.items()},
        "n_prompts": len(prompts),
        "projection_trained": args.projection_path is not None,
    }
    out_path = os.path.join(args.output_dir, "diagnostic_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nresults saved → {out_path}")
    print("run diagnostic/plot.py to visualize")
 
    return mean_similarities
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, required=True,
                        help="path to calibration prompts (.txt, one per line)")
    parser.add_argument("--n", type=int, default=100,
                        help="number of prompts to use")
    parser.add_argument("--output_dir", type=str, default="logs/diagnostic")
    parser.add_argument("--projection_path", type=str, default=None,
                        help="optional: path to saved projection weights")
    args = parser.parse_args()
    run_diagnostic(args)
 