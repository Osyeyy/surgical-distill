# Activation-guided online distillation (Qwen2.5 7B → 3B)

An honest negative result, with the diagnostics.

## TL;DR

I built an online distillation loop that compares teacher and student hidden states layer-by-layer at every batch and applies a divergence-localized loss. v1 (no protection on the layers feeding the lm_head) **catastrophically broke the model** — GSM8K dropped 43 points. v1.1 (protected last 4 layers + output-level KL anchor) recovered to baseline. A clean three-arm comparison (KL-only, activation-leaning, full method) showed all variants statistically tied with the unmodified starting model.

The method as configured here doesn't help. The failure modes are clean and diagnosable. Code, configs, and full experimental log are below so anyone can reproduce.

## Setup

- **Teacher:** [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), loaded in 4-bit nf4, frozen
- **Student:** [Malum0x/mlp-surgery-restored-top30](https://huggingface.co/Malum0x/mlp-surgery-restored-top30) — a Qwen2.5-3B-Instruct previously SFT-damaged and recovered via the [mlp-surgery](https://github.com/Malum0x/mlp-surgery) project
- **Hardware:** single RTX 4090 (24 GB)

The student is intentionally not vanilla Qwen2.5-3B-Instruct. Starting from the restored checkpoint keeps attribution clean (we're testing whether distillation pushes past base, not whether *any* training does).

## Architecture (same family, different shape)

|                    | Teacher (Qwen2.5-7B-Instruct) | Student (restored_top30, Qwen2.5-3B) |
|--------------------|-------------------------------|--------------------------------------|
| hidden_size        | 3584                          | 2048                                 |
| num_hidden_layers  | 28                            | 36                                   |
| vocab (real tokens)| 151,936                       | 151,936                              |
| activation         | SwiGLU                        | SwiGLU                               |
| RoPE               | yes                           | yes                                  |

Note the inversion: student has *more* layers than teacher (36 vs 28). Layer correspondence is many-to-few — student layer i maps to teacher layer `round(i * 28/36)`.

The 7B teacher's embedding is padded to vocab_size 152064 (next multiple of 128) but only the first 151,936 are real tokens. Output-logit comparisons must slice both sides to `[:151936]`.

## Method

For every training batch:

```
teacher.forward(no grad)         -> capture per-layer hidden states + logits
student.forward(with grad)       -> capture per-layer hidden states + logits
project student activations -> teacher hidden space
for each (s_idx, t_idx) in layer_map:
    compute (1 - cos) * w_cos + MSE * w_mse
selector picks loss strategy:
    'argmin'      = single most-divergent layer
    'topk_soft'   = softmax-weighted top-K (default)
    'all'         = TinyBERT-style uniform
output-level KL on shared vocab (151,936) anchors the lm_head
total = act_weight * activation_loss + kl_weight * kl_loss
loss.backward(); optimizer.step()
```

### Components

- **`models/loader.py`** — Loads teacher (4-bit) and student (4-bit + LoRA via QLoRA). Vocab sanity check.
- **`models/projection.py`** — Learned linear (2048 → 3584) + LayerNorm. Trained in Phase 1, optionally kept trainable through Phase 2.
- **`distill/hooks.py`** — Forward hooks capturing per-layer hidden states; PEFT-aware layer finder (walks PeftModel.base_model.model.model.layers iteratively).
- **`distill/alignment.py`** — Layer correspondence map. Proportional initial seed plus `last_n_skip` to protect lm_head-feeding layers.
- **`scoring/divergence.py`** — DivergenceSelector with three strategies (argmin / topk_soft / all). Combined cosine + MSE loss.
- **`scoring/localization/loop.py`** — DistillStep wrapper used by both Phase 1 and Phase 2.
- **`scripts/train_projection.py`** — Phase 1: projection-only training, both models frozen.
- **`scripts/train.py`** — Phase 2: LoRA on student + divergence-localized loss + optional KL co-loss.
- **`scoring/eval/baseline.py`** — lm-eval wrapper matching mlp-surgery eval settings (GSM8K flexible-extract 5-shot, ARC Challenge acc_norm 0-shot, no chat template, batch_size 8).
- **`comparison/diagnostic/run_diagnostic.py`** + `plot.py` — Phase 0 baseline divergence map.

## Phases

| Phase | Script | What |
|-------|--------|------|
| 0 — Diagnostic | `comparison/diagnostic/run_diagnostic.py` | Untrained projection baseline; should show ~0 cosine across all layer pairs |
| 1 — Projection only | `scripts/train_projection.py` | Train projection; freeze both models. Output: `logs/projection.pt` |
| 2 — Online distill | `scripts/train.py` | LoRA on student + divergence-localized loss |
| 3 — Eval | `scoring/eval/baseline.py` | GSM8K + ARC against three arms |

## Results

### Phase 0 — baseline (random projection)

Mean cosine similarity across all student-teacher layer pairs: **≈ 0.0**. As expected. Random projection produces no alignment; this is the floor.

### Phase 1 — projection-only training (2000 steps, batch 2, lr 1e-3)

Mean cosine similarity after training: **≈ 0.91**. Most layer pairs above 0.92, with a small set of structurally harder pairs (s2-3 → t2 at ~0.76, s30-32 at ~0.83-0.85, s35 → t27 at ~0.88). The projection learned a real bridge.

### Phase 2 — full distillation, four configurations

All three trained variants use `last_n_skip=4` (protect layers 32-35 from activation pull) and KL co-loss for lm_head anchoring. Trained 2000 steps each, evaluated on the same lm-eval suite.

| Configuration | GSM8K | ARC Challenge | ΔGSM | ΔARC |
|---|---:|---:|---:|---:|
| Baseline (restored_top30, no training) | 64.29% | 48.55% | — | — |
| **v1.0** activation-only, no protection | 21.08% | 34.04% | **-43.21** | **-14.51** |
| **v1.1** activation + KL (kl-dominant) | 65.20% | 47.95% | +0.91 | -0.60 |
| **B** KL-only (act_weight=0) | 63.91% | 47.95% | -0.38 | -0.60 |
| **C** activation-leaning (kl_weight=0.005) | 64.59% | 48.21% | +0.30 | -0.34 |

### What the result tells us

1. **Activation-localized distillation without lm_head protection catastrophically damages the model.** Forcing the student's last hidden states toward (a projected version of) the teacher's hidden states breaks the student's own lm_head — it expects vectors from the distribution it was trained on, gets vectors dragged toward a different one, can no longer decode. Confirmed by v1.0 → -43 GSM8K.

2. **With `last_n_skip=4` and an output-level KL anchor, the method preserves baseline performance.** All three trained variants (v1.1, B, C) land within ±1pt of baseline — well inside single-seed noise floor.

3. **Activation-localized supervision did not measurably improve over plain KL distillation at this compute budget.** C vs B is within noise. The localization didn't add information.

4. **Plain KL distillation also did not measurably improve at this compute budget.** B vs baseline is within noise. At 2000 steps with lr=1e-5, no form of distillation moved the student.

### Selection histograms

In v1.0 (no protection), the topk_soft selector concentrated 92% of attention on layer pair `35→27` — the layer that feeds the lm_head. This is what caused the collapse: the loss was dominated by exactly the layer that should never have been touched.

In v1.1 onward (with `last_n_skip=4`), selection spread cleanly across the remaining 32 layer pairs (29 of 32 selected at least once in arm C; 32/32 in arm B which uses uniform `all` strategy).

## Reproduction

```bash
# Phase 0 — diagnostic
python data/calibration_ready.py --n 100
python comparison/diagnostic/run_diagnostic.py --prompts data/calibration.txt --n 100

# Phase 1 — projection-only training
python scripts/train_projection.py --prompts data/calibration.txt --steps 2000 --output logs/projection.pt

# Phase 2 — full distillation (the v1.1 config)
python scripts/train.py --config configs/default.json --projection logs/projection.pt --output_dir logs/phase2_v1_1

# Three-arm comparison (B = KL-only, C = activation-leaning)
bash run_arms_bc.sh

# Eval any checkpoint
python scoring/eval/baseline.py \
    --model_arg "pretrained=Malum0x/mlp-surgery-restored-top30,peft=$(pwd)/logs/phase2_v1_1/final" \
    --output_dir logs/phase2_v1_1/eval
```

## Honest assessment

The method as designed doesn't help. Three-arm comparison rules out reasonable confounds — we can specifically say "the localization is not adding signal beyond what plain KL provides," not just "training didn't work."

Plausible reasons for the negative outcome:

- 2000 steps × batch=1 = small training budget. Successful layer-wise distillation in the literature (TinyBERT, MobileBERT, MiniLM) typically uses orders of magnitude more compute and carefully co-designed teacher-student pairs.
- lr=1e-5 with LoRA on an already-strong starting checkpoint is conservative.
- KL on chat-style data doesn't directly target GSM8K/ARC.
- 4-bit teacher quantization may be too lossy for activation matching to add signal beyond what KL captures.

What's worth taking from this anyway:

- The `last_n_skip` + KL-anchor pattern is a transferable lesson for anyone doing layer-wise hidden-state matching across mismatched architectures.
- The 92%-concentration failure mode of `argmin`-style localization is a clean negative datapoint for anyone considering similar designs.
- The three-arm experimental structure (no-train / KL-only / KL+method) is the right shape for honest distillation method evaluation.

## Roadmap (not run)

If anyone wants to push this further:

- **v2:** scale up the activation-leaning configuration. 10-20× steps, slightly higher lr (e.g. 5e-5), maybe 2-4× batch.
- **v3:** dense → MoE student (e.g. OLMoE-1B-7B). The expert-specialization angle is novel.
- **v4:** cross-family or cross-scale (Qwen → Mistral / Llama, or 30B → 3B).

Worth doing only if v2's signal isn't a clean tie like v1.

## Related

- **Sister projects** (the connected arc):
  - [Perplexity-weighted-selective-finetuning](https://github.com/Malum0x/Perplexity-weighted-selective-finetuning) — found that perplexity filtering doesn't protect reasoning during SFT.
  - [mlp-surgery](https://github.com/Malum0x/mlp-surgery) — gradient-norm scoring + MLP-layer restoration. The student in this project is an output of mlp-surgery.
  - [layer-vision](https://github.com/Malum0x/layer-vision) — real-time MLP activation visualizer.
- **References:**
  - ROME (Meng et al., 2022) — factual knowledge in MLP layers
  - TinyBERT, MiniLM — layer-wise distillation
  - MiniLLM, DistiLLM — adaptive online distillation

The arc: where information lives in a model matters more than how much data you train on. mlp-surgery succeeded because it **restored** what was already there. surgical-distill did not, because it tried to **add** something the student couldn't accept.
