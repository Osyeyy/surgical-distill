#!/bin/bash
# Sequentially train + eval arm B (KL-only) and arm C (activation-leaning).
set -uo pipefail
cd /home/bart/Desktop/subnets/my_projects/surgical-distill
LOGDIR=logs/arms_bc
mkdir -p "$LOGDIR"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/master.log"; }

PROJ=logs/projection.pt

# ── arm B ─────────────────────────────────────
log "==== arm B: KL-only ===="
log "training..."
rm -rf logs/arm_b
python scripts/train.py --config configs/kl_only.json --projection "$PROJ" \
    --output_dir logs/arm_b > "$LOGDIR/arm_b_train.log" 2>&1
log "training done"

log "eval..."
python scoring/eval/baseline.py \
    --model_arg "pretrained=Malum0x/mlp-surgery-restored-top30,peft=$(pwd)/logs/arm_b/final" \
    --output_dir logs/arm_b/eval > "$LOGDIR/arm_b_eval.log" 2>&1
log "arm B done — scores at logs/arm_b/eval/scores.json"

# ── arm C ─────────────────────────────────────
log ""
log "==== arm C: activation-leaning ===="
log "training..."
rm -rf logs/arm_c
python scripts/train.py --config configs/act_leaning.json --projection "$PROJ" \
    --output_dir logs/arm_c > "$LOGDIR/arm_c_train.log" 2>&1
log "training done"

log "eval..."
python scoring/eval/baseline.py \
    --model_arg "pretrained=Malum0x/mlp-surgery-restored-top30,peft=$(pwd)/logs/arm_c/final" \
    --output_dir logs/arm_c/eval > "$LOGDIR/arm_c_eval.log" 2>&1
log "arm C done — scores at logs/arm_c/eval/scores.json"

log ""
log "==== ALL ARMS COMPLETE ===="
