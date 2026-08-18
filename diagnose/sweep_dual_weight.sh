#!/usr/bin/env bash
# dual_weight sweep: Phase 2만 돌리며 0 / 0.2 / 0.6 / 1.2 비교
# base: checkpoints/8yqlvq3v/stage3_best.pt (delta 단일 항공사, Stage 3 완료)
# 병렬: cuda:2에서 dw=0→0.2 순차, cuda:3에서 dw=0.6→1.2 순차
# 실행: bash diagnose/sweep_dual_weight.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENTS_DIR="$REPO_DIR/experiments"
CKPT_DIR="$REPO_DIR/checkpoints/8yqlvq3v"
LOG_DIR="$REPO_DIR/log/dual_weight_sweep"
PYTHON="$REPO_DIR/ascp/bin/python"

mkdir -p "$LOG_DIR"

run_seq() {
    local DEVICE=$1; shift
    for DW in "$@"; do
        LOG="$LOG_DIR/dw${DW}.out"
        echo "[${DEVICE}] dual_weight=${DW} start: $(date)" | tee -a "$LOG"
        "$PYTHON" "$EXPERIMENTS_DIR/train.py" \
            --phase2-only \
            --ckpt-dir "$CKPT_DIR" \
            --dual-weight "$DW" \
            --device "$DEVICE" \
            2>&1 | tee -a "$LOG"
        echo "[${DEVICE}] dual_weight=${DW} done: $(date)" | tee -a "$LOG"
    done
}

echo "=== dual_weight sweep start: $(date) ==="
run_seq cuda:2 0 0.2 &
PID2=$!
run_seq cuda:3 0.6 1.2 &
PID3=$!

wait $PID2 && echo "cuda:2 group done"
wait $PID3 && echo "cuda:3 group done"
echo "=== sweep done: $(date) ==="
