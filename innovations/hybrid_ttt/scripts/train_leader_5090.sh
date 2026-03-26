#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORD_SCRIPT="$SCRIPT_DIR/../leader_2026_03_23_train_gpt.py"

export RUN_ID="${RUN_ID:-runpod_5090_leader_train}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-32768}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}"
export ITERATIONS="${ITERATIONS:-1000}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-131072}"
export TTT_ENABLED="${TTT_ENABLED:-0}"

python3 "$RECORD_SCRIPT"
