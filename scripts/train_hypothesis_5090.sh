#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-runpod_5090_hypothesis_train}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"

export RUN_ID
export DATA_PATH
export TOKENIZER_PATH
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export ITERATIONS="${ITERATIONS:-2000}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-131072}"

python3 train_gpt.py
