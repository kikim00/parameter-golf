#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNOVATION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$INNOVATION_DIR/../.." && pwd)"
SUBMISSION_DIR="$INNOVATION_DIR/submissions/track_10min_16mb/2026-03-26_Hybrid_DocLocalAdapterTTT"
PY_SCRIPT="$SUBMISSION_DIR/train_gpt.py"

RUN_ID="${RUN_ID:-hybrid_submission_8xh100_$(date -u +%Y%m%dT%H%M%SZ)}"
EVIDENCE_DIR="${EVIDENCE_DIR:-$INNOVATION_DIR/submission_runs/$RUN_ID}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

mkdir -p "$EVIDENCE_DIR"

export RUN_ID
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_DIR/.cache}"
export DATA_PATH="${DATA_PATH:-$REPO_DIR/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_DIR/data/tokenizers/fineweb_1024_bpe.model}"
export ATTN_BACKEND="${ATTN_BACKEND:-auto}"

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import datasets
import sentencepiece
import tiktoken
import torch
PY
then
  echo "python environment missing required packages for submission run" >&2
  echo "python_bin=$PYTHON_BIN" >&2
  echo "run bootstrap first: bash innovations/hybrid_ttt/scripts/bootstrap_h100_pod.sh" >&2
  exit 1
fi

# March 23 leader-style defaults plus document-local adapter TTT.
export NUM_LAYERS="${NUM_LAYERS:-11}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1536}"
export XSA_LAST_N="${XSA_LAST_N:-4}"
export SWA_ENABLED="${SWA_ENABLED:-1}"
export SWA_EVERY="${SWA_EVERY:-50}"
export ROPE_DIMS="${ROPE_DIMS:-16}"
export LN_SCALE="${LN_SCALE:-1}"
export LATE_QAT_THRESHOLD="${LATE_QAT_THRESHOLD:-0.15}"
export VE_ENABLED="${VE_ENABLED:-1}"
export VE_DIM="${VE_DIM:-128}"
export VE_LAYERS="${VE_LAYERS:-9,10}"
export TTT_ENABLED="${TTT_ENABLED:-1}"
export TTT_DOC_ADAPTER_RANK="${TTT_DOC_ADAPTER_RANK:-16}"
export TTT_DOC_ADAPTER_LR="${TTT_DOC_ADAPTER_LR:-0.5}"
export TTT_DOC_ADAPTER_EPOCHS="${TTT_DOC_ADAPTER_EPOCHS:-5}"
export TTT_DOC_ADAPTER_INIT_STD="${TTT_DOC_ADAPTER_INIT_STD:-0.01}"
export TTT_DOC_ADAPTER_WEIGHT_DECAY="${TTT_DOC_ADAPTER_WEIGHT_DECAY:-0.0}"
export TTT_DOC_ADAPTER_GRAD_CLIP="${TTT_DOC_ADAPTER_GRAD_CLIP:-1.0}"
export TTT_DOC_ADAPTER_MAX_DOCS="${TTT_DOC_ADAPTER_MAX_DOCS:-0}"
export TTT_LR="${TTT_LR:-0.002}"
export TTT_EPOCHS="${TTT_EPOCHS:-1}"
export TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-32768}"
export TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-2}"
export TTT_MOMENTUM="${TTT_MOMENTUM:-0.9}"
export TTT_BATCH_SEQS="${TTT_BATCH_SEQS:-32}"
export TTT_GRAD_CLIP="${TTT_GRAD_CLIP:-1.0}"
export MUON_WD="${MUON_WD:-0.04}"
export ADAM_WD="${ADAM_WD:-0.04}"
export MATRIX_LR="${MATRIX_LR:-0.025}"
export SCALAR_LR="${SCALAR_LR:-0.025}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.035}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-1500}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}"
export ITERATIONS="${ITERATIONS:-9000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export EVAL_STRIDE="${EVAL_STRIDE:-64}"
export SEED="${SEED:-1337}"

if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "missing tokenizer: $TOKENIZER_PATH" >&2
  echo "populate data with: python3 data/cached_challenge_fineweb.py --variant sp1024" >&2
  exit 1
fi
if ! compgen -G "$DATA_PATH/fineweb_train_*.bin" > /dev/null; then
  echo "missing training shards under: $DATA_PATH" >&2
  echo "populate data with: python3 data/cached_challenge_fineweb.py --variant sp1024" >&2
  exit 1
fi
if ! compgen -G "$DATA_PATH/fineweb_val_*.bin" > /dev/null; then
  echo "missing validation shards under: $DATA_PATH" >&2
  echo "populate data with: python3 data/cached_challenge_fineweb.py --variant sp1024" >&2
  exit 1
fi

{
  echo "run_id=$RUN_ID"
  echo "utc_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "cwd=$(pwd)"
  echo "branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  echo "commit=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
  echo "openai_main=$(git -C "$REPO_DIR" rev-parse openai/main 2>/dev/null || true)"
  echo "nproc_per_node=$NPROC_PER_NODE"
  echo "python_bin=$PYTHON_BIN"
  echo "attn_backend=$ATTN_BACKEND"
  echo "hf_home=$HF_HOME"
  echo "xdg_cache_home=$XDG_CACHE_HOME"
  echo "max_wallclock_seconds=$MAX_WALLCLOCK_SECONDS"
  echo "ttt_doc_adapter_rank=$TTT_DOC_ADAPTER_RANK"
  echo "ttt_doc_adapter_epochs=$TTT_DOC_ADAPTER_EPOCHS"
  echo "ttt_doc_adapter_max_docs=$TTT_DOC_ADAPTER_MAX_DOCS"
  echo "ttt_epochs=$TTT_EPOCHS"
  echo "ttt_freeze_blocks=$TTT_FREEZE_BLOCKS"
} > "$EVIDENCE_DIR/run_meta.txt"

git -C "$REPO_DIR" status --short > "$EVIDENCE_DIR/git_status.txt" || true
if git -C "$REPO_DIR" rev-parse --verify openai/main >/dev/null 2>&1; then
  git -C "$REPO_DIR" diff --name-only openai/main...HEAD > "$EVIDENCE_DIR/git_diff_vs_openai_main.txt" || true
else
  echo "openai/main not available in this clone" > "$EVIDENCE_DIR/git_diff_vs_openai_main.txt"
fi
nvidia-smi -L > "$EVIDENCE_DIR/nvidia_smi_L.txt"
nvidia-smi topo -m > "$EVIDENCE_DIR/nvidia_smi_topo.txt" || true
nvidia-smi > "$EVIDENCE_DIR/nvidia_smi.txt"
"$PYTHON_BIN" - <<'PY' > "$EVIDENCE_DIR/python_torch_env.txt"
import os, sys, torch
print("python", sys.version)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    try:
        print("device", i, torch.cuda.get_device_name(i))
    except Exception as exc:
        print("device", i, f"<unavailable: {exc}>")
print("env_run_id", os.getenv("RUN_ID"))
PY

TRAIN_LOG="$REPO_DIR/logs/$RUN_ID.txt"
STDOUT_LOG="$EVIDENCE_DIR/stdout.log"
TIME_LOG="$EVIDENCE_DIR/process_time.txt"

cd "$REPO_DIR"
echo "=== nvidia-smi -L ===" | tee "$STDOUT_LOG"
nvidia-smi -L | tee -a "$STDOUT_LOG"
echo | tee -a "$STDOUT_LOG"

if [[ -x /usr/bin/time ]]; then
  /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kb=%M' -o "$TIME_LOG" \
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "$PY_SCRIPT" 2>&1 | tee -a "$STDOUT_LOG"
else
  start_ts="$("$PYTHON_BIN" - <<'PY'
import time
print(time.time())
PY
)"
  set +e
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "$PY_SCRIPT" 2>&1 | tee -a "$STDOUT_LOG"
  torchrun_status=${PIPESTATUS[0]}
  set -e
  end_ts="$("$PYTHON_BIN" - <<'PY'
import time
print(time.time())
PY
)"
  "$PYTHON_BIN" - "$start_ts" "$end_ts" > "$TIME_LOG" <<'PY'
import sys
start = float(sys.argv[1])
end = float(sys.argv[2])
print(f"elapsed_seconds={end - start:.3f}")
print("max_rss_kb=unavailable")
PY
  if [[ $torchrun_status -ne 0 ]]; then
    exit "$torchrun_status"
  fi
fi

if [[ -f "$TRAIN_LOG" ]]; then
  cp "$TRAIN_LOG" "$EVIDENCE_DIR/train_log.txt"
fi

for artifact in "$REPO_DIR/final_model.pt" "$REPO_DIR/final_model.int6.ptz"; do
  if [[ -f "$artifact" ]]; then
    sha256sum "$artifact" >> "$EVIDENCE_DIR/artifact_sha256.txt"
    ls -l "$artifact" >> "$EVIDENCE_DIR/artifact_sizes.txt"
  fi
done

{
  echo "=== timing ==="
  cat "$TIME_LOG"
  echo
  echo "=== key log lines ==="
  if [[ -f "$TRAIN_LOG" ]]; then
    rg -n "world_size:|train_batch_tokens:|step:.* val_loss|stopping_early|Serialized model int6\\+lzma|Total submission size int6\\+lzma|final_int6_roundtrip_exact|final_int6_sliding_window_exact|final_int6_sliding_window_s64_exact|legal_ttt_exact|legal_ttt_doc_adapter_exact|eval_time:|distributed mode uses synchronized base-model TTT updates" "$TRAIN_LOG" || true
  else
    echo "train log not found: $TRAIN_LOG"
  fi
} > "$EVIDENCE_DIR/summary.txt"

echo "Evidence written to: $EVIDENCE_DIR"
