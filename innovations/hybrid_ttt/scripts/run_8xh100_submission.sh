#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNOVATION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$INNOVATION_DIR/../.." && pwd)"
PY_SCRIPT="$INNOVATION_DIR/leader_2026_03_23_train_gpt.py"

RUN_ID="${RUN_ID:-submission_8xh100_$(date -u +%Y%m%dT%H%M%SZ)}"
EVIDENCE_DIR="${EVIDENCE_DIR:-$INNOVATION_DIR/submission_runs/$RUN_ID}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

mkdir -p "$EVIDENCE_DIR"

export RUN_ID
export DATA_PATH="${DATA_PATH:-$REPO_DIR/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_DIR/data/tokenizers/fineweb_1024_bpe.model}"
export ATTN_BACKEND="${ATTN_BACKEND:-auto}"

# March 23 leader-style defaults. This is the legal-ready 8xH100 path today.
# Keep TTT_DOC_ADAPTER_RANK=0 here because the hybrid doc-adapter path remains
# single-GPU-only in this innovation copy.
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
export TTT_DOC_ADAPTER_RANK="${TTT_DOC_ADAPTER_RANK:-0}"
export TTT_LR="${TTT_LR:-0.002}"
export TTT_EPOCHS="${TTT_EPOCHS:-3}"
export TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-32768}"
export TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-0}"
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

{
  echo "run_id=$RUN_ID"
  echo "utc_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "cwd=$(pwd)"
  echo "branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  echo "commit=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
  echo "openai_main=$(git -C "$REPO_DIR" rev-parse openai/main 2>/dev/null || true)"
  echo "nproc_per_node=$NPROC_PER_NODE"
  echo "attn_backend=$ATTN_BACKEND"
  echo "max_wallclock_seconds=$MAX_WALLCLOCK_SECONDS"
} > "$EVIDENCE_DIR/run_meta.txt"

git -C "$REPO_DIR" status --short > "$EVIDENCE_DIR/git_status.txt" || true
git -C "$REPO_DIR" diff --name-only openai/main...HEAD > "$EVIDENCE_DIR/git_diff_vs_openai_main.txt" || true
nvidia-smi -L > "$EVIDENCE_DIR/nvidia_smi_L.txt"
nvidia-smi topo -m > "$EVIDENCE_DIR/nvidia_smi_topo.txt" || true
nvidia-smi > "$EVIDENCE_DIR/nvidia_smi.txt"
python3 - <<'PY' > "$EVIDENCE_DIR/python_torch_env.txt"
import os, sys, torch
print("python", sys.version)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("device", i, torch.cuda.get_device_name(i))
print("env_run_id", os.getenv("RUN_ID"))
PY

TRAIN_LOG="$REPO_DIR/logs/$RUN_ID.txt"
STDOUT_LOG="$EVIDENCE_DIR/stdout.log"
TIME_LOG="$EVIDENCE_DIR/process_time.txt"

cd "$REPO_DIR"
/usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kb=%M' -o "$TIME_LOG" \
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "$PY_SCRIPT" 2>&1 | tee "$STDOUT_LOG"

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
    rg -n "world_size:|train_batch_tokens:|step:.* val_loss|stopping_early|Serialized model int6\\+lzma|Total submission size int6\\+lzma|final_int6_roundtrip_exact|final_int6_sliding_window_exact|final_int6_sliding_window_s64_exact|legal_ttt_exact|legal_ttt_doc_adapter_exact|eval_time:" "$TRAIN_LOG" || true
  else
    echo "train log not found: $TRAIN_LOG"
  fi
} > "$EVIDENCE_DIR/summary.txt"

echo "Evidence written to: $EVIDENCE_DIR"
