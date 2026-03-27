#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNOVATION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$INNOVATION_DIR/../.." && pwd)"

VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VARIANT="${VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu128}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/$TORCH_CUDA_TAG}"
INSTALL_FA3="${INSTALL_FA3:-1}"
FA3_DIR="${FA3_DIR:-$REPO_DIR/.deps/flash-attention}"
FA3_LOG="${FA3_LOG:-$REPO_DIR/logs/fa3_build.log}"
MAX_JOBS="${MAX_JOBS:-1}"

export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_DIR/.cache}"
export DATA_PATH="${DATA_PATH:-$REPO_DIR/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_DIR/data/tokenizers/fineweb_1024_bpe.model}"

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$REPO_DIR/logs" "$(dirname "$FA3_DIR")"

echo "== persistent paths =="
echo "repo_dir=$REPO_DIR"
echo "venv_dir=$VENV_DIR"
echo "hf_home=$HF_HOME"
echo "xdg_cache_home=$XDG_CACHE_HOME"
echo "data_path=$DATA_PATH"
echo "tokenizer_path=$TOKENIZER_PATH"
echo
df -h "$REPO_DIR" | sed -n '1,2p'
echo

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade \
  numpy \
  tqdm \
  huggingface-hub \
  kernels \
  typing-extensions==4.15.0 \
  datasets \
  tiktoken \
  sentencepiece \
  packaging \
  psutil \
  ninja

if ! TORCH_VERSION="$TORCH_VERSION" TORCH_CUDA_TAG="$TORCH_CUDA_TAG" python - <<'PY'
import os
import sys

torch_version = os.environ["TORCH_VERSION"]
cuda_tag = os.environ["TORCH_CUDA_TAG"].removeprefix("cu")
try:
    import torch
except Exception:
    raise SystemExit(1)
ok = torch.__version__.startswith(f"{torch_version}+") and torch.version.cuda == cuda_tag
raise SystemExit(0 if ok else 1)
PY
then
  python -m pip uninstall -y torch triton >/dev/null 2>&1 || true
  python -m pip install --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION"
fi

python - <<'PY'
import torch

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY
echo

train_count=0
val_count=0
if [[ -d "$DATA_PATH" ]]; then
  train_count="$(find "$DATA_PATH" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l | tr -d ' ')"
  val_count="$(find "$DATA_PATH" -maxdepth 1 -name 'fineweb_val_*.bin' | wc -l | tr -d ' ')"
fi

if [[ ! -f "$TOKENIZER_PATH" || "$train_count" -lt "$TRAIN_SHARDS" || "$val_count" -eq 0 ]]; then
  echo "== downloading cached FineWeb data =="
  python data/cached_challenge_fineweb.py --variant "$VARIANT" --train-shards "$TRAIN_SHARDS"
else
  echo "== reusing existing cached FineWeb data =="
  echo "train_shards_present=$train_count"
  echo "val_shards_present=$val_count"
fi
echo

if [[ "$INSTALL_FA3" == "1" ]]; then
  if python - <<'PY' >/dev/null 2>&1
import flash_attn_interface
PY
  then
    echo "flash_attn_interface already importable"
  else
    echo "== installing FlashAttention-3 Hopper build =="
    if [[ ! -d "$FA3_DIR/.git" ]]; then
      git clone https://github.com/Dao-AILab/flash-attention "$FA3_DIR"
    fi
    git -C "$FA3_DIR" submodule update --init --recursive
    pushd "$FA3_DIR/hopper" >/dev/null
    python setup.py clean --all || true
    MAX_JOBS="$MAX_JOBS" python setup.py install 2>&1 | tee "$FA3_LOG"
    popd >/dev/null
    python - <<'PY'
import flash_attn_interface

print("flash_attn_interface OK")
PY
  fi
else
  echo "== skipping FlashAttention-3 install (INSTALL_FA3=$INSTALL_FA3) =="
fi
echo

cat <<EOF
Bootstrap complete.

Single-seed run:
  cd $REPO_DIR
  source $VENV_DIR/bin/activate
  ATTN_BACKEND=fa3 RUN_ID=hybrid_submission_seed1337_fa3 SEED=1337 \\
  bash innovations/hybrid_ttt/scripts/run_8xh100_hybrid_submission.sh

If FA3 is unavailable or you want the fallback:
  cd $REPO_DIR
  source $VENV_DIR/bin/activate
  ATTN_BACKEND=sdpa RUN_ID=hybrid_submission_seed1337_sdpa SEED=1337 \\
  bash innovations/hybrid_ttt/scripts/run_8xh100_hybrid_submission.sh
EOF
