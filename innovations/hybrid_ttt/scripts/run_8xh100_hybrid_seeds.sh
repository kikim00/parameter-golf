#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_8xh100_hybrid_submission.sh"
SEEDS_STR="${SEEDS:-1337 1338 1339}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/submission_runs}"

mkdir -p "$EVIDENCE_ROOT"

for seed in $SEEDS_STR; do
  RUN_ID_SEED="${RUN_ID_PREFIX:-hybrid_seed}_${seed}"
  EVIDENCE_DIR_SEED="$EVIDENCE_ROOT/$RUN_ID_SEED"
  echo "=== running seed $seed (run_id=$RUN_ID_SEED) ==="
  RUN_ID="$RUN_ID_SEED" EVIDENCE_DIR="$EVIDENCE_DIR_SEED" SEED="$seed" bash "$RUN_SCRIPT"
done
