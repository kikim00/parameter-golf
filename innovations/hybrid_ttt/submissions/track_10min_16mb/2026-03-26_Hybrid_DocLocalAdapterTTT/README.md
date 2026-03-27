# Hybrid Doc-Local Adapter TTT

Submission-shaped package for the hybrid document-local adapter direction, kept outside `records/`.

This folder is the canonical submission path for the hybrid experiments:
- [train_gpt.py](/Users/kdk/parameter-golf/innovations/hybrid_ttt/submissions/track_10min_16mb/2026-03-26_Hybrid_DocLocalAdapterTTT/train_gpt.py)
- [submission.json](/Users/kdk/parameter-golf/innovations/hybrid_ttt/submissions/track_10min_16mb/2026-03-26_Hybrid_DocLocalAdapterTTT/submission.json)

## Status

Not finalized. The 8xH100 run has not been executed yet, so the metrics in `submission.json` are placeholders.

## Method

- Leader-style 11-layer architecture and training recipe
- Document-local score-first output adapter
- Optional persistent base-model TTT on top of the document-local adapter
- 8xH100 launch wrappers live in:
  - [run_8xh100_hybrid_submission.sh](/Users/kdk/parameter-golf/innovations/hybrid_ttt/scripts/run_8xh100_hybrid_submission.sh)
  - [run_8xh100_hybrid_seeds.sh](/Users/kdk/parameter-golf/innovations/hybrid_ttt/scripts/run_8xh100_hybrid_seeds.sh)

## Important distributed note

On `world_size > 1`, document-local evaluation shards documents across ranks for scoring and keeps the reset-per-document adapter local to each rank's current document. If `TTT_EPOCHS > 0`, the base-model TTT gradients are averaged across the active ranks after each scored document minibatch, so the persistent base-model state remains synchronized across GPUs.

## Direct run command

```bash
RUN_ID=hybrid_submission_v1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
MAX_WALLCLOCK_SECONDS=600 \
TTT_ENABLED=1 \
TTT_DOC_ADAPTER_RANK=16 \
TTT_DOC_ADAPTER_LR=0.5 \
TTT_DOC_ADAPTER_EPOCHS=5 \
TTT_DOC_ADAPTER_MAX_DOCS=0 \
torchrun --standalone --nproc_per_node=8 \
  innovations/hybrid_ttt/submissions/track_10min_16mb/2026-03-26_Hybrid_DocLocalAdapterTTT/train_gpt.py
```

For the full wrapped/evidence-capturing path, use:

```bash
bash innovations/hybrid_ttt/scripts/run_8xh100_hybrid_submission.sh
```
