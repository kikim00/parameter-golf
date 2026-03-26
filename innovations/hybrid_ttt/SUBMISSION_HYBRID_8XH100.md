Hybrid 8xH100 submission path

This is the current 8xH100 launch path for the document-local adapter submission in the innovation tree.

Important behavior:
- Training uses the leader-style 11-layer recipe in [leader_2026_03_23_train_gpt.py](/Users/kdk/parameter-golf/innovations/hybrid_ttt/leader_2026_03_23_train_gpt.py).
- Final evaluation enables the document-local adapter path by setting `TTT_DOC_ADAPTER_RANK>0`.
- On `world_size>1`, document-local evaluation uses rank-local document streams and reduces `loss_sum`, `token_count`, and `byte_count` globally at the end.
- If `TTT_EPOCHS>0`, the base-model component of doc-local TTT is therefore rank-local persistent state during evaluation, not a fully synchronized global stream.

Single run:

```bash
source .venv/bin/activate
bash innovations/hybrid_ttt/scripts/run_8xh100_hybrid_submission.sh
```

Multi-seed record check:

```bash
source .venv/bin/activate
SEEDS="1337 1338 1339" bash innovations/hybrid_ttt/scripts/run_8xh100_hybrid_seeds.sh
```

Useful overrides:

```bash
RUN_ID=my_hybrid_run TTT_DOC_ADAPTER_RANK=16 TTT_DOC_ADAPTER_EPOCHS=5 TTT_EPOCHS=1 \
bash innovations/hybrid_ttt/scripts/run_8xh100_hybrid_submission.sh
```

Each run writes evidence under `innovations/hybrid_ttt/submission_runs/$RUN_ID`:
- `run_meta.txt`
- `git_status.txt`
- `git_diff_vs_openai_main.txt`
- `nvidia_smi*.txt`
- `python_torch_env.txt`
- `process_time.txt`
- `stdout.log`
- `train_log.txt` if generated
- `artifact_sha256.txt`
- `artifact_sizes.txt`
- `summary.txt`

Before treating this as a final record submission, verify:
- training finished within `600s`
- eval finished within the separate `600s` budget
- artifact size is under `16MB`
- repeated seeds are statistically convincing
- the documented distributed doc-local evaluation behavior is acceptable for the submission
