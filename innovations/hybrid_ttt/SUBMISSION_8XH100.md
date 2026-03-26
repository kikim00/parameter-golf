**Goal**
Run a legal-ready 8xH100 submission attempt with enough captured evidence to justify:
- training stayed within `600` seconds wallclock
- evaluation stayed within the separate `600` second limit
- hardware was `8xH100`
- artifact size and exact score are recorded

**What Is Legal-Ready Now**
- [leader_2026_03_23_train_gpt.py](/Users/kdk/parameter-golf/innovations/hybrid_ttt/leader_2026_03_23_train_gpt.py)
  This is the leader-style architecture and persistent legal score-first TTT path.
- [run_8xh100_submission.sh](/Users/kdk/parameter-golf/innovations/hybrid_ttt/scripts/run_8xh100_submission.sh)
  This captures run metadata, GPU information, process wallclock, artifact hashes, and the key log lines.

**What Is Still Experimental**
- The document-local side-adapter hybrid path remains single-GPU-only in this innovation tree.
- So the `8xH100` recipe here is the legal-ready persistent full-model TTT path, not the hybrid adapter path.

**How To Run**
From the repo root:

```bash
source .venv/bin/activate
bash innovations/hybrid_ttt/scripts/run_8xh100_submission.sh
```

Optional overrides:

```bash
RUN_ID=my_submission \
SEED=1337 \
ATTN_BACKEND=auto \
NPROC_PER_NODE=8 \
bash innovations/hybrid_ttt/scripts/run_8xh100_submission.sh
```

**Artifacts To Keep**
The wrapper writes a directory under:
- `innovations/hybrid_ttt/submission_runs/$RUN_ID`

Important files:
- `run_meta.txt`
- `git_status.txt`
- `git_diff_vs_openai_main.txt`
- `nvidia_smi.txt`
- `nvidia_smi_topo.txt`
- `python_torch_env.txt`
- `process_time.txt`
- `stdout.log`
- `train_log.txt`
- `artifact_sha256.txt`
- `artifact_sizes.txt`
- `summary.txt`

**What To Quote In A Submission**
- `world_size:8`
- `train_batch_tokens`, `train_seq_len`, `iterations`, `MAX_WALLCLOCK_SECONDS`
- final `step:... val_loss ... val_bpb ... train_time`
- `Total submission size int6+lzma`
- `final_int6_roundtrip_exact`
- `final_int6_sliding_window_exact` or `legal_ttt_exact`
- `eval_time` lines for the final eval procedure

**How To Compare Against Upstream**
```bash
git diff --name-only openai/main...HEAD
git log --oneline openai/main..HEAD
```

Only the `innovations/hybrid_ttt/` tree should differ from upstream.
