Experimental hybrid TTT work lives here so the upstream repo code stays clean.

Files:
- `portable_train_gpt.py`: root CUDA baseline with eval-only hypothesis harness.
- `portable_train_gpt_mlx.py`: MLX baseline with local validation cap support.
- `eval_ttt_mlx.py`: MLX evaluator for adapter-only, persistent-model, and hybrid TTT experiments.
- `leader_2026_03_23_train_gpt.py`: copied March 23 leader architecture with local portability and hybrid-TTT experiments.
- `scripts/`: one-command launchers for 5090 hypothesis and leader-architecture runs.

Intent:
- leave `train_gpt.py`, `train_gpt_mlx.py`, and `records/...` identical to upstream
- keep all experimental code in one isolated place
