#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import math
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from train_gpt_mlx import GPT, Hyperparameters, build_sentencepiece_luts, load_data_shard


def build_parser() -> argparse.ArgumentParser:
    defaults = Hyperparameters()
    parser = argparse.ArgumentParser(description="Evaluate a small validation slice with optional score-first adapter TTT.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .npz checkpoint produced by train_gpt_mlx.py")
    parser.add_argument("--data-path", default=defaults.data_path, help="Dataset directory containing fineweb_val_*.bin")
    parser.add_argument("--tokenizer-path", default=defaults.tokenizer_path, help="SentencePiece .model path")
    parser.add_argument("--vocab-size", type=int, default=defaults.vocab_size)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--model-dim", type=int, default=defaults.model_dim)
    parser.add_argument("--num-heads", type=int, default=defaults.num_heads)
    parser.add_argument("--num-kv-heads", type=int, default=defaults.num_kv_heads)
    parser.add_argument("--mlp-mult", type=int, default=defaults.mlp_mult)
    parser.add_argument("--tied-embed-init-std", type=float, default=defaults.tied_embed_init_std)
    parser.add_argument("--logit-softcap", type=float, default=defaults.logit_softcap)
    parser.add_argument("--logit-chunk-tokens", type=int, default=0)
    parser.add_argument("--rope-base", type=float, default=defaults.rope_base)
    parser.add_argument("--qk-gain-init", type=float, default=defaults.qk_gain_init)
    parser.add_argument("--seq-len", type=int, default=defaults.train_seq_len, help="Maximum evaluation context length")
    parser.add_argument("--eval-stride", type=int, default=128, help="Targets scored per score-first update chunk")
    parser.add_argument("--max-docs", type=int, default=200, help="Number of validation documents to benchmark")
    parser.add_argument("--adapter-rank", type=int, default=8, help="LoRA-style rank for the fast output adapter")
    parser.add_argument("--adapter-lr", type=float, default=0.02)
    parser.add_argument("--adapter-epochs", type=int, default=1)
    parser.add_argument("--adapter-init-std", type=float, default=0.01)
    parser.add_argument("--adapter-weight-decay", type=float, default=0.0)
    parser.add_argument("--adapter-grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--adapter-scope",
        choices=("reset_per_doc", "persist"),
        default="reset_per_doc",
        help="Whether fast weights reset at document boundaries or carry across documents.",
    )
    return parser


def apply_softcap(logits: mx.array, logit_softcap: float) -> mx.array:
    return logit_softcap * mx.tanh(logits / logit_softcap)


def load_checkpoint(path: Path) -> dict[str, mx.array]:
    loaded = mx.load(str(path))
    if hasattr(loaded, "items"):
        return {str(k): v for k, v in loaded.items()}
    raise TypeError(f"Unexpected checkpoint payload type: {type(loaded)}")


def load_validation_documents(pattern: str, bos_id: int, max_docs: int) -> list[np.ndarray]:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    docs: list[np.ndarray] = []
    current_parts: list[np.ndarray] = []
    current_open = False

    for file in files:
        tokens = load_data_shard(file)
        starts = np.flatnonzero(tokens == bos_id)
        cursor = 0

        for start in starts:
            if current_open:
                if start > cursor:
                    current_parts.append(tokens[cursor:start])
                doc = np.concatenate(current_parts, axis=0)
                if doc.size > 1:
                    docs.append(doc)
                    if len(docs) >= max_docs:
                        return docs
                current_parts = []
            elif start > 0:
                raise ValueError(f"Found tokens before BOS boundary in {file}")

            current_parts = [tokens[start : start + 1]]
            current_open = True
            cursor = start + 1

        if current_open and cursor < tokens.size:
            current_parts.append(tokens[cursor:])
        elif not current_open and tokens.size:
            raise ValueError(f"No BOS token found while streaming {file}")

    if current_open and current_parts and len(docs) < max_docs:
        doc = np.concatenate(current_parts, axis=0)
        if doc.size > 1:
            docs.append(doc)
    return docs[:max_docs]


def adapter_template(model_dim: int, vocab_size: int, rank: int, init_std: float) -> dict[str, mx.array]:
    if rank <= 0:
        raise ValueError("adapter rank must be positive")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((model_dim, rank), dtype=np.float32) * init_std
    b = np.zeros((rank, vocab_size), dtype=np.float32)
    return {"a": mx.array(a, dtype=mx.float32), "b": mx.array(b, dtype=mx.float32)}


def clone_adapter(template: dict[str, mx.array]) -> dict[str, mx.array]:
    return {k: mx.array(np.array(v), dtype=v.dtype) for k, v in template.items()}


def maybe_clip_grads(grads: dict[str, mx.array], max_norm: float) -> dict[str, mx.array]:
    if max_norm <= 0:
        return grads
    total_sq = 0.0
    for grad in grads.values():
        total_sq += float(np.sum(np.square(np.array(grad.astype(mx.float32))), dtype=np.float64))
    if total_sq <= 0.0:
        return grads
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads
    scale = max_norm / (total_norm + 1e-12)
    return {k: g * scale for k, g in grads.items()}


def selected_logits(
    hidden: mx.array,
    embed_weight: mx.array,
    adapter: dict[str, mx.array] | None,
    logit_softcap: float,
) -> mx.array:
    logits_proj = hidden @ embed_weight.astype(hidden.dtype).T
    if adapter is not None:
        logits_proj = logits_proj + (hidden @ adapter["a"].astype(hidden.dtype)) @ adapter["b"].astype(hidden.dtype)
    return apply_softcap(logits_proj, logit_softcap).astype(mx.float32)


def selected_loss(
    hidden: mx.array,
    targets: mx.array,
    embed_weight: mx.array,
    adapter: dict[str, mx.array] | None,
    logit_softcap: float,
) -> tuple[mx.array, mx.array]:
    logits = selected_logits(hidden, embed_weight, adapter, logit_softcap)
    losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    return losses.mean(), losses


def adapter_loss_fn(
    adapter: dict[str, mx.array],
    hidden: mx.array,
    targets: mx.array,
    embed_weight: mx.array,
    logit_softcap: float,
    weight_decay: float,
) -> mx.array:
    mean_loss, _ = selected_loss(hidden, targets, embed_weight, adapter, logit_softcap)
    if weight_decay > 0:
        reg = mx.array(0.0, dtype=mx.float32)
        for value in adapter.values():
            reg = reg + mx.mean(value.astype(mx.float32) * value.astype(mx.float32))
        mean_loss = mean_loss + weight_decay * reg
    return mean_loss


def train_adapter_on_chunk(
    adapter: dict[str, mx.array],
    hidden: mx.array,
    targets: mx.array,
    embed_weight: mx.array,
    logit_softcap: float,
    lr: float,
    epochs: int,
    weight_decay: float,
    grad_clip: float,
) -> dict[str, mx.array]:
    if epochs <= 0:
        return adapter

    loss_and_grad = mx.value_and_grad(adapter_loss_fn)
    for _ in range(epochs):
        loss, grads = loss_and_grad(adapter, hidden, targets, embed_weight, logit_softcap, weight_decay)
        grads = maybe_clip_grads(grads, grad_clip)
        adapter = {k: adapter[k] - lr * grads[k].astype(adapter[k].dtype) for k in adapter}
        mx.eval(loss, *adapter.values())
    return adapter


def token_bytes_for_chunk(
    prev_ids: np.ndarray,
    tgt_ids: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
) -> float:
    token_bytes = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
    token_bytes += (
        has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
    ).astype(np.int16, copy=False)
    return float(token_bytes.astype(np.float64).sum())


def evaluate_documents(
    *,
    model: GPT,
    docs: list[np.ndarray],
    seq_len: int,
    eval_stride: int,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    adapter_proto: dict[str, mx.array] | None,
    adapter_scope: str,
    adapter_lr: float,
    adapter_epochs: int,
    adapter_weight_decay: float,
    adapter_grad_clip: float,
) -> dict[str, float]:
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    doc_lengths: list[int] = []
    start = time.perf_counter()

    persistent_adapter = clone_adapter(adapter_proto) if adapter_proto is not None else None
    embed_weight = model.tok_emb.weight

    for doc in docs:
        n_targets = int(doc.size - 1)
        if n_targets <= 0:
            continue
        doc_lengths.append(n_targets)
        adapter = persistent_adapter if persistent_adapter is not None else None
        if adapter_proto is not None and adapter_scope == "reset_per_doc":
            adapter = clone_adapter(adapter_proto)

        for t_start in range(0, n_targets, eval_stride):
            t_end = min(t_start + eval_stride, n_targets)
            raw_end = t_end + 1
            raw_start = max(0, raw_end - (seq_len + 1))
            chunk = doc[raw_start:raw_end]

            x_np = chunk[:-1].astype(np.int32, copy=False)
            y_np = chunk[1:].astype(np.int32, copy=False)
            offset = t_start - raw_start
            if offset < 0:
                raise ValueError("negative score offset")

            x = mx.array(x_np[None, :], dtype=mx.int32)
            hidden = model(x).reshape(-1, model.tok_emb.weight.shape[1])
            hidden_sel = hidden[offset:]
            y_sel = mx.array(y_np[offset:], dtype=mx.int32)

            _, losses = selected_loss(hidden_sel, y_sel, embed_weight, adapter, model.logit_softcap)
            mx.eval(losses)
            total_loss_sum += float(mx.sum(losses).item())
            total_tokens += float(y_np[offset:].size)
            total_bytes += token_bytes_for_chunk(
                x_np[offset:],
                y_np[offset:],
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )

            if adapter is not None:
                adapter = train_adapter_on_chunk(
                    adapter,
                    hidden_sel,
                    y_sel,
                    embed_weight,
                    model.logit_softcap,
                    adapter_lr,
                    adapter_epochs,
                    adapter_weight_decay,
                    adapter_grad_clip,
                )
                if persistent_adapter is not None and adapter_scope == "persist":
                    persistent_adapter = adapter

    elapsed = time.perf_counter() - start
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return {
        "docs": float(len(doc_lengths)),
        "tokens": total_tokens,
        "avg_doc_tokens": float(sum(doc_lengths) / max(len(doc_lengths), 1)),
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "elapsed_s": elapsed,
    }


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError("tokenizer path must point to a SentencePiece .model file")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )

    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    )
    flat_state = load_checkpoint(checkpoint_path)
    model.update(tree_unflatten(list(flat_state.items())))

    val_pattern = str(Path(args.data_path) / "fineweb_val_*.bin")
    docs = load_validation_documents(val_pattern, int(sp.bos_id()), args.max_docs)
    if not docs:
        raise ValueError("No validation documents loaded")

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size)

    print(
        f"benchmark_docs:{len(docs)} checkpoint:{checkpoint_path.name} seq_len:{args.seq_len} "
        f"eval_stride:{args.eval_stride}"
    )
    baseline = evaluate_documents(
        model=model,
        docs=docs,
        seq_len=args.seq_len,
        eval_stride=args.eval_stride,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        adapter_proto=None,
        adapter_scope=args.adapter_scope,
        adapter_lr=args.adapter_lr,
        adapter_epochs=args.adapter_epochs,
        adapter_weight_decay=args.adapter_weight_decay,
        adapter_grad_clip=args.adapter_grad_clip,
    )
    print(
        f"baseline docs:{int(baseline['docs'])} tokens:{int(baseline['tokens'])} "
        f"avg_doc_tokens:{baseline['avg_doc_tokens']:.1f} val_loss:{baseline['val_loss']:.6f} "
        f"val_bpb:{baseline['val_bpb']:.6f} elapsed_s:{baseline['elapsed_s']:.2f}"
    )

    if args.adapter_rank > 0:
        proto = adapter_template(args.model_dim, args.vocab_size, args.adapter_rank, args.adapter_init_std)
        ttt = evaluate_documents(
            model=model,
            docs=docs,
            seq_len=args.seq_len,
            eval_stride=args.eval_stride,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            adapter_proto=proto,
            adapter_scope=args.adapter_scope,
            adapter_lr=args.adapter_lr,
            adapter_epochs=args.adapter_epochs,
            adapter_weight_decay=args.adapter_weight_decay,
            adapter_grad_clip=args.adapter_grad_clip,
        )
        print(
            f"adapter_ttt scope:{args.adapter_scope} rank:{args.adapter_rank} lr:{args.adapter_lr} "
            f"epochs:{args.adapter_epochs} val_loss:{ttt['val_loss']:.6f} "
            f"val_bpb:{ttt['val_bpb']:.6f} delta_bpb:{ttt['val_bpb'] - baseline['val_bpb']:+.6f} "
            f"elapsed_s:{ttt['elapsed_s']:.2f}"
        )


if __name__ == "__main__":
    main()
