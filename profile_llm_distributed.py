#!/usr/bin/env python3
"""Minimal multi-GPU causal LM on Wikitext-2: timed training + inference (DDP or FSDP).

  torchrun --nproc_per_node=4 profile_llm_distributed.py --strategy fsdp --bf16
  torchrun --nproc_per_node=4 profile_llm_distributed.py --strategy ddp --bf16 \\
      --metrics-out run.jsonl

  Metrics file: one JSON object per line (rank 0). Plot: ``pandas.read_json(path, lines=True)``.

  W&B (rank 0): ``--wandb-project myproj`` (optional ``--wandb-entity``, ``--wandb-run-name``); ``wandb login`` or ``WANDB_API_KEY``.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import os
import time
from datetime import datetime, timezone
from contextlib import nullcontext
from typing import Any, Iterator

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer


def _dist_info() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    return True, rank, world, local


def _rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _init_dist(backend: str, local_rank: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return t
    out = t.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def _all_reduce_max(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return t
    out = t.clone()
    dist.all_reduce(out, op=dist.ReduceOp.MAX)
    return out


def _amp_ctx(dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _prepare_wikitext2(tokenizer: Any, seq_len: int) -> tuple[Any, Any]:
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tok(batch: dict[str, list]) -> dict[str, Any]:
        o = tokenizer(batch["text"], add_special_tokens=False, padding=False, truncation=False)
        return {"input_ids": o["input_ids"], "attention_mask": o["attention_mask"]}

    tt = raw["train"].map(
        tok, batched=True, remove_columns=raw["train"].column_names, desc="tokenize train"
    )
    tv = raw["validation"].map(
        tok, batched=True, remove_columns=raw["validation"].column_names, desc="tokenize val"
    )

    def group_texts(examples: dict[str, list]) -> dict[str, list]:
        ids = list(itertools.chain.from_iterable(examples["input_ids"]))
        attn = list(itertools.chain.from_iterable(examples["attention_mask"]))
        total = (len(ids) // seq_len) * seq_len
        if total == 0:
            return {"input_ids": [], "attention_mask": []}
        return {
            "input_ids": [ids[i : i + seq_len] for i in range(0, total, seq_len)],
            "attention_mask": [attn[i : i + seq_len] for i in range(0, total, seq_len)],
        }

    lm_train = tt.map(group_texts, batched=True, desc="group train")
    lm_val = tv.map(group_texts, batched=True, desc="group val")
    lm_train = lm_train.filter(lambda x: len(x["input_ids"]) > 0)
    lm_val = lm_val.filter(lambda x: len(x["input_ids"]) > 0)
    if len(lm_train) == 0:
        raise RuntimeError("No training blocks; check seq_len / tokenizer.")

    cols = ["input_ids", "attention_mask"]
    lm_train.set_format(type="torch", columns=cols)
    lm_val.set_format(type="torch", columns=cols)
    return lm_train, lm_val


def _collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([s["input_ids"] for s in samples])
    attention_mask = torch.stack([s["attention_mask"] for s in samples])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}


def _cycle(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    for b in itertools.cycle(loader):
        yield b


def _to_dev(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_train(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    device: torch.device,
    it: Iterator[dict[str, torch.Tensor]],
    *,
    steps: int,
    warmup: int,
    amp_dtype: torch.dtype | None,
    grad_clip: float | None,
) -> dict[str, float]:
    model.train()
    tokens = 0
    for _ in range(warmup):
        b = _to_dev(next(it), device)
        with _amp_ctx(amp_dtype):
            out = model(**b)
            loss = out.loss if hasattr(out, "loss") and out.loss is not None else out[0]
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
    _barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(steps):
        b = _to_dev(next(it), device)
        bs, seql = b["input_ids"].shape
        with _amp_ctx(amp_dtype):
            out = model(**b)
            loss = out.loss if hasattr(out, "loss") and out.loss is not None else out[0]
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)
        tokens += bs * seql
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _barrier()
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else 0.0
    tok_t = torch.tensor([float(tokens)], device=device)
    tok_t = _all_reduce_sum(tok_t)
    el_t = torch.tensor([elapsed], device=device)
    el_t = _all_reduce_max(el_t)
    pk_t = torch.tensor([peak], device=device)
    pk_t = _all_reduce_max(pk_t)
    return {
        "train_elapsed_s": el_t.item(),
        "train_tokens_global": tok_t.item(),
        "train_tokens_per_s": tok_t.item() / max(el_t.item(), 1e-9),
        "train_peak_memory_mib": pk_t.item(),
    }


@torch.no_grad()
def run_eval(
    model: nn.Module,
    device: torch.device,
    it: Iterator[dict[str, torch.Tensor]],
    *,
    steps: int,
    warmup: int,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    for _ in range(warmup):
        b = _to_dev(next(it), device)
        with _amp_ctx(amp_dtype):
            model(**b)
    _barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tokens = 0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(steps):
        b = _to_dev(next(it), device)
        bs, seql = b["input_ids"].shape
        with _amp_ctx(amp_dtype):
            model(**b)
        tokens += bs * seql
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _barrier()
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else 0.0
    tok_t = torch.tensor([float(tokens)], device=device)
    tok_t = _all_reduce_sum(tok_t)
    el_t = torch.tensor([elapsed], device=device)
    el_t = _all_reduce_max(el_t)
    pk_t = torch.tensor([peak], device=device)
    pk_t = _all_reduce_max(pk_t)
    return {
        "infer_elapsed_s": el_t.item(),
        "infer_tokens_global": tok_t.item(),
        "infer_tokens_per_s": tok_t.item() / max(el_t.item(), 1e-9),
        "infer_peak_memory_mib": pk_t.item(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wikitext-2 causal LM train/eval: DDP or FSDP")
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--strategy", choices=("ddp", "fsdp"), default="fsdp")
    p.add_argument("--mode", choices=("train", "infer", "both"), default="both")
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--dataloader-workers", type=int, default=2)
    p.add_argument("--backend", type=str, default="nccl")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--fsdp-sharding", default="full", choices=("full", "hybrid", "no_shard"))
    p.add_argument("--fsdp-min-num-params", type=int, default=10_000_000)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--metrics-out",
        type=str,
        default="profile_metrics.jsonl",
        help="Append one JSON line per run (rank 0). Empty string disables. "
        "Use with pandas.read_json(path, lines=True) for plots.",
    )
    p.add_argument("--wandb-project", type=str, default="", help="Weights & Biases project (empty = disabled).")
    p.add_argument("--wandb-entity", type=str, default="", help="W&B entity/team (optional).")
    p.add_argument("--wandb-run-name", type=str, default="", help="W&B run name (optional).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = _dist_info()
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if distributed:
        _init_dist(args.backend, local_rank)

    load_dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else None
    amp_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else None
    if amp_dtype is None and args.fp16 and torch.cuda.is_available():
        amp_dtype = torch.float16

    if _rank0():
        print(
            f"[config] model={args.model} strategy={args.strategy} mode={args.mode} "
            f"world={world_size} batch={args.per_device_batch} seq_len={args.seq_len} "
            f"steps={args.steps} warmup={args.warmup} bf16={args.bf16}",
            flush=True,
        )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    lm_train, lm_val = _prepare_wikitext2(tok, args.seq_len)
    train_sampler = DistributedSampler(lm_train, shuffle=True, drop_last=True, seed=args.seed)
    val_sampler = DistributedSampler(lm_val, shuffle=False, drop_last=True)
    bs = args.per_device_batch
    train_loader = DataLoader(
        lm_train,
        batch_size=bs,
        sampler=train_sampler,
        num_workers=args.dataloader_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        lm_val,
        batch_size=bs,
        sampler=val_sampler,
        num_workers=args.dataloader_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )

    kw: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if load_dtype is not None:
        kw["torch_dtype"] = load_dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if load_dtype is None:
        model = model.float()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.strategy == "ddp":
        model = model.to(device)
        if distributed and world_size > 1:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=False,
            )
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("FSDP requires CUDA/ROCm.")
        sharding = {"full": ShardingStrategy.FULL_SHARD, "hybrid": ShardingStrategy.HYBRID_SHARD, "no_shard": ShardingStrategy.NO_SHARD}[args.fsdp_sharding]
        mp = None
        if args.bf16:
            mp = MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
            )
        elif args.fp16:
            mp = MixedPrecision(
                param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16
            )
        model = FSDP(
            model,
            sharding_strategy=sharding,
            auto_wrap_policy=functools.partial(
                size_based_auto_wrap_policy, min_num_params=args.fsdp_min_num_params
            ),
            mixed_precision=mp,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)
    train_it = _cycle(train_loader)
    val_it = _cycle(val_loader)
    out: dict[str, float] = {}

    wandb_run = None
    try:
        if _rank0() and args.wandb_project:
            try:
                import wandb
            except ImportError as e:
                raise ImportError("Install wandb: pip install wandb") from e
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=args.wandb_run_name or None,
                config={
                    "model": args.model,
                    "strategy": args.strategy,
                    "mode": args.mode,
                    "world_size": world_size,
                    "per_device_batch": args.per_device_batch,
                    "seq_len": args.seq_len,
                    "steps": args.steps,
                    "warmup": args.warmup,
                    "bf16": args.bf16,
                    "fp16": args.fp16,
                    "backend": args.backend,
                    "lr": args.lr,
                    "fsdp_sharding": args.fsdp_sharding,
                    "train_dataset_len": len(lm_train),
                    "val_dataset_len": len(lm_val),
                },
            )

        if args.mode in ("train", "both"):
            train_sampler.set_epoch(args.seed)
            out.update(
                run_train(
                    model,
                    optim,
                    device,
                    train_it,
                    steps=args.steps,
                    warmup=args.warmup,
                    amp_dtype=amp_dtype,
                    grad_clip=args.grad_clip,
                )
            )
        if args.mode in ("infer", "both"):
            val_sampler.set_epoch(0)
            out.update(
                run_eval(
                    model,
                    device,
                    val_it,
                    steps=args.steps,
                    warmup=args.warmup,
                    amp_dtype=amp_dtype,
                )
            )

        _barrier()
        if _rank0():
            print("[results]", flush=True)
            for k in sorted(out):
                print(f"  {k}: {out[k]:.6g}", flush=True)

            row: dict[str, Any] = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "strategy": args.strategy,
                "model": args.model,
                "mode": args.mode,
                "world_size": world_size,
                "per_device_batch": args.per_device_batch,
                "seq_len": args.seq_len,
                "steps": args.steps,
                "warmup": args.warmup,
                "bf16": args.bf16,
                "fp16": args.fp16,
                "backend": args.backend,
                "lr": args.lr,
                "fsdp_sharding": args.fsdp_sharding,
                "train_dataset_len": len(lm_train),
                "val_dataset_len": len(lm_val),
            }
            row.update({k: float(v) for k, v in out.items()})

            if args.metrics_out:
                path = os.path.abspath(args.metrics_out)
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(path, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(row, sort_keys=True) + "\n")
                print(f"[metrics] appended -> {path}", flush=True)

            if wandb_run is not None:
                import wandb

                wandb.log({k: float(v) for k, v in out.items()})
    finally:
        if _rank0() and wandb_run is not None:
            import wandb

            wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
