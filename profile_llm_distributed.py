#!/usr/bin/env python3
"""
Multi-GPU LLM continued pretraining / inference profiler: DDP vs FSDP on Wikitext-2.

Uses real token blocks from wikitext-2-raw-v1 (next-token loss for "continued pretraining").

VRAM saturation:
  - Raise ``--seq-len`` and use ``--auto-max-batch`` to binary-search the largest
    micro-batch (backward probes, then a full optimizer step at the chosen size to match real training).
  - Prefer ``--bf16`` so activations are smaller and you can often pick a *larger*
    micro-batch — that uses more memory and is usually closer to "saturating" the GPU.
  - Turn **off** gradient checkpointing (default here) when you want peak activation
    memory rather than trading compute for headroom.
  - After tuning batch/seq, check ``train_peak_memory_mib`` / ``infer_peak_memory_mib``:
    if you are far below physical memory, increase batch or sequence until near OOM.

Examples:
  torchrun --nproc_per_node=4 profile_llm_distributed.py --strategy fsdp --auto-max-batch
  torchrun --nproc_per_node=4 profile_llm_distributed.py --strategy ddp --bf16 --seq-len 1024 \\
      --per-device-batch 8 --mode both
  # Weights & Biases (rank 0 only; set WANDB_API_KEY or ``wandb login``)
  torchrun --nproc_per_node=4 profile_llm_distributed.py --strategy fsdp --wandb-project llm-profile \\
      --wandb-run-name fsdp-gpt2 --wandb-tags amd,rocm
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
    if not dist.is_available():
        return False, 0, 1, 0
    if "RANK" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    return True, rank, world, local


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


def _all_reduce_min(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return t
    out = t.clone()
    dist.all_reduce(out, op=dist.ReduceOp.MIN)
    return out


def _build_fsdp_wrap_policy(min_num_params: int):
    return functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=min_num_params,
    )


def _get_amp_ctx(dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _load_causal_lm(
    model_name: str,
    trust_remote_code: bool,
    dtype: torch.dtype | None,
) -> nn.Module:
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if dtype is None:
        model = model.float()
    return model


def _prepare_wikitext2(
    tokenizer: Any,
    seq_len: int,
    rank: int,
) -> tuple[Any, Any]:
    """HF datasets: blocked causal LM chunks from Wikitext-2 raw."""
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tok(batch: dict[str, list]) -> dict[str, Any]:
        o = tokenizer(batch["text"], add_special_tokens=False, padding=False, truncation=False)
        return {"input_ids": o["input_ids"], "attention_mask": o["attention_mask"]}

    tokenized = raw.map(
        tok,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="tokenize wikitext-2",
    )

    block_size = seq_len

    def group_texts(examples: dict[str, list]) -> dict[str, list]:
        ids = list(itertools.chain.from_iterable(examples["input_ids"]))
        attn = list(itertools.chain.from_iterable(examples["attention_mask"]))
        total = (len(ids) // block_size) * block_size
        if total == 0:
            return {"input_ids": [], "attention_mask": []}
        out_ids = [ids[i : i + block_size] for i in range(0, total, block_size)]
        out_attn = [attn[i : i + block_size] for i in range(0, total, block_size)]
        return {"input_ids": out_ids, "attention_mask": out_attn}

    lm_train = tokenized["train"].map(
        group_texts,
        batched=True,
        desc="group train",
    )
    lm_val = tokenized["validation"].map(
        group_texts,
        batched=True,
        desc="group val",
    )

    lm_train = lm_train.filter(lambda x: len(x["input_ids"]) > 0)
    lm_val = lm_val.filter(lambda x: len(x["input_ids"]) > 0)

    if rank == 0 and len(lm_train) == 0:
        raise RuntimeError("Wikitext-2 produced zero training blocks; check tokenizer/seq_len.")

    columns = ["input_ids", "attention_mask"]
    lm_train.set_format(type="torch", columns=columns)
    lm_val.set_format(type="torch", columns=columns)
    return lm_train, lm_val


def _collate_lm_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([s["input_ids"] for s in samples], dim=0)
    attention_mask = torch.stack([s["attention_mask"] for s in samples], dim=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }


def _iter_forever(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    for batch in itertools.cycle(loader):
        yield batch


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _reset_peak_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mib(device: torch.device) -> float:
    if not torch.cuda.is_available() or device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def _build_cpu_token_window(
    loader: DataLoader,
    max_rows: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First max_rows rows of tokenized data on CPU for OOM-safe batch search."""
    rows_ids: list[torch.Tensor] = []
    rows_mask: list[torch.Tensor] = []
    n = 0
    for batch in loader:
        b = batch["input_ids"].shape[0]
        take = min(b, max_rows - n)
        rows_ids.append(batch["input_ids"][:take].contiguous())
        rows_mask.append(batch["attention_mask"][:take].contiguous())
        n += take
        if n >= max_rows:
            break
    if n == 0:
        raise RuntimeError("No samples in loader for batch search buffer.")
    ids = torch.cat(rows_ids, dim=0)
    mask = torch.cat(rows_mask, dim=0)
    assert ids.shape[1] == seq_len, (ids.shape, seq_len)
    return ids, mask


def _batch_from_cpu_window(
    ids: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sl = slice(0, batch_size)
    input_ids = ids[sl].to(device, non_blocking=True)
    attention_mask = mask[sl].to(device, non_blocking=True)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }


def find_max_batch_train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    cpu_ids: torch.Tensor,
    cpu_mask: torch.Tensor,
    cap: int,
    amp_dtype: torch.dtype | None,
    grad_clip: float | None,
) -> int:
    """Largest batch size in [1, cap] that survives a full training step on this process."""
    if not torch.cuda.is_available():
        return min(cap, cpu_ids.shape[0])

    def try_train(bs: int, *, do_optimizer_step: bool) -> bool:
        optimizer.zero_grad(set_to_none=True)
        try:
            batch = _batch_from_cpu_window(cpu_ids, cpu_mask, bs, device)
            with _get_amp_ctx(amp_dtype):
                out = model(**batch)
                loss = out.loss if hasattr(out, "loss") and out.loss is not None else out[0]
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if do_optimizer_step:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            return True
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda out of memory" in msg:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                return False
            raise

    cap = min(cap, cpu_ids.shape[0])
    if not try_train(1, do_optimizer_step=True):
        raise RuntimeError("Batch size 1 OOM during --auto-max-batch; lower --seq-len or use smaller model / bf16.")

    lo, hi = 1, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if try_train(mid, do_optimizer_step=False):
            lo = mid
        else:
            hi = mid - 1

    while lo >= 1 and not try_train(lo, do_optimizer_step=True):
        lo -= 1
    if lo < 1:
        raise RuntimeError("No batch size survived full train step after search.")
    return lo


def find_max_batch_infer(
    model: nn.Module,
    device: torch.device,
    *,
    cpu_ids: torch.Tensor,
    cpu_mask: torch.Tensor,
    cap: int,
    amp_dtype: torch.dtype | None,
) -> int:
    if not torch.cuda.is_available():
        return min(cap, cpu_ids.shape[0])
    model.eval()

    def ok(bs: int) -> bool:
        try:
            with torch.no_grad():
                batch = _batch_from_cpu_window(cpu_ids, cpu_mask, bs, device)
                with _get_amp_ctx(amp_dtype):
                    model(**batch)
            torch.cuda.synchronize()
            return True
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda out of memory" in msg:
                torch.cuda.empty_cache()
                return False
            raise

    cap = min(cap, cpu_ids.shape[0])
    if not ok(1):
        raise RuntimeError("Batch size 1 OOM during inference batch search; lower --seq-len or model size.")

    lo, hi = 1, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def profile_train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_iter: Iterator[dict[str, torch.Tensor]],
    *,
    steps: int,
    warmup: int,
    amp_dtype: torch.dtype | None,
    grad_clip: float | None,
) -> dict[str, float]:
    model.train()
    total_tokens = 0
    seq_len = 0

    for _ in range(warmup):
        batch = _move_batch(next(batch_iter), device)
        seq_len = int(batch["input_ids"].shape[1])
        with _get_amp_ctx(amp_dtype):
            out = model(**batch)
            loss = out.loss if hasattr(out, "loss") and out.loss is not None else out[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    _barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    _reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        batch = _move_batch(next(batch_iter), device)
        seq_len = int(batch["input_ids"].shape[1])
        bsz = int(batch["input_ids"].shape[0])
        with _get_amp_ctx(amp_dtype):
            out = model(**batch)
            loss = out.loss if hasattr(out, "loss") and out.loss is not None else out[0]
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_tokens += bsz * seq_len

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _barrier()
    elapsed_local = time.perf_counter() - t0
    peak_local = _peak_memory_mib(device)

    tok = torch.tensor([float(total_tokens)], device=device)
    tok = _all_reduce_sum(tok)
    el = torch.tensor([elapsed_local], device=device)
    el = _all_reduce_max(el)
    peak_t = torch.tensor([peak_local], device=device)
    peak_t = _all_reduce_max(peak_t)

    tokens, elapsed_wall = tok[0].item(), el[0].item()
    peak_max = peak_t[0].item()
    return {
        "train_elapsed_s": elapsed_wall,
        "train_tokens_global": tokens,
        "train_tokens_per_s": tokens / max(elapsed_wall, 1e-9),
        "train_peak_memory_mib": peak_max,
    }


@torch.no_grad()
def profile_infer(
    model: nn.Module,
    device: torch.device,
    batch_iter: Iterator[dict[str, torch.Tensor]],
    *,
    steps: int,
    warmup: int,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    seq_len = 0

    for _ in range(warmup):
        batch = _move_batch(next(batch_iter), device)
        seq_len = int(batch["input_ids"].shape[1])
        with _get_amp_ctx(amp_dtype):
            model(**batch)

    _barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_tokens = 0
    _reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        batch = _move_batch(next(batch_iter), device)
        seq_len = int(batch["input_ids"].shape[1])
        bsz = int(batch["input_ids"].shape[0])
        with _get_amp_ctx(amp_dtype):
            model(**batch)
        total_tokens += bsz * seq_len

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _barrier()
    elapsed_local = time.perf_counter() - t0
    peak_local = _peak_memory_mib(device)

    tok = torch.tensor([float(total_tokens)], device=device)
    tok = _all_reduce_sum(tok)
    el = torch.tensor([elapsed_local], device=device)
    el = _all_reduce_max(el)
    peak_t = torch.tensor([peak_local], device=device)
    peak_t = _all_reduce_max(peak_t)

    tokens, elapsed_wall = tok[0].item(), el[0].item()
    peak_max = peak_t[0].item()
    return {
        "infer_elapsed_s": elapsed_wall,
        "infer_tokens_global": tokens,
        "infer_tokens_per_s": tokens / max(elapsed_wall, 1e-9),
        "infer_peak_memory_mib": peak_max,
    }


def _wandb_tags_list(raw: str) -> list[str] | None:
    if not raw.strip():
        return None
    return [t.strip() for t in raw.split(",") if t.strip()]


def _build_wandb_config(
    args: argparse.Namespace,
    *,
    world_size: int,
    train_bs: int,
    infer_bs: int,
    train_dataset_len: int,
    val_dataset_len: int,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "model": args.model,
        "strategy": args.strategy,
        "mode": args.mode,
        "world_size": world_size,
        "seq_len": args.seq_len,
        "per_device_batch_cli": args.per_device_batch,
        "per_device_batch_train": train_bs,
        "per_device_batch_infer": infer_bs,
        "auto_max_batch": args.auto_max_batch,
        "batch_search_cap": args.batch_search_cap,
        "steps": args.steps,
        "warmup": args.warmup,
        "lr": args.lr,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "backend": args.backend,
        "gradient_checkpointing": args.gradient_checkpointing,
        "fsdp_sharding": args.fsdp_sharding,
        "fsdp_min_num_params": args.fsdp_min_num_params,
        "grad_clip": args.grad_clip,
        "dataloader_workers": args.dataloader_workers,
        "seed": args.seed,
        "data": "wikitext-2-raw-v1",
        "train_dataset_len": train_dataset_len,
        "val_dataset_len": val_dataset_len,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "hip_version": getattr(torch.version, "hip", None),
        "device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
    }
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile LLM on Wikitext-2: DDP vs FSDP")
    p.add_argument("--model", type=str, default="gpt2", help="HF model id")
    p.add_argument("--strategy", type=str, choices=("ddp", "fsdp"), default="fsdp")
    p.add_argument("--mode", type=str, choices=("train", "infer", "both"), default="both")
    p.add_argument("--per-device-batch", type=int, default=2, help="Micro-batch per GPU (ignored if --auto-max-batch)")
    p.add_argument("--auto-max-batch", action="store_true", help="Binary-search largest micro-batch that fits")
    p.add_argument(
        "--batch-search-cap",
        type=int,
        default=512,
        help="Upper bound for --auto-max-batch (per device)",
    )
    p.add_argument("--seq-len", type=int, default=512, help="Blocked context length for LM")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--dataloader-workers", type=int, default=2)
    p.add_argument("--backend", type=str, default="nccl")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--fsdp-sharding", type=str, default="full", choices=("full", "hybrid", "no_shard"))
    p.add_argument("--fsdp-min-num-params", type=int, default=10_000_000)
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Lowers activation memory (usually *reduces* peak VRAM — disable to push memory higher).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--wandb-project",
        type=str,
        default="",
        help="Weights & Biases project name; leave empty to disable logging.",
    )
    p.add_argument("--wandb-entity", type=str, default="", help="W&B team/entity (optional).")
    p.add_argument("--wandb-run-name", type=str, default="", help="W&B run display name (optional).")
    p.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="Comma-separated W&B tags (e.g. nvidia,a100 or amd,mi300).",
    )
    p.add_argument(
        "--plot-jsonl",
        type=str,
        default="",
        help="Append one JSON object per run (rank 0) for plotting, e.g. pandas.read_json(path, lines=True).",
    )
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

    amp_dtype: torch.dtype | None = None
    load_dtype: torch.dtype | None = None
    if args.bf16 and torch.cuda.is_available():
        load_dtype = torch.bfloat16
        amp_dtype = torch.bfloat16
    elif args.fp16 and torch.cuda.is_available():
        amp_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    lm_train, lm_val = _prepare_wikitext2(tokenizer, args.seq_len, rank)

    train_sampler = DistributedSampler(
        lm_train,
        shuffle=True,
        drop_last=True,
        seed=args.seed,
    )
    val_sampler = DistributedSampler(
        lm_val,
        shuffle=False,
        drop_last=True,
    )

    train_loader = DataLoader(
        lm_train,
        batch_size=args.per_device_batch,
        sampler=train_sampler,
        num_workers=args.dataloader_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_lm_batch,
    )
    val_loader = DataLoader(
        lm_val,
        batch_size=args.per_device_batch,
        sampler=val_sampler,
        num_workers=args.dataloader_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_lm_batch,
    )

    if rank == 0:
        print(
            f"[config] data=wikitext-2-raw-v1 model={args.model} strategy={args.strategy} mode={args.mode} "
            f"world_size={world_size} seq_len={args.seq_len} per_device_batch={args.per_device_batch} "
            f"steps={args.steps} warmup={args.warmup} bf16={args.bf16} fp16={args.fp16} "
            f"auto_max_batch={args.auto_max_batch}",
            flush=True,
        )

    model = _load_causal_lm(args.model, args.trust_remote_code, load_dtype)
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
    elif args.strategy == "fsdp":
        if not torch.cuda.is_available():
            raise RuntimeError("FSDP path expects CUDA/ROCm GPUs.")
        sharding = {
            "full": ShardingStrategy.FULL_SHARD,
            "hybrid": ShardingStrategy.HYBRID_SHARD,
            "no_shard": ShardingStrategy.NO_SHARD,
        }[args.fsdp_sharding]
        mp = None
        if args.bf16:
            mp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif args.fp16:
            mp = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        model = FSDP(
            model,
            sharding_strategy=sharding,
            auto_wrap_policy=_build_fsdp_wrap_policy(args.fsdp_min_num_params),
            mixed_precision=mp,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )
    else:
        raise ValueError(args.strategy)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Optional: auto max batch (train and/or infer caps), then rebuild loaders with chosen batch.
    train_bs = args.per_device_batch
    infer_bs = args.per_device_batch
    if args.auto_max_batch:
        search_train_sampler = DistributedSampler(
            lm_train,
            shuffle=True,
            drop_last=False,
            seed=args.seed,
        )
        search_loader = DataLoader(
            lm_train,
            batch_size=min(8, max(1, len(lm_train) // max(world_size, 1))),
            sampler=search_train_sampler,
            num_workers=args.dataloader_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate_lm_batch,
        )
        search_train_sampler.set_epoch(args.seed)
        cpu_ids, cpu_mask = _build_cpu_token_window(search_loader, args.batch_search_cap, args.seq_len)

        if args.mode in ("train", "both"):
            _barrier()
            train_bs_local = find_max_batch_train(
                model,
                optim,
                device,
                cpu_ids=cpu_ids,
                cpu_mask=cpu_mask,
                cap=args.batch_search_cap,
                amp_dtype=amp_dtype,
                grad_clip=args.grad_clip,
            )
            bs_tensor = torch.tensor([float(train_bs_local)], device=device)
            train_bs = int(_all_reduce_min(bs_tensor)[0].item())
            if rank == 0:
                print(f"[auto-max-batch] train per_device_batch={train_bs}", flush=True)

        if args.mode in ("infer", "both"):
            search_val_sampler = DistributedSampler(lm_val, shuffle=False, drop_last=False)
            val_search = DataLoader(
                lm_val,
                batch_size=min(8, max(1, len(lm_val) // max(world_size, 1))),
                sampler=search_val_sampler,
                num_workers=args.dataloader_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=_collate_lm_batch,
            )
            cpu_ids_v, cpu_mask_v = _build_cpu_token_window(val_search, args.batch_search_cap, args.seq_len)
            _barrier()
            infer_bs_local = find_max_batch_infer(
                model,
                device,
                cpu_ids=cpu_ids_v,
                cpu_mask=cpu_mask_v,
                cap=args.batch_search_cap,
                amp_dtype=amp_dtype,
            )
            bs_tensor = torch.tensor([float(infer_bs_local)], device=device)
            infer_bs = int(_all_reduce_min(bs_tensor)[0].item())
            if rank == 0:
                print(f"[auto-max-batch] infer per_device_batch={infer_bs}", flush=True)

        train_loader = DataLoader(
            lm_train,
            batch_size=train_bs,
            sampler=train_sampler,
            num_workers=args.dataloader_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate_lm_batch,
        )
        val_loader = DataLoader(
            lm_val,
            batch_size=infer_bs,
            sampler=val_sampler,
            num_workers=args.dataloader_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate_lm_batch,
        )
        if rank == 0:
            print(
                f"[config] effective per_device_batch train={train_bs} infer={infer_bs}",
                flush=True,
            )

    train_iter = _iter_forever(train_loader)
    val_iter = _iter_forever(val_loader)

    wandb_run = None
    if rank == 0 and args.wandb_project:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "Weights & Biases is not installed. Run: pip install wandb"
            ) from e
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            tags=_wandb_tags_list(args.wandb_tags),
            config=_build_wandb_config(
                args,
                world_size=world_size,
                train_bs=train_bs,
                infer_bs=infer_bs,
                train_dataset_len=len(lm_train),
                val_dataset_len=len(lm_val),
            ),
        )

    results: dict[str, float] = {}
    try:
        if args.mode in ("train", "both"):
            train_sampler.set_epoch(args.seed)
            results.update(
                profile_train(
                    model,
                    optim,
                    device,
                    train_iter,
                    steps=args.steps,
                    warmup=args.warmup,
                    amp_dtype=amp_dtype,
                    grad_clip=args.grad_clip,
                )
            )
        if args.mode in ("infer", "both"):
            val_sampler.set_epoch(0)
            results.update(
                profile_infer(
                    model,
                    device,
                    val_iter,
                    steps=args.steps,
                    warmup=args.warmup,
                    amp_dtype=amp_dtype,
                )
            )

        _barrier()
        if rank == 0:
            print("[results]", flush=True)
            for k in sorted(results):
                print(f"  {k}: {results[k]:.6g}", flush=True)
            if args.plot_jsonl:
                row: dict[str, Any] = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "strategy": args.strategy,
                    "world_size": world_size,
                    "model": args.model,
                    "mode": args.mode,
                    "seq_len": args.seq_len,
                    "per_device_batch_train": train_bs,
                    "per_device_batch_infer": infer_bs,
                    "steps": args.steps,
                    "warmup": args.warmup,
                    "bf16": args.bf16,
                    "fp16": args.fp16,
                    "auto_max_batch": args.auto_max_batch,
                    "backend": args.backend,
                }
                row.update({k: float(v) for k, v in results.items()})
                plot_path = os.path.abspath(args.plot_jsonl)
                parent = os.path.dirname(plot_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(plot_path, "a", encoding="utf-8") as pf:
                    pf.write(json.dumps(row, sort_keys=True) + "\n")
                print(f"[plot] appended row to {plot_path}", flush=True)
            print(
                "[note] For max VRAM: raise seq_len, use --auto-max-batch (and --batch-search-cap), prefer --bf16. "
                "Peak MiB is max across ranks. Avoid --gradient-checkpointing if you want highest activation memory.",
                flush=True,
            )
            if wandb_run is not None:
                import wandb

                wandb.log(results)
    finally:
        if rank == 0 and wandb_run is not None:
            import wandb

            wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
