#!/usr/bin/env python3
"""
Multi-GPU LLM training / inference profiler: DDP vs FSDP.

Runs under torchrun for multi-GPU (NCCL on NVIDIA; PyTorch still uses the "nccl"
backend name on ROCm/AMD with RCCL underneath).

Examples:
  # 4 GPUs, FSDP, training + inference, small default model
  torchrun --nproc_per_node=4 profile_llm_distributed.py --strategy fsdp --mode both

  # DDP training only, larger batch
  torchrun --nproc_per_node=8 profile_llm_distributed.py --strategy ddp --mode train \\
      --per-device-batch 4 --seq-len 2048 --steps 50

  # Single GPU (no torchrun)
  python profile_llm_distributed.py --strategy ddp --mode infer --steps 30
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoConfig, AutoModelForCausalLM


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
    rank: int,
) -> nn.Module:
    # Load on CPU then move; FSDP will shard after wrap. DDP moves per device below.
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if dtype is None:
        model = model.float()
    return model


def _synthetic_batch(
    batch: int,
    seq: int,
    vocab: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(0, vocab, (batch, seq), device=device, dtype=torch.long)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}


def profile_train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    batch: int,
    seq: int,
    vocab: int,
    steps: int,
    warmup: int,
    amp_dtype: torch.dtype | None,
    grad_clip: float | None,
) -> dict[str, float]:
    model.train()
    total_tokens = 0
    # Warmup
    for _ in range(warmup):
        batch_d = _synthetic_batch(batch, seq, vocab, device, torch.long)
        with _get_amp_ctx(amp_dtype):
            out = model(**batch_d)
            loss = out.loss if hasattr(out, "loss") and out.loss is not None else out[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    _barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        batch_d = _synthetic_batch(batch, seq, vocab, device, torch.long)
        with _get_amp_ctx(amp_dtype):
            out = model(**batch_d)
            loss = out.loss if hasattr(out, "loss") and out.loss is not None else out[0]
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_tokens += batch * seq

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _barrier()
    elapsed_local = time.perf_counter() - t0

    tok = torch.tensor([float(total_tokens)], device=device)
    tok = _all_reduce_sum(tok)
    el = torch.tensor([elapsed_local], device=device)
    el = _all_reduce_max(el)
    tokens, elapsed_wall = tok[0].item(), el[0].item()
    return {
        "train_elapsed_s": elapsed_wall,
        "train_tokens_global": tokens,
        "train_tokens_per_s": tokens / max(elapsed_wall, 1e-9),
    }


@torch.no_grad()
def profile_infer(
    model: nn.Module,
    device: torch.device,
    *,
    batch: int,
    seq: int,
    vocab: int,
    steps: int,
    warmup: int,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    for _ in range(warmup):
        batch_d = _synthetic_batch(batch, seq, vocab, device, torch.long)
        with _get_amp_ctx(amp_dtype):
            model(**batch_d)
    _barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_tokens = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        batch_d = _synthetic_batch(batch, seq, vocab, device, torch.long)
        with _get_amp_ctx(amp_dtype):
            model(**batch_d)
        total_tokens += batch * seq

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _barrier()
    elapsed_local = time.perf_counter() - t0

    tok = torch.tensor([float(total_tokens)], device=device)
    tok = _all_reduce_sum(tok)
    el = torch.tensor([elapsed_local], device=device)
    el = _all_reduce_max(el)
    tokens, elapsed_wall = tok[0].item(), el[0].item()
    return {
        "infer_elapsed_s": elapsed_wall,
        "infer_tokens_global": tokens,
        "infer_tokens_per_s": tokens / max(elapsed_wall, 1e-9),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile LLM training/inference: DDP vs FSDP")
    p.add_argument("--model", type=str, default="gpt2", help="HF model id (gpt2, meta-llama/Llama-3.2-1B, ...)")
    p.add_argument(
        "--strategy",
        type=str,
        choices=("ddp", "fsdp"),
        default="fsdp",
        help="Distributed wrapping strategy",
    )
    p.add_argument("--mode", type=str, choices=("train", "infer", "both"), default="both")
    p.add_argument("--per-device-batch", type=int, default=2, help="Micro-batch per GPU")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=20, help="Timed steps (after warmup)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--backend", type=str, default="nccl", help="Process group backend (nccl for CUDA/ROCm)")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--bf16", action="store_true", help="Load model and autocast in bfloat16")
    p.add_argument("--fp16", action="store_true", help="Autocast float16 (model may stay fp32)")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--fsdp-sharding", type=str, default="full", choices=("full", "hybrid", "no_shard"))
    p.add_argument(
        "--fsdp-min-num-params",
        type=int,
        default=10_000_000,
        help="size_based_auto_wrap_policy threshold (tune per model size)",
    )
    p.add_argument("--seed", type=int, default=42)
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

    if rank == 0:
        print(
            f"[config] model={args.model} strategy={args.strategy} mode={args.mode} "
            f"world_size={world_size} local_batch={args.per_device_batch} seq_len={args.seq_len} "
            f"steps={args.steps} warmup={args.warmup} bf16={args.bf16} fp16={args.fp16}",
            flush=True,
        )

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    vocab = int(config.vocab_size)

    model = _load_causal_lm(args.model, args.trust_remote_code, load_dtype, rank)

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
        # Keep weights on CPU until FSDP shards them onto this rank's GPU (lower peak VRAM).
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

    results: dict[str, float] = {}
    if args.mode in ("train", "both"):
        results.update(
            profile_train(
                model,
                optim,
                device,
                batch=args.per_device_batch,
                seq=args.seq_len,
                vocab=vocab,
                steps=args.steps,
                warmup=args.warmup,
                amp_dtype=amp_dtype,
                grad_clip=args.grad_clip,
            )
        )
    if args.mode in ("infer", "both"):
        results.update(
            profile_infer(
                model,
                device,
                batch=args.per_device_batch,
                seq=args.seq_len,
                vocab=vocab,
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
        # Throughput note: train_tokens_per_s is summed micro-batch tokens across ranks per second
        # (each rank processes per_device_batch each step).
        print(
            "[note] *_tokens_global / *_tokens_per_s are summed across ranks; "
            "elapsed is max across ranks after synchronize (wall clock).",
            flush=True,
        )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
