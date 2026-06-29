"""Load Jefqwen (Jefcoder) checkpoints for local evaluation."""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

log = logging.getLogger("model_eval.load")


@contextmanager
def _remote_code_path(model_dir: str | Path | None):
    """Checkpoint ``*.py`` modules use flat imports (``configuration_jefqwen``)."""
    if not model_dir or not os.path.isdir(model_dir):
        yield
        return
    root = str(Path(model_dir).resolve())
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(root)


def _build_sharded_device_map(gpu_ids: list[int]) -> dict[int | str, str]:
    per_gpu_gib = int(os.environ.get("SHARD_PER_GPU_GIB", "240"))
    max_memory: dict[int | str, str] = {}
    for gid in range(torch.cuda.device_count()):
        max_memory[gid] = f"{per_gpu_gib}GiB" if gid in gpu_ids else "0GiB"
    return max_memory


def load_model(
    repo: str | Path,
    device: str | None,
    *,
    label: str = "model",
    shard_across_gpus: list[int] | None = None,
) -> AutoModelForCausalLM:
    """Load a Jefqwen checkpoint via checkpoint ``auto_map`` remote code."""
    repo = str(repo)
    target = (
        f"sharded({','.join(str(g) for g in shard_across_gpus)})"
        if shard_across_gpus
        else device
    )
    log.info("loading %s from %s onto %s (trust_remote_code=True)", label, repo, target)
    t0 = time.time()

    if shard_across_gpus:
        load_kwargs: dict = {
            "device_map": "auto",
            "max_memory": _build_sharded_device_map(shard_across_gpus),
        }
    else:
        load_kwargs = {"device_map": {"": device}}

    model = None
    with _remote_code_path(repo if os.path.isdir(repo) else None):
        for attn_impl in ("flash_attention_2", "sdpa", "eager"):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    repo,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=attn_impl,
                    force_download=False,
                    use_safetensors=True,
                    trust_remote_code=True,
                    **load_kwargs,
                )
                log.info("using attn_implementation=%s", attn_impl)
                break
            except Exception as e:
                log.warning("attn %s failed (%s), trying next", attn_impl, e)
    if model is None:
        raise RuntimeError("could not load jefqwen model with any attention implementation")

    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info("%s loaded: %.1fB params in %.1fs", label, params, time.time() - t0)
    return model


def parse_gpu_ids(gpu_str: str) -> list[int]:
    if gpu_str == "auto":
        return list(range(torch.cuda.device_count()))
    return [int(x.strip()) for x in gpu_str.split(",") if x.strip()]
