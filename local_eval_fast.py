#!/usr/bin/env python3
"""Fast local CE evaluation for Jefcoder / Jefqwen checkpoints.

Loads checkpoint ``configuration_jefqwen.py`` / ``modeling_jefqwen.py`` via
``trust_remote_code=True``.

Default workflow:
  1. download pretokenized .npy shards into model_eval/eval_datasets/
  2. python model_eval/local_eval_fast.py --model /path/to/checkpoint

Examples:
  python model_eval/local_eval_fast.py \\
    --model /path/to/jefcoder-checkpoint \\
    --eval-config eval/local_eval_datasets.json \\
    --n-sequences 32 --seq-len 2048 --batch-size 8 --gpus auto

  python model_eval/local_eval_fast.py \\
    --model /path/to/checkpoint --ids 0-7 --dataset-dir token_shards
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch

_MODEL_EVAL_ROOT = pathlib.Path(__file__).resolve().parent
if str(_MODEL_EVAL_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_MODEL_EVAL_ROOT.parent))

from model_eval.load import load_model, parse_gpu_ids
from model_eval.losses import DEFAULT_LM_HEAD_CHUNK, compute_batch_losses
from model_eval.manifest import (
    DEFAULT_TOKEN_SHARD_DIR,
    FINEWEBEDU_MANIFEST_URL,
    append_evaluation_history,
    build_eval_units,
    build_shard_index,
    eval_config_dataset_dirs,
    fetch_manifest,
    load_eval_datasets_config,
    local_token_shard_path,
    lookup_by_id,
    parse_id_spec,
    quarter_seq_offset,
    resolve_eval_config_path,
    token_shard_manifest_cache_path,
    utcnow_iso,
    validate_ids,
)
from model_eval.shard_io import count_packed_sequences, load_token_shard

log = logging.getLogger("jefqwen_eval")

_QUIET_LOGGERS = (
    "model_eval",
    "transformers",
    "accelerate",
    "httpx",
)

DEFAULT_EVAL_CONFIG = "eval/local_eval_datasets.json"


def _meta_key(file_id: int) -> str:
    return f"{file_id:05d}"


def _load_meta(cache_dir: pathlib.Path) -> dict[str, Any]:
    path = cache_dir / "_meta.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _validate_cache_entry(
    meta: dict[str, Any],
    file_id: int,
    seq_len: int,
    n_sequences: int | None,
) -> None:
    key = _meta_key(file_id)
    entry = meta.get(key)
    if entry is None:
        log.warning("id %05d has no _meta.json entry; skipping param check", file_id)
        return
    problems = []
    if entry.get("seq_len") != seq_len:
        problems.append(f"seq_len {entry['seq_len']} != {seq_len}")
    if n_sequences is not None and entry.get("n_sequences") != n_sequences:
        problems.append(f"n_sequences {entry['n_sequences']} != {n_sequences}")
    if problems:
        raise ValueError(
            f"id {file_id:05d}: cached data doesn't match requested params "
            f"({'; '.join(problems)}). Re-run pretokenize_parquets.py with "
            f"matching --seq-len / --n-sequences."
        )


def _load_models(
    repo: str,
    gpu_ids: list[int],
    shard: bool,
) -> tuple[dict, dict]:
    models: dict = {}
    devices: dict = {}

    if shard:
        log.info("loading sharded model across GPUs %s", gpu_ids)
        model = load_model(repo, device=None, label="eval-shard", shard_across_gpus=gpu_ids)
        input_device = f"cuda:{gpu_ids[0]}"
        try:
            for name, dev in model.hf_device_map.items():
                if "embed_tokens" in name:
                    input_device = f"cuda:{dev}" if isinstance(dev, int) else str(dev)
                    break
        except AttributeError:
            pass
        models["sharded"] = model
        devices["sharded"] = input_device
        log.info("sharded model ready, input device: %s", input_device)
    else:
        for gid in gpu_ids:
            log.info("loading replica on cuda:%d", gid)
            models[gid] = load_model(repo, f"cuda:{gid}", label=f"eval-gpu{gid}")
            devices[gid] = f"cuda:{gid}"
        log.info("loaded %d replica(s)", len(gpu_ids))

    return models, devices


def _run_losses(
    model: Any,
    token_batches: list[list[int]],
    device: str,
    chunk_size: int,
) -> list[float]:
    return compute_batch_losses(model, token_batches, device, chunk_size=chunk_size)


def _dispatch_parallel(
    models: dict,
    devices: dict,
    sequences: np.ndarray | list,
    batch_size: int,
    chunk_size: int,
) -> list[float]:
    if isinstance(sequences, np.ndarray):
        seq_list: list[list[int]] = sequences.tolist()
    else:
        seq_list = list(sequences)

    batches = [seq_list[i : i + batch_size] for i in range(0, len(seq_list), batch_size)]

    sharded = "sharded" in models
    gpu_keys = ["sharded"] if sharded else list(models.keys())
    n_workers = len(gpu_keys)

    assignments: list[tuple[Any, list[list[int]]]] = [
        (gpu_keys[i % n_workers], batch) for i, batch in enumerate(batches)
    ]

    results: dict[int, list[float]] = {}

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures: dict[Any, int] = {}
        for batch_idx, (key, batch) in enumerate(assignments):
            fut = pool.submit(
                _run_losses,
                models[key],
                batch,
                devices[key],
                chunk_size,
            )
            futures[fut] = batch_idx

        for fut in as_completed(futures):
            batch_idx = futures[fut]
            results[batch_idx] = fut.result()

    flat: list[float] = []
    for i in range(len(batches)):
        flat.extend(results[i])
    return flat


def _print_summary(
    *,
    model: str,
    file_results: list[dict],
    aggregate_mean_loss: float,
    n_sequences: int,
    seq_len: int,
    total_wall_s: float,
    history_path: pathlib.Path,
    lm_head_chunk: int,
    per_dataset: list[dict] | None = None,
) -> None:
    id_w = max(8, max((len(str(r.get("eval_key", r.get("id", "")))) for r in file_results), default=8))
    label_w = max(
        len("shard"),
        max(len(str(r.get("label", f"{r.get('id', ''):05d}.npy"))) for r in file_results),
    )
    loss_w = 10

    lines = [
        "",
        "=" * 72,
        "JEFQWEN LOCAL EVAL SUMMARY",
        "=" * 72,
        f"model:              {model}",
        f"files evaluated:    {len(file_results)}",
        f"sequences / file:   {n_sequences}",
        f"seq_len:            {seq_len}",
        f"lm_head_chunk:      {lm_head_chunk}",
        f"wall_time_s:        {total_wall_s:.1f}",
        "",
    ]
    if per_dataset:
        lines.append(f"{'dataset':<{label_w}}  {'mean_loss':>{loss_w}}")
        lines.append(f"{'-' * label_w}  {'-' * loss_w}")
        for row in per_dataset:
            lines.append(
                f"{str(row.get('dataset', '')):<{label_w}}  "
                f"{float(row['mean_loss']):>{loss_w}.6f}"
            )
        lines.extend(["", f"{'TOTAL (avg datasets)':<{label_w}}  {aggregate_mean_loss:>{loss_w}.6f}", ""])
    lines.extend([
        f"{'eval_key':<{id_w}}  {'shard':<{label_w}}  {'mean_loss':>{loss_w}}",
        f"{'-' * id_w}  {'-' * label_w}  {'-' * loss_w}",
    ])
    for row in file_results:
        label = row.get("label", f"{row.get('id', ''):05d}.npy")
        key = str(row.get("eval_key", row.get("id", "")))
        lines.append(
            f"{key:<{id_w}}  {label:<{label_w}}  {row['mean_loss']:>{loss_w}.6f}"
        )
    lines.extend([
        "",
        f"{'aggregate_mean_loss':<{id_w + 2 + label_w}}  {aggregate_mean_loss:>{loss_w}.6f}",
        f"history:            {history_path}",
        "=" * 72,
        "",
    ])
    print("\n".join(lines), flush=True)


def _evaluate_one_unit(
    *,
    unit: dict[str, Any],
    dataset_dir: pathlib.Path,
    shard_index: list[dict],
    models: dict,
    devices: dict,
    seq_len: int,
    n_sequences: int,
    batch_size: int,
    lm_head_chunk: int,
    idx: int,
    n_total: int,
) -> dict[str, Any]:
    shard_id = int(unit["shard_id"])
    row = lookup_by_id(shard_index, shard_id) if shard_index else {}
    label = row.get("source_file") or row.get("key") or f"{shard_id:05d}.npy"
    if label and "/" in str(label):
        label = str(label).rsplit("/", 1)[-1]
    eval_key = str(unit.get("eval_key") or f"{unit.get('dataset')}:{shard_id}")

    seq_offset = 0
    if unit.get("sequence_quarters"):
        npy_path = local_token_shard_path(dataset_dir, shard_id)
        n_seq_total = count_packed_sequences(npy_path, seq_len=seq_len)
        quarter = int(unit.get("quarter") or 0)
        seq_offset = quarter_seq_offset(n_seq_total, quarter, n_sequences)
        label = f"{label} q{quarter}"

    log.info("[%d/%d] evaluating %s (%s)", idx, n_total, eval_key, label)
    t0 = time.monotonic()

    npy_path = local_token_shard_path(dataset_dir, shard_id)
    tokens = load_token_shard(
        npy_path,
        seq_len=seq_len,
        n_sequences=n_sequences,
        seq_offset=seq_offset,
    )
    n_seq = tokens.shape[0]

    losses = _dispatch_parallel(models, devices, tokens, batch_size, lm_head_chunk)
    mean_loss = float(np.mean(losses))
    wall = time.monotonic() - t0

    log.info("%s: mean_loss=%.6f n_seq=%d wall=%.1fs", eval_key, mean_loss, n_seq, wall)

    result: dict[str, Any] = {
        "id": shard_id,
        "eval_key": eval_key,
        "dataset": str(unit.get("dataset") or ""),
        "label": label,
        "n_sequences": n_seq,
        "mean_loss": mean_loss,
        "wall_time_s": round(wall, 2),
        "seq_offset": seq_offset,
    }
    if unit.get("quarter") is not None:
        result["quarter"] = int(unit["quarter"])
    result["shard_key"] = row.get("key")
    result["source_file"] = row.get("source_file")
    return result


def _run_multi_dataset_eval(
    *,
    args: argparse.Namespace,
    models: dict,
    devices: dict,
    data_root: pathlib.Path,
) -> tuple[list[dict], list[dict], float]:
    cfg_path = resolve_eval_config_path(args.eval_config, base=_MODEL_EVAL_ROOT)
    cfg = load_eval_datasets_config(cfg_path, data_root=data_root)
    seq_len = int(args.seq_len or cfg.get("seq_len") or 2048)
    n_sequences = int(args.n_sequences or cfg.get("n_sequences") or 128)
    batch_size = int(args.batch_size or cfg.get("batch_size") or 16)

    units: list[tuple[dict[str, Any], pathlib.Path, list[dict]]] = []
    for ds in eval_config_dataset_dirs(cfg, data_root=data_root):
        dataset_dir = pathlib.Path(ds["dataset_dir_abs"])
        manifest_url = str(ds.get("manifest_url") or "")
        manifest = fetch_manifest(
            dataset_dir,
            manifest_url=manifest_url,
            cache_path=token_shard_manifest_cache_path(dataset_dir),
        )
        shard_index = build_shard_index(manifest)
        ds_units = build_eval_units(ds)
        physical_ids = sorted({int(u["shard_id"]) for u in ds_units})
        validate_ids(physical_ids, len(shard_index))
        for file_id in physical_ids:
            npy = local_token_shard_path(dataset_dir, file_id)
            if not npy.is_file():
                log.error(
                    "missing shard: %s — download token shards into the dataset dir first",
                    npy,
                )
                sys.exit(1)
        for unit in ds_units:
            units.append((unit, dataset_dir, shard_index))

    file_results: list[dict] = []
    n_total = len(units)
    for idx, (unit, dataset_dir, shard_index) in enumerate(units, start=1):
        file_results.append(
            _evaluate_one_unit(
                unit=unit,
                dataset_dir=dataset_dir,
                shard_index=shard_index,
                models=models,
                devices=devices,
                seq_len=seq_len,
                n_sequences=n_sequences,
                batch_size=batch_size,
                lm_head_chunk=args.lm_head_chunk,
                idx=idx,
                n_total=n_total,
            )
        )

    per_dataset: list[dict] = []
    by_dataset: dict[str, list[dict]] = {}
    for row in file_results:
        ds_name = str(row.get("dataset") or "")
        by_dataset.setdefault(ds_name, []).append(row)
    for ds_name in sorted(by_dataset):
        rows = by_dataset[ds_name]
        ds_mean = float(np.mean([r["mean_loss"] for r in rows]))
        per_dataset.append(
            {
                "dataset": ds_name,
                "mean_loss": ds_mean,
                "shard_ids": [r["id"] for r in rows],
                "eval_keys": [str(r.get("eval_key") or r["id"]) for r in rows],
            }
        )
    aggregate = float(np.mean([d["mean_loss"] for d in per_dataset])) if per_dataset else 0.0
    return file_results, per_dataset, aggregate


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fast local CE evaluation for Jefcoder / Jefqwen checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --model /path/to/jefcoder --eval-config eval/local_eval_datasets.json\n"
            "  %(prog)s --model /path/to/checkpoint --ids 0-7 --dataset-dir token_shards\n"
        ),
    )
    ap.add_argument("--model", required=True, help="Local Jefqwen checkpoint path or HF repo id")
    ap.add_argument(
        "--eval-config",
        default=None,
        help=f"Multi-dataset eval JSON (default when using multi-dataset: {DEFAULT_EVAL_CONFIG})",
    )
    ap.add_argument(
        "--ids",
        default=None,
        help="Comma-separated ids and/or ranges, e.g. 0-19,42 (single-dataset mode)",
    )
    ap.add_argument(
        "--dataset-dir",
        default=str(_MODEL_EVAL_ROOT / DEFAULT_TOKEN_SHARD_DIR),
        help="Directory with pretokenized .npy shards ({id:05d}.npy)",
    )
    ap.add_argument(
        "--manifest-url",
        default=FINEWEBEDU_MANIFEST_URL,
        help="Finewebedu manifest URL (for shard metadata in history)",
    )
    ap.add_argument(
        "--legacy-cache",
        action="store_true",
        help="Use legacy 2D int32 token_cache from pretokenize_parquets.py",
    )
    ap.add_argument("--seq-len", type=int, default=2048, help="Tokens per sequence")
    ap.add_argument(
        "--n-sequences",
        type=int,
        default=128,
        help="Sequences evaluated per shard (first N packed windows)",
    )
    ap.add_argument("--batch-size", type=int, default=16, help="Sequences per forward pass")
    ap.add_argument("--gpus", default="auto", help="Comma-separated GPU ids or 'auto'")
    ap.add_argument(
        "--shard",
        action="store_true",
        help="Shard model across all listed GPUs (accelerate device_map=auto)",
    )
    ap.add_argument(
        "--lm-head-chunk",
        type=int,
        default=DEFAULT_LM_HEAD_CHUNK,
        help=(
            f"Sequence positions per lm_head matmul chunk (default {DEFAULT_LM_HEAD_CHUNK}). "
            "Reduce if OOM on large vocabs."
        ),
    )
    ap.add_argument(
        "--history",
        default="evaluation_history.json",
        help="Append-only results file (newest run first)",
    )
    ap.add_argument(
        "--data-root",
        default=str(_MODEL_EVAL_ROOT),
        help="Root for eval_datasets/ paths in --eval-config (default: model_eval/)",
    )
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = ap.parse_args()

    if not args.eval_config and not args.ids:
        args.eval_config = DEFAULT_EVAL_CONFIG
    if args.eval_config and args.ids:
        ap.error("use either --eval-config or --ids, not both")

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        for name in _QUIET_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

    if not torch.cuda.is_available():
        log.error("CUDA is required for local evaluation")
        sys.exit(1)

    gpu_ids = parse_gpu_ids(args.gpus)
    if not gpu_ids:
        log.error("no GPUs available")
        sys.exit(1)

    shard_mode = args.shard or os.environ.get("SHARD_ACROSS_GPUS", "0") == "1"
    data_root = pathlib.Path(args.data_root).resolve()

    models, devices = _load_models(args.model, gpu_ids, shard_mode)

    run_started = time.monotonic()
    file_results: list[dict] = []
    per_dataset: list[dict] | None = None

    try:
        if args.eval_config:
            file_results, per_dataset, aggregate = _run_multi_dataset_eval(
                args=args,
                models=models,
                devices=devices,
                data_root=data_root,
            )
            cfg = load_eval_datasets_config(
                resolve_eval_config_path(args.eval_config, base=_MODEL_EVAL_ROOT),
                data_root=data_root,
            )
            seq_len = int(args.seq_len or cfg.get("seq_len") or 2048)
            n_sequences = int(args.n_sequences or cfg.get("n_sequences") or 128)
            batch_size = int(args.batch_size or cfg.get("batch_size") or 16)
            dataset_dir = data_root / "eval_datasets"
            manifest_url = None
            eval_config_path = str(resolve_eval_config_path(args.eval_config, base=_MODEL_EVAL_ROOT))
            file_ids = [str(r.get("eval_key") or r["id"]) for r in file_results]
        else:
            dataset_dir = pathlib.Path(args.dataset_dir)
            file_ids = parse_id_spec(args.ids)
            seq_len = args.seq_len
            n_sequences = args.n_sequences
            batch_size = args.batch_size
            eval_config_path = None
            manifest_url = args.manifest_url

            shard_index: list[dict] | None = None
            if not args.legacy_cache:
                manifest = fetch_manifest(
                    dataset_dir,
                    manifest_url=args.manifest_url,
                    cache_path=token_shard_manifest_cache_path(dataset_dir),
                )
                shard_index = build_shard_index(manifest)
                validate_ids(file_ids, len(shard_index))

            meta = _load_meta(dataset_dir) if args.legacy_cache else {}

            for file_id in file_ids:
                npy = local_token_shard_path(dataset_dir, file_id)
                if not npy.is_file():
                    log.error(
                        "missing shard: %s — download token shards first",
                        npy,
                    )
                    sys.exit(1)
                if args.legacy_cache:
                    _validate_cache_entry(meta, file_id, args.seq_len, args.n_sequences)

            n_files = len(file_ids)
            for idx, file_id in enumerate(file_ids, start=1):
                row = shard_index[file_id] if shard_index is not None else {}
                unit = {"shard_id": file_id, "eval_key": str(file_id), "dataset": ""}
                file_results.append(
                    _evaluate_one_unit(
                        unit=unit,
                        dataset_dir=dataset_dir,
                        shard_index=shard_index or [],
                        models=models,
                        devices=devices,
                        seq_len=seq_len,
                        n_sequences=n_sequences,
                        batch_size=batch_size,
                        lm_head_chunk=args.lm_head_chunk,
                        idx=idx,
                        n_total=n_files,
                    )
                )
            aggregate = float(np.mean([r["mean_loss"] for r in file_results]))
    finally:
        for model in models.values():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_wall = time.monotonic() - run_started

    entry: dict[str, Any] = {
        "timestamp": utcnow_iso(),
        "model": args.model,
        "arch": "jefqwen_text",
        "dataset_dir": str(dataset_dir),
        "eval_config": eval_config_path,
        "manifest_url": manifest_url if not args.legacy_cache else None,
        "tokenizer": "jefout/6_29_base",
        "file_ids": file_ids,
        "n_sequences_per_file": file_results[0]["n_sequences"] if file_results else 0,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "lm_head_chunk": args.lm_head_chunk,
        "gpus": gpu_ids,
        "shard": shard_mode,
        "files": file_results,
        "aggregate_mean_loss": aggregate,
        "total_sequences": sum(r["n_sequences"] for r in file_results),
        "wall_time_s": round(total_wall, 2),
    }
    if per_dataset is not None:
        entry["per_dataset"] = per_dataset
        entry["datasets"] = per_dataset

    history_path = pathlib.Path(args.history)
    if not history_path.is_absolute():
        history_path = _MODEL_EVAL_ROOT / history_path
    append_evaluation_history(history_path, entry)

    _print_summary(
        model=args.model,
        file_results=file_results,
        aggregate_mean_loss=aggregate,
        n_sequences=entry["n_sequences_per_file"],
        seq_len=seq_len,
        total_wall_s=total_wall,
        history_path=history_path,
        lm_head_chunk=args.lm_head_chunk,
        per_dataset=per_dataset,
    )


if __name__ == "__main__":
    main()
