"""Manifest and eval-config helpers for local token-shard evaluation."""
from __future__ import annotations

import json
import pathlib
import re
from datetime import datetime, timezone
from typing import Any

import httpx

DEFAULT_TOKEN_SHARD_DIR = "token_shards"
DEFAULT_EVAL_DATASETS_CONFIG = "eval/local_eval_datasets.json"
FINEWEBEDU_MANIFEST_URL = (
    ""
)


def local_token_shard_path(dataset_dir: pathlib.Path | str, file_id: int) -> pathlib.Path:
    return pathlib.Path(dataset_dir) / f"{file_id:05d}.npy"


def token_shard_manifest_cache_path(dataset_dir: pathlib.Path | str) -> pathlib.Path:
    return pathlib.Path(dataset_dir) / "manifest.json"


def fetch_manifest(
    dataset_dir: pathlib.Path | str,
    *,
    manifest_path: pathlib.Path | str | None = None,
    manifest_url: str = FINEWEBEDU_MANIFEST_URL,
    cache_path: pathlib.Path | str | None = None,
) -> dict[str, Any]:
    cache = pathlib.Path(cache_path) if cache_path is not None else token_shard_manifest_cache_path(dataset_dir)
    if manifest_path is not None:
        data = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    elif cache.is_file():
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=30.0)) as client:
            resp = client.get(manifest_url)
            resp.raise_for_status()
            data = resp.json()
        cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def build_shard_index(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    shards = manifest.get("shards") or []
    rows: list[dict[str, Any]] = []
    for item in shards:
        key = item.get("key")
        if not key or not str(key).endswith(".npy"):
            continue
        rows.append(
            {
                "key": str(key),
                "source_file": str(item.get("source_file") or ""),
                "size_bytes": int(item.get("size_bytes", 0)),
                "n_tokens": int(item.get("n_tokens", 0)),
                "sha256": item.get("sha256") or "",
            }
        )
    rows.sort(key=lambda r: r["source_file"] or r["key"])
    for i, row in enumerate(rows):
        row["id"] = i
    return rows


def parse_id_spec(id_spec: str) -> list[int]:
    out: set[int] = set()
    for part in id_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            m = re.fullmatch(r"(\d+)-(\d+)", part)
            if not m:
                raise ValueError(f"invalid id range: {part!r}")
            start, end = int(m.group(1)), int(m.group(2))
            if end < start:
                raise ValueError(f"invalid id range (end < start): {part!r}")
            out.update(range(start, end + 1))
        else:
            if not part.isdigit():
                raise ValueError(f"invalid id: {part!r}")
            out.add(int(part))
    if not out:
        raise ValueError("no file ids provided")
    return sorted(out)


def validate_ids(file_ids: list[int], total_files: int) -> None:
    bad = [i for i in file_ids if i < 0 or i >= total_files]
    if bad:
        raise ValueError(
            f"file id(s) out of range 0..{total_files - 1}: {bad[:20]}"
            + (f" (+{len(bad) - 20} more)" if len(bad) > 20 else "")
        )


def lookup_by_id(file_index: list[dict[str, Any]], file_id: int) -> dict[str, Any]:
    if file_id < 0 or file_id >= len(file_index):
        raise KeyError(f"file id {file_id} not in manifest index (0..{len(file_index) - 1})")
    return file_index[file_id]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def quarter_seq_offset(n_seq: int, quarter: int, n_take: int, quarters: int = 4) -> int:
    if n_seq <= 0:
        raise ValueError(f"n_seq must be positive, got {n_seq}")
    start = quarter * n_seq // quarters
    end = (quarter + 1) * n_seq // quarters
    mid = start + (end - start) // 2
    offset = mid - n_take // 2
    return max(start, min(offset, max(start, end - n_take)))


def resolve_eval_config_path(config_path: pathlib.Path | str, *, base: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(config_path)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_eval_datasets_config(
    config_path: pathlib.Path | str,
    *,
    data_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    root = data_root or pathlib.Path(__file__).resolve().parent
    path = resolve_eval_config_path(config_path, base=root)
    if not path.is_file():
        raise FileNotFoundError(f"eval datasets config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid eval config (expected object): {path}")
    datasets = data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"eval config must include non-empty datasets[]: {path}")
    return data


def build_eval_units(dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    name = str(dataset_cfg.get("name") or "").strip()
    if not name:
        raise ValueError("dataset config missing name")
    shard_ids = [int(x) for x in (dataset_cfg.get("shard_ids") or [])]
    if not shard_ids:
        raise ValueError(f"dataset {name!r} missing shard_ids")
    if dataset_cfg.get("sequence_quarters"):
        shard_id = shard_ids[0]
        quarters = int(dataset_cfg.get("quarters") or 4)
        return [
            {
                "dataset": name,
                "short_name": str(dataset_cfg.get("short_name") or name),
                "shard_id": shard_id,
                "quarter": q,
                "eval_key": f"{name}:{shard_id}/q{q}",
                "sequence_quarters": True,
            }
            for q in range(quarters)
        ]
    return [
        {
            "dataset": name,
            "short_name": str(dataset_cfg.get("short_name") or name),
            "shard_id": sid,
            "eval_key": f"{name}:{sid}",
            "sequence_quarters": False,
        }
        for sid in shard_ids
    ]


def eval_config_dataset_dirs(cfg: dict[str, Any], *, data_root: pathlib.Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ds in cfg.get("datasets") or []:
        if not isinstance(ds, dict):
            continue
        rel = str(ds.get("dataset_dir") or "").strip()
        entry = dict(ds)
        entry["dataset_dir_abs"] = (data_root / rel).resolve() if rel else data_root
        out.append(entry)
    return out


def append_evaluation_history(history_path: pathlib.Path | str, entry: dict[str, Any]) -> None:
    path = pathlib.Path(history_path)
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc = {"history": []}
    else:
        doc = {"history": []}
    history = doc.get("history")
    if not isinstance(history, list):
        history = []
    history.insert(0, entry)
    doc["history"] = history
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
