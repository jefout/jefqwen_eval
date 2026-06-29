"""Sample pretokenized .npy shards and optionally export for Axolotl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ─── Config ────────────────────────────────────────────────────────────────────
TARGET_SEQUENCES = 1_000_000   # total 2048-token sequences across all datasets
SEQ_LEN          = 2048
SEED             = 42
OUTPUT_PATH      = "sample_1M.npy"
# Per shard: "random" = scattered sequences; "consecutive" = one run from random start.
SEQUENCE_SELECTION = "random"

# HuggingFace export for Axolotl (requires `datasets`, installed with axolotl).
EXPORT_HF_DATASET = True
HF_OUTPUT_PATH    = None   # default: <OUTPUT_PATH stem>_hf
HF_CHUNK_TARGET_MB = 128    # target RAM per export chunk when auto-sizing

EVAL_SEED = 4242
EVAL_SEQUENCES_PER_DATASET = 128
EVAL_NUM_SHARDS_PER_DATASET = 8
EVAL_CACHE_DIR = Path(__file__).resolve().parent.parent / "eval_cache"

# Per-dataset: ratio (fraction of TARGET_SEQUENCES) and num_shards (.npy files to pick).
# Ratios must sum to 1.0. See readme_db.md for dataset reference.
DATASETS = [
    {
        "name": "openthoughts3-1.2m",
        "manifest_url": "",
        "ratio": 1.0,
        "num_shards": 350,
    },
]

SHARD_WORKERS    = 4
MAX_RETRIES      = 3
RATE_LIMIT_WAIT  = 30   # seconds to wait on HTTP 429 before retrying
HEADERS          = {}
_rate_limit_lock = threading.Lock()

random.seed(SEED)
np.random.seed(SEED)

# ─── Thread-safe counter ───────────────────────────────────────────────────────
class AtomicCounter:
    def __init__(self):
        self._val  = 0
        self._lock = threading.Lock()

    def add(self, n):
        with self._lock:
            self._val += n
            return self._val

    @property
    def value(self):
        return self._val

# ─── Session ───────────────────────────────────────────────────────────────────
def make_session():
    s     = requests.Session()
    retry = Retry(total=MAX_RETRIES, backoff_factor=0.5,
                  status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=SHARD_WORKERS * 4,
        pool_maxsize=SHARD_WORKERS * 4,
    )
    s.mount("https://", adapter)
    s.mount("http://",  adapter)
    s.headers.update(HEADERS)
    return s

SESSION = make_session()

# ─── HTTP helpers ─────────────────────────────────────────────────────────────
def session_get(url, **kwargs):
    """GET with retry on HTTP 429 (rate limit)."""
    while True:
        resp = SESSION.get(url, **kwargs)
        if resp.status_code == 429:
            with _rate_limit_lock:
                tqdm.write(f"  Rate limited (429), waiting {RATE_LIMIT_WAIT}s …")
            time.sleep(RATE_LIMIT_WAIT)
            continue
        resp.raise_for_status()
        return resp

def http_get(url, byte_start=None, byte_end=None, max_attempts=5):
    hdrs = {}
    if byte_start is not None:
        hdrs["Range"] = f"bytes={byte_start}-{byte_end}"
    last_err = None
    for attempt in range(max_attempts):
        try:
            return session_get(url, headers=hdrs, timeout=120).content
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            tqdm.write(f"  HTTP retry {attempt + 1}/{max_attempts} after {type(e).__name__}, wait {wait}s")
            time.sleep(wait)
    raise last_err

def fetch_manifest(manifest_url):
    return session_get(manifest_url, timeout=30).json()

def manifest_base_url(manifest_url):
    if "/dataset/" not in manifest_url:
        raise ValueError(f"Cannot derive base URL from manifest: {manifest_url}")
    return manifest_url.rsplit("/dataset/", 1)[0] + "/"

def dataset_seed(base_seed, dataset_name):
    digest = hashlib.blake2b(
        f"{base_seed}:{dataset_name}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")

# ─── Derive .npy layout from manifest ─────────────────────────────────────────
def header_info_from_manifest(shard, dtype_str):
    """Byte offset and dtype for range reads; manifest already has size_bytes/n_tokens."""
    dtype_obj = np.dtype(dtype_str)
    header_offset = shard["size_bytes"] - shard["n_tokens"] * dtype_obj.itemsize
    if header_offset < 0:
        raise ValueError(
            f"invalid shard metadata {shard['key']!r}: "
            f"size_bytes={shard['size_bytes']}, n_tokens={shard['n_tokens']}"
        )
    return header_offset, dtype_obj

# ─── Pick sequence slices ─────────────────────────────────────────────────────
def pick_slices(n_tokens, n_sample, seq_len=SEQ_LEN, selection=SEQUENCE_SELECTION):
    n_seqs_in_shard = n_tokens // seq_len
    n_seqs_to_take  = min(n_sample // seq_len, n_seqs_in_shard)
    if n_seqs_to_take == 0:
        return [], 0

    if selection == "consecutive":
        seq_start = random.randint(0, n_seqs_in_shard - n_seqs_to_take)
        return [(seq_start, seq_start + n_seqs_to_take - 1)], n_seqs_to_take * seq_len

    if selection != "random":
        raise ValueError(f"unknown sequence selection mode: {selection!r}")

    seq_indices = sorted(random.sample(range(n_seqs_in_shard), n_seqs_to_take))

    slices  = []
    s_start = seq_indices[0]
    s_end   = seq_indices[0]
    for idx in seq_indices[1:]:
        if idx == s_end + 1:
            s_end = idx
        else:
            slices.append((s_start, s_end))
            s_start = s_end = idx
    slices.append((s_start, s_end))

    return slices, n_seqs_to_take * seq_len

# ─── Fetch one contiguous slice ───────────────────────────────────────────────
def fetch_slice(url, header_offset, dtype_obj, seq_start, seq_end, seq_len,
                pbar_tok, pbar_net, tok_ctr, byte_ctr):
    token_bytes = dtype_obj.itemsize
    tok_start   = seq_start * seq_len
    tok_end_ex  = (seq_end + 1) * seq_len
    byte_start  = header_offset + tok_start  * token_bytes
    byte_end    = header_offset + tok_end_ex * token_bytes - 1

    raw  = http_get(url, byte_start, byte_end)
    arr  = np.frombuffer(raw, dtype=dtype_obj).copy()
    seqs = arr.reshape(-1, seq_len)

    n_toks  = seqs.size
    n_bytes = len(raw)
    tok_ctr.add(n_toks)
    byte_ctr.add(n_bytes)
    pbar_tok.update(n_toks)
    pbar_net.update(n_bytes)

    return seqs

# ─── Process one shard ────────────────────────────────────────────────────────
def process_shard(shard, n_sample, base_url, dataset_name, dtype_str,
                  tok_ctr, byte_ctr, pbar_tok, pbar_shard, pbar_net,
                  sequence_selection=SEQUENCE_SELECTION):
    key    = shard["key"]
    url    = urljoin(base_url, key)
    name   = key.split("/")[-1]

    header_offset, dtype_obj = header_info_from_manifest(shard, dtype_str)
    slices, _actual_tokens = pick_slices(
        shard["n_tokens"], n_sample, selection=sequence_selection
    )

    if not slices:
        pbar_shard.update(1)
        return np.array([], dtype=dtype_obj).reshape(0, SEQ_LEN)

    parts = []
    for seq_start, seq_end in slices:
        seqs = fetch_slice(url, header_offset, dtype_obj,
                           seq_start, seq_end, SEQ_LEN,
                           pbar_tok, pbar_net, tok_ctr, byte_ctr)
        parts.append(seqs)

    result = np.concatenate(parts, axis=0)

    pbar_shard.update(1)
    pbar_shard.set_postfix({
        "shard": name,
        "collected": f"{tok_ctr.value / 1e9:.3f}B tok",
    })
    return result

def validate_sampling_config(all_shards, target_sequences, num_shards_to_select, dataset_name):
    if target_sequences <= 0:
        raise ValueError(
            f"{dataset_name}: target sequences must be positive, got {target_sequences}"
        )
    if num_shards_to_select <= 0:
        raise ValueError(
            f"{dataset_name}: num_shards must be positive, got {num_shards_to_select}"
        )
    if num_shards_to_select > len(all_shards):
        raise ValueError(
            f"{dataset_name}: num_shards ({num_shards_to_select}) exceeds "
            f"dataset shard count ({len(all_shards)})"
        )
    if num_shards_to_select > target_sequences:
        raise ValueError(
            f"{dataset_name}: num_shards ({num_shards_to_select}) exceeds "
            f"allocated sequences ({target_sequences}); each shard needs at least 1 sequence"
        )


def validate_multi_dataset_config(datasets, target_sequences):
    if not datasets:
        raise ValueError("DATASETS must contain at least one entry")
    if target_sequences <= 0:
        raise ValueError(f"TARGET_SEQUENCES must be positive, got {target_sequences}")

    ratios = [entry["ratio"] for entry in datasets]
    if any(r <= 0 for r in ratios):
        raise ValueError("Each dataset ratio must be positive")

    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Dataset ratios must sum to 1.0, got {ratio_sum}")

    for entry in datasets:
        if entry["num_shards"] <= 0:
            raise ValueError(f"{entry['name']}: num_shards must be positive")
        if entry["num_shards"] > target_sequences:
            raise ValueError(
                f"{entry['name']}: num_shards ({entry['num_shards']}) exceeds "
                f"TARGET_SEQUENCES ({target_sequences})"
            )


def allocate_dataset_sequences(target_sequences, ratios):
    total_w = sum(ratios)
    raw     = [target_sequences * w / total_w for w in ratios]
    targets = [int(r) for r in raw]
    remainder = target_sequences - sum(targets)
    order = sorted(range(len(ratios)), key=lambda i: -(raw[i] - targets[i]))
    for i in order[:remainder]:
        targets[i] += 1
    return targets


def select_shards_and_quotas(all_shards, target_sequences, num_shards, seq_len, rng):
    """
    Randomly pick num_shards .npy files, assign equal sequence quotas, and
    redistribute shortfall from capacity-limited shards to others with spare room.
    """
    selected   = rng.sample(all_shards, num_shards)
    base, rem  = divmod(target_sequences, num_shards)
    quotas     = [base + (1 if i < rem else 0) for i in range(num_shards)]
    capacities = [s["n_tokens"] // seq_len for s in selected]

    deficit = 0
    for i in range(num_shards):
        if quotas[i] > capacities[i]:
            deficit += quotas[i] - capacities[i]
            quotas[i] = capacities[i]

    while deficit > 0:
        spare_idxs = [i for i in range(num_shards) if quotas[i] < capacities[i]]
        if not spare_idxs:
            break
        for i in spare_idxs:
            if deficit == 0:
                break
            room = capacities[i] - quotas[i]
            add  = min(room, deficit)
            quotas[i] += add
            deficit -= add

    sequences_planned = sum(quotas)
    plan = [
        (selected[i], quotas[i] * seq_len)
        for i in range(num_shards)
        if quotas[i] > 0
    ]
    tokens_planned = sequences_planned * seq_len
    return plan, tokens_planned, sequences_planned, quotas, selected


def build_multi_dataset_plan(datasets, target_sequences, seq_len, seed):
    validate_multi_dataset_config(datasets, target_sequences)

    ratios    = [entry["ratio"] for entry in datasets]
    allocated = allocate_dataset_sequences(target_sequences, ratios)

    sample_plan       = []
    dataset_summaries = []
    tokens_planned    = 0
    sequences_planned = 0
    expected_dtype    = None

    for entry, target_for_dataset in zip(datasets, allocated):
        name          = entry["name"]
        manifest_url  = entry["manifest_url"]
        num_shards    = entry["num_shards"]
        base_url      = manifest_base_url(manifest_url)

        manifest   = fetch_manifest(manifest_url)
        all_shards = manifest["shards"]
        dtype      = manifest["dtype"]

        if expected_dtype is None:
            expected_dtype = dtype
        elif dtype != expected_dtype:
            raise ValueError(
                f"{name}: dtype {dtype!r} does not match bundle dtype {expected_dtype!r}"
            )

        validate_sampling_config(all_shards, target_for_dataset, num_shards, name)

        rng = random.Random(dataset_seed(seed, name))
        plan, ds_tokens, ds_sequences, quotas, _selected = select_shards_and_quotas(
            all_shards, target_for_dataset, num_shards, seq_len, rng
        )

        for shard, n_sample in plan:
            sample_plan.append((shard, n_sample, base_url, name))

        tokens_planned    += ds_tokens
        sequences_planned += ds_sequences

        summary = {
            "name": name,
            "manifest_url": manifest_url,
            "base_url": base_url,
            "allocated_sequences": target_for_dataset,
            "planned_sequences": ds_sequences,
            "num_shards_requested": num_shards,
            "num_shards_in_plan": len(plan),
            "quota_min": min(quotas) if quotas else 0,
            "quota_max": max(quotas) if quotas else 0,
            "total_shards": len(all_shards),
        }
        dataset_summaries.append(summary)

        if ds_sequences < target_for_dataset:
            tqdm.write(
                f"  Warning: {name} can only supply "
                f"{ds_sequences:,} / {target_for_dataset:,} sequences after redistribution"
            )

    return sample_plan, tokens_planned, sequences_planned, dataset_summaries, expected_dtype


def default_hf_output_path(npy_path: str | Path) -> str:
    path = Path(npy_path)
    if path.suffix == ".npy":
        return str(path.with_suffix("")) + "_hf"
    return f"{path}_hf"


def export_chunk_size_for_seq_len(seq_len: int, requested: int | None = None) -> int:
    if requested is not None and requested > 0:
        return requested
    bytes_per_seq = seq_len * np.dtype(np.uint32).itemsize
    target_bytes = HF_CHUNK_TARGET_MB * 1024 * 1024
    return max(1, int(target_bytes / bytes_per_seq))


# Columns required by axolotl.utils.data.wrappers._is_dataset_already_tokenized
AXOLOTL_TOKENIZED_COLUMNS = ("input_ids", "attention_mask", "labels")


def axolotl_tokenized_features(seq_len: int):
    """HuggingFace Features schema for Axolotl pretokenized causal-LM datasets."""
    from datasets import Features, Sequence, Value

    return Features(
        {
            "input_ids": Sequence(Value("uint32"), length=seq_len),
            "attention_mask": Sequence(Value("int64"), length=seq_len),
            "labels": Sequence(Value("uint32"), length=seq_len),
        }
    )


def _assert_axolotl_tokenized_dataset(dataset, *, seq_len: int) -> None:
    """Verify export matches axolotl pretokenized dataset expectations."""
    missing = [
        col for col in AXOLOTL_TOKENIZED_COLUMNS if col not in dataset.features
    ]
    if missing:
        raise RuntimeError(
            f"HF export missing required columns {missing}; "
            f"expected {list(AXOLOTL_TOKENIZED_COLUMNS)}"
        )
    if len(dataset) > 0:
        row = dataset[0]
        for col in AXOLOTL_TOKENIZED_COLUMNS:
            if len(row[col]) != seq_len:
                raise RuntimeError(
                    f"column {col!r} has length {len(row[col])}, expected {seq_len}"
                )
        if not all(v == 1 for v in row["attention_mask"]):
            raise RuntimeError("attention_mask must be all ones for fixed-length rows")


def export_npy_to_hf_dataset(
    npy_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_size: int | None = None,
) -> Path:
    """Convert a pretokenized (N, seq_len) uint32 .npy file to an Axolotl-ready HF dataset.

    Writes `input_ids`, `attention_mask` (all ones), and `labels` (identical to
    input_ids for causal LM pretraining). Uses memory-mapped reads, chunked
    Parquet staging, and streaming `save_to_disk` so multi‑tens‑of‑GB exports
    stay bounded in RAM.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "HF export requires `datasets` and `pyarrow` (bundled with axolotl). "
            "Install with: pip install datasets pyarrow"
        ) from exc

    npy_path = Path(npy_path)
    output_dir = Path(output_dir)
    if not npy_path.is_file():
        raise FileNotFoundError(npy_path)

    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(
            f"expected pretokenized .npy with shape (sequences, seq_len), got {arr.shape}"
        )
    if arr.dtype != np.dtype("uint32"):
        raise ValueError(f"expected uint32 tokens, got dtype {arr.dtype}")

    n_seqs, seq_len = arr.shape
    chunk_size = export_chunk_size_for_seq_len(seq_len, chunk_size)

    def block_to_table(block: np.ndarray) -> pa.Table:
        n_rows = block.shape[0]
        flat_ids = pa.array(block.reshape(-1), type=pa.uint32())
        input_ids = pa.FixedSizeListArray.from_arrays(flat_ids, seq_len)
        flat_labels = pa.array(np.asarray(block, dtype=np.uint32).reshape(-1), type=pa.uint32())
        labels = pa.FixedSizeListArray.from_arrays(flat_labels, seq_len)
        ones = pa.array(np.ones(n_rows * seq_len, dtype=np.int64))
        attention_mask = pa.FixedSizeListArray.from_arrays(ones, seq_len)
        return pa.table(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    staging_dir = output_dir.parent / f".{output_dir.name}.parquet_staging"
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing HF dataset directory: {output_dir}"
        )
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(
        f"\nExporting HF dataset from {npy_path.name}: "
        f"{n_seqs:,} sequences x {seq_len} tokens -> {output_dir} "
        f"(chunk_size={chunk_size:,})"
    )

    parquet_paths: list[Path] = []
    with tqdm(total=n_seqs, desc="  HF export", unit=" seq", dynamic_ncols=True) as pbar:
        for chunk_idx, start in enumerate(range(0, n_seqs, chunk_size)):
            end = min(start + chunk_size, n_seqs)
            block = np.asarray(arr[start:end], dtype=np.uint32)
            parquet_path = staging_dir / f"train-{chunk_idx:05d}.parquet"
            pq.write_table(block_to_table(block), parquet_path)
            parquet_paths.append(parquet_path)
            pbar.update(end - start)

    tqdm.write("  Building HuggingFace dataset from parquet shards...")
    dataset = load_dataset(
        "parquet",
        data_files=[str(path) for path in parquet_paths],
        split="train",
    )
    dataset = dataset.cast(axolotl_tokenized_features(seq_len))
    _assert_axolotl_tokenized_dataset(dataset, seq_len=seq_len)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir), max_shard_size="512MB")

    shutil.rmtree(staging_dir)

    saved = load_from_disk(str(output_dir))
    _assert_axolotl_tokenized_dataset(saved, seq_len=seq_len)

    disk_bytes = sum(
        f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
    )
    tqdm.write(
        f"  HF dataset saved: {output_dir} "
        f"({n_seqs:,} rows, seq_len={seq_len}, {disk_bytes / 1e9:.2f} GB on disk)"
    )
    return output_dir


def build_single_dataset_eval_plan(
    dataset_entry: dict,
    target_sequences: int,
    num_shards: int,
    seq_len: int,
    seed: int,
) -> tuple[list, dict]:
    """Deterministic per-dataset eval plan (one dataset, fixed shard pick)."""
    name = dataset_entry["name"]
    manifest_url = dataset_entry["manifest_url"]
    base_url = manifest_base_url(manifest_url)
    manifest = fetch_manifest(manifest_url)
    all_shards = manifest["shards"]
    dtype = manifest["dtype"]
    num_shards = min(num_shards, len(all_shards))
    validate_sampling_config(all_shards, target_sequences, num_shards, name)
    rng = random.Random(dataset_seed(seed, name))
    plan, tokens_planned, sequences_planned, quotas, selected = select_shards_and_quotas(
        all_shards, target_sequences, num_shards, seq_len, rng
    )
    summary = {
        "name": name,
        "manifest_url": manifest_url,
        "dtype": dtype,
        "target_sequences": target_sequences,
        "planned_sequences": sequences_planned,
        "num_shards": len(plan),
        "shard_keys": [s["key"] for s, _ in plan],
        "quotas": quotas,
    }
    return [(shard, n_sample, base_url, name, dtype) for shard, n_sample in plan], summary


def fetch_eval_sequences(plan_rows: list) -> np.ndarray:
    """Download eval sequences from a single-dataset plan."""
    tok_ctr = AtomicCounter()
    byte_ctr = AtomicCounter()
    tokens_planned = sum(row[1] for row in plan_rows)
    pbar_tok = tqdm(total=tokens_planned, desc="  eval tok", unit=" tok", leave=False)
    pbar_shard = tqdm(total=len(plan_rows), desc="  eval shards", unit=" shard", leave=False)
    pbar_net = tqdm(
        total=tokens_planned * 4,
        desc="  eval net",
        unit="B",
        unit_scale=True,
        leave=False,
    )
    parts = []
    for shard, n_sample, base_url, dataset_name, dtype in plan_rows:
        arr = process_shard(
            shard,
            n_sample,
            base_url,
            dataset_name,
            dtype,
            tok_ctr,
            byte_ctr,
            pbar_tok,
            pbar_shard,
            pbar_net,
            sequence_selection="random",
        )
        if arr.size:
            parts.append(arr)
    pbar_tok.close()
    pbar_shard.close()
    pbar_net.close()
    if not parts:
        return np.array([], dtype=np.uint32).reshape(0, SEQ_LEN)
    return np.concatenate(parts, axis=0)


def build_fixed_eval_cache(
    cache_dir: Path | str | None = None,
    *,
    datasets: list | None = None,
    sequences_per_dataset: int = EVAL_SEQUENCES_PER_DATASET,
    num_shards: int = EVAL_NUM_SHARDS_PER_DATASET,
    seq_len: int = SEQ_LEN,
    seed: int = EVAL_SEED,
    force: bool = False,
) -> dict:
    """Download and cache fixed per-dataset eval shards for forensics."""
    cache_dir = Path(cache_dir or EVAL_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    datasets = datasets or DATASETS
    manifest_path = cache_dir / "eval_manifest.json"
    summaries = []
    for entry in datasets:
        name = entry["name"]
        out_npy = cache_dir / f"{name}.npy"
        if out_npy.is_file() and not force:
            arr = np.load(out_npy, mmap_mode="r")
            summaries.append({
                "name": name,
                "cache_path": str(out_npy),
                "shape": list(arr.shape),
                "cached": True,
            })
            continue
        tqdm.write(
            f"Building eval cache for {name}: "
            f"{sequences_per_dataset} seq from up to {num_shards} shards ..."
        )
        plan_rows, summary = build_single_dataset_eval_plan(
            entry, sequences_per_dataset, num_shards, seq_len, seed
        )
        tqdm.write(
            f"  plan: {summary['planned_sequences']} seq across "
            f"{summary['num_shards']} shard(s), quotas={summary['quotas']}"
        )
        seqs = fetch_eval_sequences(plan_rows)
        np.save(out_npy, seqs)
        summary["cache_path"] = str(out_npy)
        summary["shape"] = list(seqs.shape)
        summary["cached"] = False
        summaries.append(summary)
    manifest = {
        "seed": seed,
        "seq_len": seq_len,
        "sequences_per_dataset": sequences_per_dataset,
        "num_shards_per_dataset": num_shards,
        "datasets": summaries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample pretokenized shards and optionally export for Axolotl."
    )
    parser.add_argument(
        "--build-eval-cache",
        action="store_true",
        help="download fixed per-dataset eval shards (forensics exp #2) and exit",
    )
    parser.add_argument(
        "--eval-cache-dir",
        default=str(EVAL_CACHE_DIR),
        help="directory for cached eval .npy files",
    )
    parser.add_argument(
        "--eval-sequences",
        type=int,
        default=EVAL_SEQUENCES_PER_DATASET,
        help="sequences per dataset for fixed eval cache",
    )
    parser.add_argument(
        "--eval-num-shards",
        type=int,
        default=EVAL_NUM_SHARDS_PER_DATASET,
        help="number of .npy shards to sample per dataset (128 seq / 8 shards = 16 each)",
    )
    parser.add_argument(
        "--eval-force",
        action="store_true",
        help="re-download eval cache even if files exist",
    )
    parser.add_argument(
        "--export-hf",
        action=argparse.BooleanOptionalAction,
        default=EXPORT_HF_DATASET,
        help="after saving .npy, export an Axolotl-ready HuggingFace dataset",
    )
    parser.add_argument(
        "--hf-output",
        default=None,
        help="HF dataset output directory (default: <npy-stem>_hf)",
    )
    parser.add_argument(
        "--hf-chunk-size",
        type=int,
        default=None,
        help=(
            "sequences per chunk while exporting "
            f"(default: auto from {HF_CHUNK_TARGET_MB}MB target)"
        ),
    )
    parser.add_argument(
        "--export-hf-only",
        metavar="NPY",
        default=None,
        help="skip sampling; convert an existing .npy file to HuggingFace format",
    )
    parser.add_argument(
        "--sequence-selection",
        choices=("random", "consecutive"),
        default=SEQUENCE_SELECTION,
        help=(
            "how to pick sequences within each shard: "
            "random (default) or consecutive from a random start"
        ),
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.build_eval_cache:
        manifest = build_fixed_eval_cache(
            args.eval_cache_dir,
            sequences_per_dataset=args.eval_sequences,
            num_shards=args.eval_num_shards,
            force=args.eval_force,
        )
        tqdm.write(f"Eval cache ready: {args.eval_cache_dir}")
        tqdm.write(json.dumps(manifest, indent=2))
        return

    if args.export_hf_only:
        npy_path = Path(args.export_hf_only)
        hf_output = Path(args.hf_output or default_hf_output_path(npy_path))
        export_npy_to_hf_dataset(
            npy_path,
            hf_output,
            chunk_size=args.hf_chunk_size,
        )
        return

    t0 = time.time()

    tqdm.write("Building multi-dataset sampling plan...")
    sample_plan, tokens_planned, sequences_planned, dataset_summaries, dtype = (
        build_multi_dataset_plan(DATASETS, TARGET_SEQUENCES, SEQ_LEN, SEED)
    )
    shards_needed = len(sample_plan)

    tqdm.write(f"\n{'─'*62}")
    tqdm.write(f"  Bundle          : {len(DATASETS)} datasets")
    tqdm.write(f"  Dtype           : {dtype}")
    tqdm.write(f"  Seq length      : {SEQ_LEN}")
    tqdm.write(f"  Seq selection   : {args.sequence_selection}")
    tqdm.write(f"  Target sequences: {TARGET_SEQUENCES:,}  (across all datasets)")
    tqdm.write(f"  Target tokens   : {TARGET_SEQUENCES * SEQ_LEN:,}  ({TARGET_SEQUENCES * SEQ_LEN / 1e9:.1f}B)")
    tqdm.write(f"{'─'*62}")

    for summary in dataset_summaries:
        tqdm.write(
            f"  {summary['name']}:"
            f" {summary['allocated_sequences']:,} seq allocated,"
            f" {summary['planned_sequences']:,} planned,"
            f" {summary['num_shards_in_plan']}/{summary['total_shards']} shards,"
            f" quota {summary['quota_min']:,}–{summary['quota_max']:,}"
        )

    tqdm.write(f"{'─'*62}")
    if sequences_planned < TARGET_SEQUENCES:
        tqdm.write(
            f"  Warning: bundle can only supply "
            f"{sequences_planned:,} / {TARGET_SEQUENCES:,} sequences after redistribution"
        )

    tqdm.write(f"\nSampling plan:")
    tqdm.write(f"   Shards to fetch : {shards_needed:,}")
    tqdm.write(f"   Sequences total : {sequences_planned:,} / {TARGET_SEQUENCES:,}")
    tqdm.write(f"   Tokens to fetch : {tokens_planned:,}  ({tokens_planned / 1e9:.2f}B)\n")

    tqdm.write("Sampling tokens...\n")

    tok_ctr  = AtomicCounter()
    byte_ctr = AtomicCounter()

    pbar_tok   = tqdm(total=tokens_planned,
                      desc="  Tokens ", unit=" tok",
                      unit_scale=True, unit_divisor=1000,
                      colour="cyan",   dynamic_ncols=True, position=0)
    pbar_shard = tqdm(total=shards_needed,
                      desc="  Shards ", unit=" shard",
                      colour="green",  dynamic_ncols=True, position=1)
    pbar_net   = tqdm(total=tokens_planned * np.dtype(dtype).itemsize,
                      desc="  Network", unit="B",
                      unit_scale=True, unit_divisor=1024,
                      colour="yellow", dynamic_ncols=True, position=2)

    results = [None] * len(sample_plan)

    with ThreadPoolExecutor(max_workers=SHARD_WORKERS) as ex:
        future_to_idx = {
            ex.submit(
                process_shard,
                shard, n_sample, base_url, dataset_name, dtype,
                tok_ctr, byte_ctr, pbar_tok, pbar_shard, pbar_net,
                args.sequence_selection,
            ): i
            for i, (shard, n_sample, base_url, dataset_name) in enumerate(sample_plan)
        }
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            shard, _n_sample, _base_url, dataset_name = sample_plan[i]
            try:
                results[i] = fut.result()
            except Exception as e:
                tqdm.write(f"\n  FAILED: {dataset_name}:{shard['key']} — {e}")
                results[i] = np.array([], dtype=np.dtype(dtype)).reshape(0, SEQ_LEN)

    pbar_tok.close()
    pbar_shard.close()
    pbar_net.close()

    tqdm.write("\nConcatenating...")
    valid = [r for r in results if r is not None and r.size > 0]
    seqs  = np.concatenate(valid, axis=0)

    tqdm.write("Shuffling sequences...")
    np.random.default_rng(SEED).shuffle(seqs)

    tqdm.write(f"Saving -> {OUTPUT_PATH}  ({seqs.nbytes / 1e9:.2f} GB)")
    np.save(OUTPUT_PATH, seqs)

    if args.export_hf:
        hf_output = Path(args.hf_output or default_hf_output_path(OUTPUT_PATH))
        export_npy_to_hf_dataset(
            OUTPUT_PATH,
            hf_output,
            chunk_size=args.hf_chunk_size,
        )

    elapsed = time.time() - t0
    tqdm.write(f"\n{'─'*62}")
    tqdm.write("  Done!")
    tqdm.write(f"  Output shape : {seqs.shape}  (seqs x seq_len)")
    tqdm.write(f"  Total tokens : {seqs.size:,}")
    tqdm.write(f"  Downloaded   : {byte_ctr.value / 1e6:.1f} MB")
    tqdm.write(f"  Time elapsed : {elapsed:.1f}s")
    if elapsed > 0:
        tqdm.write(f"  Avg speed    : {seqs.size / elapsed / 1e6:.2f}M tokens/sec")
    tqdm.write(f"  Output file  : {OUTPUT_PATH}")
    if args.export_hf:
        hf_output = Path(args.hf_output or default_hf_output_path(OUTPUT_PATH))
        tqdm.write(f"  HF dataset   : {hf_output}")
    tqdm.write(f"{'─'*62}")

if __name__ == "__main__":
    main()
