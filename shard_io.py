"""Load pretokenized .npy shards for local evaluation."""
from __future__ import annotations

import io
import pathlib
import struct
from typing import Any

import numpy as np


def parse_npy_header(raw: bytes) -> tuple[int, dict[str, Any]]:
    buf = io.BytesIO(raw)
    if buf.read(6) != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack(
        "<H" if ver[0] == 1 else "<I",
        buf.read(2 if ver[0] == 1 else 4),
    )[0]
    header = eval(buf.read(hl).decode("latin1").strip())
    return buf.tell(), header


def _read_npy_mmap(path: pathlib.Path) -> tuple[np.memmap, dict[str, Any], int]:
    with path.open("rb") as f:
        header_bytes = f.read(256)
    data_offset, header = parse_npy_header(header_bytes)
    dtype = np.dtype(header["descr"])
    shape = tuple(header["shape"])
    arr = np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=shape)
    return arr, header, data_offset


def count_packed_sequences(path: pathlib.Path | str, *, seq_len: int = 2048) -> int:
    path = pathlib.Path(path)
    arr, header, _data_offset = _read_npy_mmap(path)
    shape = header["shape"]
    dtype = np.dtype(header["descr"])
    if len(shape) == 1:
        if dtype != np.dtype("<u4"):
            raise ValueError(f"expected 1D uint32 shard, got dtype={dtype}")
        return int(shape[0]) // seq_len
    if len(shape) == 2:
        return int(shape[0])
    raise ValueError(f"{path.name}: unexpected shard shape {shape}")


def load_token_shard(
    path: pathlib.Path | str,
    *,
    seq_len: int = 2048,
    n_sequences: int | None = None,
    seq_offset: int = 0,
) -> np.ndarray:
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    arr, header, _data_offset = _read_npy_mmap(path)
    shape = header["shape"]
    dtype = np.dtype(header["descr"])
    offset = max(0, int(seq_offset))

    if len(shape) == 1:
        if dtype != np.dtype("<u4"):
            raise ValueError(f"expected 1D uint32 shard, got dtype={dtype}")
        n_total = int(shape[0])
        n_seq = n_total // seq_len
        if n_seq == 0:
            raise ValueError(f"{path.name}: only {n_total} tokens, need at least {seq_len}")
        if offset >= n_seq:
            raise ValueError(f"{path.name}: seq_offset {offset} >= n_seq {n_seq}")
        available = n_seq - offset
        take = available if n_sequences is None else min(n_sequences, available)
        start_token = offset * seq_len
        flat = np.asarray(arr[start_token : start_token + take * seq_len], dtype=np.int64)
        return flat.reshape(take, seq_len)

    if len(shape) == 2:
        n_seq, actual_seq_len = int(shape[0]), int(shape[1])
        if actual_seq_len != seq_len:
            raise ValueError(
                f"{path.name}: shard seq_len={actual_seq_len}, expected {seq_len}"
            )
        if offset >= n_seq:
            raise ValueError(f"{path.name}: seq_offset {offset} >= n_seq {n_seq}")
        available = n_seq - offset
        take = available if n_sequences is None else min(n_sequences, available)
        return np.asarray(arr[offset : offset + take], dtype=np.int64)

    raise ValueError(f"{path.name}: unexpected shard shape {shape}")
