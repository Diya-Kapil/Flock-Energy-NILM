"""Reading and writing helpers shared by the pipeline stages."""
from __future__ import annotations

import hashlib
import io
import json
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import py7zr
from py7zr.io import BytesIOFactory, Py7zIO, WriterFactory

from src.config import RAW_COLUMNS, SENSORS

MAX_MEMBER_BYTES = 2_000_000_000


class SchemaError(ValueError):
    """Raised when a CSV does not have the documented REFIT columns."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw_part(source: Path | bytes, part: int) -> pd.DataFrame:
    """Read one raw CSV, keeping file order, original timestamps and all readings."""
    buffer = source if isinstance(source, Path) else io.BytesIO(source)
    frame = pd.read_csv(buffer, engine="pyarrow")
    if list(frame.columns) != RAW_COLUMNS:
        raise SchemaError(f"Part {part} columns are {list(frame.columns)}, expected {RAW_COLUMNS}")
    frame["Time"] = frame["Time"].astype("datetime64[s]")
    frame["Unix"] = frame["Unix"].astype("int64")
    for sensor in SENSORS:
        frame[sensor] = frame[sensor].astype("float64")
    frame["part"] = np.int8(part)
    frame["file_row"] = np.arange(len(frame), dtype="int64")
    return frame


def read_official_clean(source: Path | bytes) -> pd.DataFrame:
    """Read one official cleaned CSV without changing any value."""
    buffer = source if isinstance(source, Path) else io.BytesIO(source)
    frame = pd.read_csv(buffer, engine="pyarrow")
    expected = [*RAW_COLUMNS, "Issues"]
    if list(frame.columns) != expected:
        raise SchemaError(f"Cleaned columns are {list(frame.columns)}, expected {expected}")
    frame["Time"] = frame["Time"].astype("datetime64[s]")
    for sensor in SENSORS:
        frame[sensor] = frame[sensor].astype("float64")
    return frame


def read_archive_member(archive: Path, name: str) -> bytes:
    """Decompress one archive member into memory. The archive is opened read-only."""
    factory = BytesIOFactory(limit=MAX_MEMBER_BYTES)
    with py7zr.SevenZipFile(archive, "r") as handle:
        handle.extract(targets=[name], factory=factory)
    member = factory.get(name)
    member.seek(0)
    return member.read()


class _MemoryWriter(Py7zIO):
    """Collect one archive member in memory, then hand it over when py7zr closes it."""

    def __init__(self, name: str, on_close: Callable[[str, bytes], None]):
        self.name, self.on_close, self.buffer = name, on_close, io.BytesIO()

    def write(self, s: bytes | bytearray) -> int:
        return self.buffer.write(s)

    def read(self, size: int | None = None) -> bytes:
        return self.buffer.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.buffer.seek(offset, whence)

    def flush(self) -> None:
        pass

    def size(self) -> int:
        return self.buffer.getbuffer().nbytes

    def close(self) -> None:
        self.on_close(self.name, self.buffer.getvalue())
        self.buffer = io.BytesIO()


class _MemoryFactory(WriterFactory):
    def __init__(self, on_close: Callable[[str, bytes], None]):
        self.on_close = on_close

    def create(self, filename: str) -> Py7zIO:
        return _MemoryWriter(filename, self.on_close)


def stream_members(archive: Path, handler: Callable[[str, bytes], None], targets: list[str] | None = None) -> None:
    """Call handler(name, data) per member, in memory and one at a time, in one archive pass."""
    lock = threading.Lock()

    def guarded(name: str, data: bytes) -> None:
        with lock:
            handler(name, data)

    with py7zr.SevenZipFile(archive, "r") as handle:
        if targets is None:
            handle.extractall(factory=_MemoryFactory(guarded))
        else:
            handle.extract(targets=targets, factory=_MemoryFactory(guarded))


def to_jsonable(value):
    """Convert NumPy and pandas values so reports can be written as JSON."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if value is pd.NaT or value is pd.NA:
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


