"""Verify the downloaded inputs and extract only the raw House 1 files."""
from __future__ import annotations

import zlib
from pathlib import Path

import py7zr

from src.config import INPUT_SHA256, RAW_ARCHIVE, RAW_EXTRACT_DIR, RAW_PART_NAMES
from src.io_utils import sha256_file

MISSING_HELP = (
    "Download the official files listed in DATA.md into data/raw/archives and data/raw/docs, "
    "then run this stage again."
)


class InputError(RuntimeError):
    """Raised when a required input is missing or differs from the recorded checksum."""


def verify_inputs(expected: dict[Path, str] = INPUT_SHA256) -> dict[str, str]:
    """Stop with a clear message unless every input matches its recorded checksum."""
    verified = {}
    for path, digest in expected.items():
        if not path.exists():
            raise InputError(f"Missing input: {path}. {MISSING_HELP}")
        actual = sha256_file(path)
        if actual != digest:
            raise InputError(f"Checksum mismatch for {path}: expected {digest}, got {actual}.")
        verified[path.name] = actual
    return verified


def file_crc32(path: Path) -> int:
    crc = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            crc = zlib.crc32(chunk, crc)
    return crc


def extract_members(archive: Path, names: list[str], destination: Path) -> dict[str, dict]:
    """Extract only the named members and check each against the archive's CRC.

    Members that already exist with the right CRC are left alone, so reruns are safe.
    """
    destination.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, "r") as handle:
        listing = {item.filename: item for item in handle.list()}
    unknown = [name for name in names if name not in listing]
    if unknown:
        raise InputError(f"{archive.name} does not contain {unknown}.")

    pending = [name for name in names if not (
        (destination / name).exists() and file_crc32(destination / name) == listing[name].crc32)]
    if pending:
        with py7zr.SevenZipFile(archive, "r") as handle:
            handle.extract(path=destination, targets=pending)

    report = {}
    for name in names:
        path = destination / name
        crc = file_crc32(path)
        if crc != listing[name].crc32:
            raise InputError(f"{path} failed its CRC check after extraction.")
        report[name] = {"path": str(path), "bytes": path.stat().st_size, "crc32": crc,
                        "extracted_this_run": name in pending}
    return report


def run_acquire() -> dict:
    checksums = verify_inputs()
    extracted = extract_members(RAW_ARCHIVE, list(RAW_PART_NAMES.values()), RAW_EXTRACT_DIR)
    return {"input_sha256": checksums, "extracted": extracted}
