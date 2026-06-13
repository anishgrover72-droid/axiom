"""Streaming JSONL I/O for pydantic records. Never load a whole corpus into RAM."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def ensure_dir(path: str | Path) -> Path:
    """Create `path` (a directory) and parents; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_records(path: str | Path, records: Iterable[BaseModel]) -> int:
    """Stream pydantic records to a JSONL file. Returns the count written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")
            n += 1
    return n


def read_records(path: str | Path, model: type[M]) -> Iterator[M]:
    """Stream and validate JSONL rows into `model` instances."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield model.model_validate_json(line)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream raw JSON objects from a JSONL file."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
