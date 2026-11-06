"""File I/O and path resolution utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to REPO_ROOT if it is not already absolute."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if needed and return the Path."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 YAML configuration file."""
    p = resolve_path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """Write a dictionary to a UTF-8 YAML file."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 JSON file."""
    p = resolve_path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(data: dict[str, Any], path: str | Path, indent: int = 2) -> Path:
    """Write a dictionary to a UTF-8 JSON file."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
    return p
