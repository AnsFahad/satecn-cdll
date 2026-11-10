"""Config loading with project-root-relative paths."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, val in (overrides or {}).items():  # dotted keys, e.g. {"objective.beta": 0.5}
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = val
    return cfg


def clone(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
