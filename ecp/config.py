"""Load config.yaml and resolve paths relative to the repository root."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text())
    root = path.resolve().parent
    cfg["_root"] = root
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = (root / v).resolve()
    return cfg
