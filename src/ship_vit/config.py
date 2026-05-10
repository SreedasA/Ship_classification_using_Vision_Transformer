from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")
    return data


def resolve_path(root: str | Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return Path(root) / p

