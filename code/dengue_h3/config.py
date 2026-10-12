from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = path.parent.parent
    return root, config


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
