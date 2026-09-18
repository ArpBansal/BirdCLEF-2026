from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve every entry in the ``paths`` section from project root."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key, value in config.get("paths", {}).items():
        if key.endswith("glob"):
            continue
        resolved = Path(value)
        config["paths"][key] = resolved if resolved.is_absolute() else PROJECT_ROOT / resolved
    return config
