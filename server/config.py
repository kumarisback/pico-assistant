"""Configuration loading. config.yaml is the source of truth; env vars override secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str) -> Path:
    """Project-relative paths become absolute; absolute paths pass through."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        cfg_path = path or ROOT / "config.yaml"
        with open(cfg_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        token = os.environ.get("ASSISTANT_TOKEN")
        if token:
            raw["server"]["auth_token"] = token

        base_url = os.environ.get("LLAMA_BASE_URL")
        if base_url:
            raw["llm"]["base_url"] = base_url

        return cls(raw=raw)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    # Paths are resolved here rather than at each call site so that the rest of
    # the code never has to care about the project root.
    def path(self, *keys: str) -> Path:
        node: Any = self.raw
        for k in keys:
            node = node[k]
        return _resolve(node)

    @property
    def frame_bytes(self) -> int:
        """Bytes in one PCM16 mono audio frame."""
        a = self.raw["audio"]
        return int(a["sample_rate"] * a["frame_ms"] / 1000) * 2

    @property
    def frame_samples(self) -> int:
        a = self.raw["audio"]
        return int(a["sample_rate"] * a["frame_ms"] / 1000)
