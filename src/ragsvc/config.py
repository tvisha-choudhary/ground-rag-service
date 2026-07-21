"""Configuration loading.

All tunables live in a YAML file. Any value can be overridden with an
environment variable named RAGSVC_<SECTION>_<KEY>, for example
RAGSVC_SERVING_PORT=9090. Nothing is hardcoded in the model or serving code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(os.environ.get("RAGSVC_CONFIG", "config/default.yaml"))


class ConfigError(ValueError):
    """Raised when configuration is missing or malformed."""


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise ConfigError(f"config file not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as fh:
            try:
                raw = yaml.safe_load(fh)
            except yaml.YAMLError as exc:
                raise ConfigError(f"invalid YAML in {cfg_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
        return cls(raw=raw)

    def get(self, section: str, key: str, cast: type = str) -> Any:
        env_name = f"RAGSVC_{section.upper()}_{key.upper()}"
        if env_name in os.environ:
            return cast(os.environ[env_name])
        try:
            value = self.raw[section][key]
        except KeyError as exc:
            raise ConfigError(f"missing config value: {section}.{key}") from exc
        return cast(value)

    def section(self, name: str) -> dict[str, Any]:
        if name not in self.raw:
            raise ConfigError(f"missing config section: {name}")
        return dict(self.raw[name])
