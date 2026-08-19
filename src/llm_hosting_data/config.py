"""Load YAML config files listing AWS instances, HuggingFace repos, or benchmark models to track.

Lets a refresh run be driven by a checked-in list instead of typing instance
types, repo IDs, or model slugs on the command line every time (see
``config/aws_instances.yaml``, ``config/hf_models.yaml``, and
``config/benchmark_models.yaml`` for the seed lists, pulled from
``research_topics/llm_hosting/``'s existing research).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


class ConfigError(RuntimeError):
    """Raised when a config file is missing, unreadable, or malformed."""


@dataclass(frozen=True)
class HfTargets:
    """Repo IDs and discovery targets loaded from a YAML config file."""

    repo_ids: list[str]
    collections: list[str]
    authors: list[str]


@dataclass(frozen=True)
class AwsTargets:
    """Instance families and individual instance types loaded from a YAML config file."""

    families: list[str]
    instance_types: list[str]


def _load_mapping(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise ConfigError(msg)

    raw: Any = yaml.safe_load(config_path.read_text())
    raw = raw or {}
    if not isinstance(raw, dict):
        msg = f"Config file must contain a YAML mapping, got {type(raw).__name__}: {config_path}"
        raise ConfigError(msg)
    return raw


def load_hf_targets(config_path: Path) -> HfTargets:
    """Load ``repo_ids``/``collections``/``authors`` lists from a YAML config file.

    Each key is optional and defaults to an empty list. Expected shape::

        repo_ids:
          - Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
        collections:
          - nvidia/nemotron-v3
        authors:
          - moonshotai
    """
    raw = _load_mapping(config_path)
    return HfTargets(
        repo_ids=list(raw.get("repo_ids") or []),
        collections=list(raw.get("collections") or []),
        authors=list(raw.get("authors") or []),
    )


def load_aws_targets(config_path: Path) -> AwsTargets:
    """Load ``families``/``instance_types`` lists from a YAML config file.

    Each key is optional and defaults to an empty list. A family (e.g.
    ``"g6e"``) expands to every size AWS currently ships in that family, so
    it doesn't need to enumerate individual instance types. Expected shape::

        families:
          - g6e
          - g7e
        instance_types:
          - p5.4xlarge
    """
    raw = _load_mapping(config_path)
    return AwsTargets(
        families=list(raw.get("families") or []),
        instance_types=list(raw.get("instance_types") or []),
    )


def load_benchmark_targets(config_path: Path) -> list[str]:
    """Load a flat list of model-slug substrings from a YAML config file.

    Expected shape::

        models:
          - anthropic/claude-sonnet-5
    """
    raw = _load_mapping(config_path)
    return list(raw.get("models") or [])
