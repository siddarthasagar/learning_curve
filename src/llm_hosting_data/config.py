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


@dataclass(frozen=True)
class BenchmarkComparisonTargets:
    """Baseline model slugs loaded from a YAML config file.

    Candidates are deliberately not configured here — they're auto-derived
    from whichever ``config/hf_models.yaml`` repos have real OpenRouter
    coverage (see ``model_instance_fit.discover_candidate_slugs()``), so
    there's no second, driftable list of model families to keep in sync as
    collections change. Only the baseline needs a manual slug: it's
    typically closed-source and never appears in ``hf_models.yaml`` at all.
    """

    baseline: list[str]


def load_benchmark_targets(config_path: Path) -> BenchmarkComparisonTargets:
    """Load the ``baseline`` model-slug list from a YAML config file.

    Optional, defaults to an empty list. Expected shape::

        baseline:
          - anthropic/claude-sonnet-5
    """
    raw = _load_mapping(config_path)
    return BenchmarkComparisonTargets(baseline=list(raw.get("baseline") or []))


def load_gpu_hardware_specs(config_path: Path) -> dict[str, dict[str, Any]]:
    """Load the per-family GPU hardware spec table from a YAML file.

    Kept as external data rather than a Python constant (2026-08-19, user
    request, same reasoning as ``load_kernel_support_matrix``) — see
    ``config/gpu_hardware_specs.yaml`` for the full field definitions and
    sourcing. Expected shape::

        p4de:
          memory_gb_per_gpu: 80.0
          memory_bandwidth_gbps: 2039.0
          nvlink_generation: NVLink3
          nvlink_bandwidth_gbps: 600.0
    """
    raw = _load_mapping(config_path)
    for family, spec in raw.items():
        if not isinstance(spec, dict):
            msg = (
                f"GPU hardware spec entry {family!r} must be a mapping, "
                f"got {type(spec).__name__}: {config_path}"
            )
            raise ConfigError(msg)
    return raw


def load_kernel_support_matrix(config_path: Path) -> dict[str, dict[str, str]]:
    """Load the SM -> format -> support-level kernel matrix from a YAML file.

    Kept as external data rather than a Python constant (2026-08-19, user
    request): unlike fixed GPU hardware specs, kernel/software support for a
    format on an architecture moves as vLLM/CUTLASS etc. mature, so this is
    the file to update, not code — see ``config/kernel_support_matrix.yaml``
    for the full doc-04-sourced table and level definitions. Expected
    shape::

        SM90:
          bf16: native
          fp8_dense: native
          nvfp4_dense: fallback
    """
    raw = _load_mapping(config_path)
    for sm, formats in raw.items():
        if not isinstance(formats, dict):
            msg = (
                f"Kernel support matrix entry {sm!r} must be a mapping of "
                f"format -> support level, got {type(formats).__name__}: {config_path}"
            )
            raise ConfigError(msg)
    return raw
