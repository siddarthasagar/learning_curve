"""Fetch benchmark/agentic-capability scores from OpenRouter's benchmarks API.

Aggregates three sources with very different schemas — Artificial Analysis
composite indices, Design Arena creative/UI ELO, and OpenRouter's own
independently-run evals (``gpqa_diamond``, ``tau_bench_verified_airline``).
Only the last of those directly measures agentic tool-use the way
``research_topics/llm_hosting/05-sonnet5-replacement-shortlist.md`` needs —
and, checked live, it's the one place Claude Sonnet 5's tool-use performance
shows up at all, since Anthropic never published it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

CACHE_DIR = Path.home() / ".cache" / "llm_hosting_data" / "openrouter"

_BENCHMARKS_URL = "https://openrouter.ai/api/v1/benchmarks"
_REQUEST_TIMEOUT_SECONDS = 30
_TOKEN_ENV_VAR = "OPENROUTER_API_TOKEN"  # noqa: S105 -- env var name, not a secret
_HTTP_UNAUTHORIZED = 401


class OpenRouterAuthError(RuntimeError):
    """Raised when no API token is available, or OpenRouter rejects it."""


class OpenRouterFetchError(RuntimeError):
    """Raised when the benchmarks API cannot be reached or returns an error."""


@dataclass(frozen=True)
class BenchmarkEntry:
    """One row from OpenRouter's benchmarks API.

    Schema varies by ``source`` (artificial-analysis / design-arena /
    openrouter), so beyond the three fields every source shares, the rest is
    kept as a raw dict rather than forced into one over-general dataclass.
    """

    source: str
    model_permaslug: str
    display_name: str
    fields: dict[str, Any]


def _resolve_token(token: str | None) -> str:
    resolved = token or os.environ.get(_TOKEN_ENV_VAR)
    if not resolved:
        msg = f"No OpenRouter API token given and {_TOKEN_ENV_VAR} is not set."
        raise OpenRouterAuthError(msg)
    return resolved


def _cache_path() -> Path:
    return CACHE_DIR / "benchmarks.json"


def fetch_benchmarks(
    *,
    token: str | None = None,
    force_refresh: bool = False,
) -> list[BenchmarkEntry]:
    """Fetch (and cache) the full OpenRouter benchmarks dataset — all sources, all models.

    Cached locally since the API is rate-limited (30 requests/minute, 500/day
    per OpenRouter's docs) — pass ``force_refresh=True`` to bypass. Raises
    :class:`OpenRouterAuthError` for a missing/rejected token and
    :class:`OpenRouterFetchError` for any other request failure.
    """
    destination = _cache_path()
    if destination.exists() and not force_refresh:
        payload = json.loads(destination.read_text())
    else:
        resolved_token = _resolve_token(token)
        try:
            response = requests.get(
                _BENCHMARKS_URL,
                headers={"Authorization": f"Bearer {resolved_token}"},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            msg = f"Failed to reach {_BENCHMARKS_URL}"
            raise OpenRouterFetchError(msg) from exc

        if response.status_code == _HTTP_UNAUTHORIZED:
            msg = "OpenRouter rejected the API token (401)."
            raise OpenRouterAuthError(msg)
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            msg = f"OpenRouter benchmarks request failed: HTTP {response.status_code}"
            raise OpenRouterFetchError(msg) from exc

        payload = response.json()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload))

    return [
        BenchmarkEntry(
            source=item["source"],
            model_permaslug=item["model_permaslug"],
            display_name=item.get("display_name", item["model_permaslug"]),
            fields={
                key: value
                for key, value in item.items()
                if key not in ("source", "model_permaslug", "display_name")
            },
        )
        for item in payload["data"]
    ]


def filter_by_model(
    entries: list[BenchmarkEntry],
    model_slugs: list[str],
) -> list[BenchmarkEntry]:
    """Keep only entries whose ``model_permaslug`` contains one of ``model_slugs``.

    Matching is case-insensitive substring, not exact: OpenRouter's slugs
    carry a release-date suffix (e.g. ``...-20260630``) that changes as new
    snapshots get evaluated, so matching the stable prefix keeps resolving
    across re-dated entries.
    """
    lowered = [slug.lower() for slug in model_slugs]
    return [
        entry
        for entry in entries
        if any(slug in entry.model_permaslug.lower() for slug in lowered)
    ]


def filter_by_type(
    entries: list[BenchmarkEntry],
    *,
    source: str | None = None,
    benchmark_type: str | None = None,
) -> list[BenchmarkEntry]:
    """Keep only entries matching ``source`` and/or ``benchmark_type``.

    ``benchmark_type`` (e.g. ``"tau_bench_verified_airline"``) is only
    present on ``source == "openrouter"`` rows; Artificial Analysis and
    Design Arena rows carry differently-shaped ``fields`` instead.
    """
    return [
        entry
        for entry in entries
        if (source is None or entry.source == source)
        and (
            benchmark_type is None
            or entry.fields.get("benchmark_type") == benchmark_type
        )
    ]
