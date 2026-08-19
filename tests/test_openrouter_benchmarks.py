"""Tests for the OpenRouter benchmarks client.

Stays offline: HTTP calls go through a stubbed response object. A live check
against the real endpoint (with a real token) already confirmed the schema
this module parses, and that Claude Sonnet 5 shows up under the
"openrouter"-native ``tau_bench_verified_airline`` benchmark type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import requests

from llm_hosting_data import openrouter_benchmarks as orb

if TYPE_CHECKING:
    from pathlib import Path


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            msg = f"status {self.status_code}"
            raise requests.HTTPError(msg)


def test_resolve_token_raises_without_token_or_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_TOKEN", raising=False)
    with pytest.raises(orb.OpenRouterAuthError):
        orb._resolve_token(None)  # noqa: SLF001


def test_fetch_benchmarks_parses_and_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orb, "CACHE_DIR", tmp_path)
    payload = {
        "data": [
            {
                "source": "openrouter",
                "model_permaslug": "anthropic/claude-sonnet-5-20260630",
                "display_name": "Claude Sonnet 5",
                "benchmark_type": "tau_bench_verified_airline",
                "accuracy": 0.771,
            },
        ],
        "meta": {"model_count": 1},
    }
    monkeypatch.setattr(
        orb.requests,
        "get",
        lambda *_a, **_k: _FakeResponse(200, payload),
    )

    entries = orb.fetch_benchmarks(token="fake-token")

    assert len(entries) == 1
    assert entries[0].source == "openrouter"
    assert entries[0].fields["accuracy"] == pytest.approx(0.771)
    assert (tmp_path / "benchmarks.json").exists()

    def _fail(*_a: object, **_k: object) -> None:
        msg = "should not hit the network on a cached read"
        raise AssertionError(msg)

    monkeypatch.setattr(orb.requests, "get", _fail)
    cached = orb.fetch_benchmarks(token="fake-token")
    assert cached == entries


def test_fetch_benchmarks_raises_auth_error_on_401(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orb, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(orb.requests, "get", lambda *_a, **_k: _FakeResponse(401, {}))

    with pytest.raises(orb.OpenRouterAuthError):
        orb.fetch_benchmarks(token="bad-token")


def test_filter_by_model_matches_stable_prefix_case_insensitive() -> None:
    entries = [
        orb.BenchmarkEntry("openrouter", "Anthropic/Claude-Sonnet-5-20260630", "x", {}),
        orb.BenchmarkEntry("openrouter", "qwen/qwen3.5-397b-a17b-20260216", "y", {}),
    ]

    matched = orb.filter_by_model(entries, ["anthropic/claude-sonnet-5"])

    assert len(matched) == 1
    assert matched[0].model_permaslug.startswith("Anthropic")


def test_filter_by_type_combines_source_and_benchmark_type() -> None:
    entries = [
        orb.BenchmarkEntry(
            "openrouter",
            "a/a",
            "a",
            {"benchmark_type": "tau_bench_verified_airline"},
        ),
        orb.BenchmarkEntry(
            "openrouter",
            "a/a",
            "a",
            {"benchmark_type": "gpqa_diamond"},
        ),
        orb.BenchmarkEntry("artificial-analysis", "a/a", "a", {"agentic_index": 50}),
    ]

    matched = orb.filter_by_type(
        entries,
        source="openrouter",
        benchmark_type="gpqa_diamond",
    )

    assert len(matched) == 1
    assert matched[0].fields["benchmark_type"] == "gpqa_diamond"
