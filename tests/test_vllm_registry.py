"""Tests for the vLLM quantization-registry fetcher.

Stays offline: HTTP calls go through a stubbed response object, same
pattern as test_openrouter_benchmarks.py. A live check (2026-08-19) already
confirmed the real GitHub Contents API response shape and that the current
registry differs from documents/fetch_info.md's own stale example output.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import pytest
import requests
import yaml

from llm_hosting_data import vllm_registry as reg

if TYPE_CHECKING:
    from pathlib import Path

_SAMPLE_SOURCE = """
from typing import Literal

QuantizationMethods = Literal[
    "awq",
    "fp8",
    "gptq_marlin",
    # Below are online quant shorthand names.
    "fp8_per_tensor",
]
QUANTIZATION_METHODS: list[str] = []

DEPRECATED_QUANTIZATION_METHODS = [
    "fbgemm_fp8",
]
"""


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


def _fake_contents_response(source: str) -> _FakeResponse:
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return _FakeResponse(200, {"content": encoded})


def test_fetch_quant_registry_parses_methods_and_deprecated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reg.requests,
        "get",
        lambda *_a, **_k: _fake_contents_response(_SAMPLE_SOURCE),
    )

    snapshot = reg.fetch_quant_registry(ref="main")

    assert snapshot.ref == "main"
    assert snapshot.methods == ["awq", "fp8", "gptq_marlin", "fp8_per_tensor"]
    assert snapshot.deprecated_methods == ["fbgemm_fp8"]


def test_fetch_quant_registry_wraps_request_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*_a: object, **_k: object) -> None:
        msg = "network unreachable"
        raise requests.ConnectionError(msg)

    monkeypatch.setattr(reg.requests, "get", _raise)

    with pytest.raises(reg.VllmRegistryFetchError):
        reg.fetch_quant_registry()


def test_fetch_quant_registry_raises_when_content_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reg.requests, "get", lambda *_a, **_k: _FakeResponse(200, {}))

    with pytest.raises(reg.VllmRegistryFetchError):
        reg.fetch_quant_registry()


def test_fetch_quant_registry_raises_when_literal_block_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reg.requests,
        "get",
        lambda *_a, **_k: _fake_contents_response("# no registry here"),
    )

    with pytest.raises(reg.VllmRegistryFetchError):
        reg.fetch_quant_registry()


def test_write_registry_yaml_round_trips(tmp_path: Path) -> None:
    snapshot = reg.QuantRegistrySnapshot(
        ref="main",
        fetched_at="2026-08-19T00:00:00+00:00",
        methods=["awq", "fp8"],
        deprecated_methods=["fbgemm_fp8"],
    )
    out_path = tmp_path / "vllm_quant_registry.yaml"

    reg.write_registry_yaml(snapshot, out_path)

    written = yaml.safe_load(out_path.read_text())
    assert written["ref"] == "main"
    assert written["methods"] == ["awq", "fp8"]
    assert written["deprecated_methods"] == ["fbgemm_fp8"]
