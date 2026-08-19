"""Tests for the Vantage-backed GPU hardware reference lookup.

Stays offline: the raw Vantage catalog is a small hand-built fixture file,
not a live download. A live check (2026-08-18) already confirmed the real
feed's schema and that its GPU_model/gpu_architectures values agree with
this project's g6e/g7e/p4d/p4de/p5/p6 research.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from llm_hosting_data import gpu_reference as gpu_ref

if TYPE_CHECKING:
    from pathlib import Path


def _raw_item(
    instance_type: str,
    *,
    gpu: int = 0,
    gpu_model: str | None = None,
    architectures: list[str] | None = None,
    compute_capability: float | None = None,
) -> dict[str, Any]:
    return {
        "instance_type": instance_type,
        "GPU": gpu,
        "GPU_model": gpu_model,
        "gpu_architectures": architectures,
        "compute_capability": compute_capability,
    }


def test_extract_gpu_catalog_skips_non_gpu_instances(tmp_path: Path) -> None:
    raw_catalog = tmp_path / "instances.json"
    raw_catalog.write_text(
        json.dumps(
            [
                _raw_item("m5.xlarge", gpu=0),
                _raw_item(
                    "g6e.xlarge",
                    gpu=1,
                    gpu_model="NVIDIA L40S",
                    architectures=["Ada Lovelace"],
                    compute_capability=8.9,
                ),
            ],
        ),
    )

    catalog = gpu_ref._extract_gpu_catalog(raw_catalog)  # noqa: SLF001

    assert list(catalog) == ["g6e.xlarge"]
    ref = catalog["g6e.xlarge"]
    assert ref.gpu_model == "NVIDIA L40S"
    assert ref.architecture == "Ada Lovelace"
    assert ref.compute_capability == "SM89"


def test_extract_gpu_catalog_keeps_gpu_model_when_architecture_missing(
    tmp_path: Path,
) -> None:
    """Test that a missing field doesn't drop the whole GPU record.

    Vantage's own data has gaps per instance type (p5.4xlarge: GPU_model
    present, gpu_architectures key entirely absent -- confirmed live
    2026-08-19).
    """
    raw_catalog = tmp_path / "instances.json"
    raw_catalog.write_text(
        json.dumps(
            [
                _raw_item(
                    "p5.4xlarge",
                    gpu=1,
                    gpu_model="NVIDIA H100",
                    architectures=None,
                ),
            ],
        ),
    )

    catalog = gpu_ref._extract_gpu_catalog(raw_catalog)  # noqa: SLF001

    assert catalog["p5.4xlarge"].gpu_model == "NVIDIA H100"
    assert catalog["p5.4xlarge"].architecture is None


def test_compute_capability_label_handles_non_numeric() -> None:
    assert gpu_ref._compute_capability_label(12) == "SM120"  # noqa: SLF001
    assert gpu_ref._compute_capability_label(None) is None  # noqa: SLF001
    assert gpu_ref._compute_capability_label("unknown") is None  # noqa: SLF001


def test_get_gpu_catalog_uses_extracted_cache_on_second_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_ref, "CACHE_DIR", tmp_path)
    raw_catalog = tmp_path / "raw-source.json"
    raw_catalog.write_text(
        json.dumps(
            [
                _raw_item(
                    "g7e.2xlarge",
                    gpu=1,
                    gpu_model="NVIDIA RTX PRO 6000 Blackwell",
                    architectures=["Blackwell"],
                    compute_capability=12,
                ),
            ],
        ),
    )
    monkeypatch.setattr(gpu_ref, "_download_raw_catalog", lambda *_a, **_k: raw_catalog)

    first = gpu_ref.get_gpu_catalog()
    assert first["g7e.2xlarge"].gpu_model == "NVIDIA RTX PRO 6000 Blackwell"

    def _fail(*_a: object, **_k: object) -> None:
        msg = "should read the extracted cache, not re-download"
        raise AssertionError(msg)

    monkeypatch.setattr(gpu_ref, "_download_raw_catalog", _fail)
    second = gpu_ref.get_gpu_catalog()
    assert second == first


def test_get_gpu_catalog_wraps_download_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_ref, "CACHE_DIR", tmp_path)

    def _raise(*_a: object, **_k: object) -> None:
        msg = "network unreachable"
        raise gpu_ref.GpuReferenceFetchError(msg)

    monkeypatch.setattr(gpu_ref, "_download_raw_catalog", _raise)

    with pytest.raises(gpu_ref.GpuReferenceFetchError):
        gpu_ref.get_gpu_catalog()


def test_lookup_gpu_reference_strips_sagemaker_ml_prefix() -> None:
    catalog = {
        "g6e.xlarge": gpu_ref.GpuReference(
            "g6e.xlarge",
            "NVIDIA L40S",
            "Ada Lovelace",
            "SM89",
        ),
    }

    assert (
        gpu_ref.lookup_gpu_reference("ml.g6e.xlarge", catalog) is catalog["g6e.xlarge"]
    )
    assert gpu_ref.lookup_gpu_reference("g6e.xlarge", catalog) is catalog["g6e.xlarge"]
    assert gpu_ref.lookup_gpu_reference("m5.xlarge", catalog) is None
