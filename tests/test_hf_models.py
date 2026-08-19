"""Tests for HuggingFace model sizing.

These stay offline: HTTP calls go through a stubbed ``HfApi``, since a live
integration check (repo IDs from ``research_topics/llm_hosting/``, run
against the real Hub) was already used to validate the GiB numbers by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pytest
from huggingface_hub.utils import RepositoryNotFoundError

from llm_hosting_data import hf_models
from llm_hosting_data.hf_models import ModelNotFoundError, ModelSize, get_model_size

if TYPE_CHECKING:
    from pathlib import Path

_BYTES_PER_GIB = 1024**3


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test's model-size cache in its own tmp dir, not dump/cache/."""
    monkeypatch.setattr(hf_models, "CACHE_DIR", tmp_path)


def test_total_gib_uses_binary_not_decimal_conversion() -> None:
    """A checkpoint reported as decimal GB by HF's UI must not leak into GiB math."""
    size = ModelSize(
        repo_id="org/model",
        revision="main",
        total_bytes=20 * _BYTES_PER_GIB,
        file_count=3,
    )

    assert size.total_gib == pytest.approx(20.0)
    # 1000**3-based (decimal GB) math would read ~21.47 here -- the exact
    # ~7.4% error the research docs flagged as a past mistake.
    assert size.total_gib != pytest.approx(size.total_bytes / 1000**3)


def test_get_model_size_raises_not_found_instead_of_inferring_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_not_found(*_args: object, **_kwargs: object) -> None:
        msg = "no such repo"
        raise RepositoryNotFoundError(msg)

    monkeypatch.setattr("llm_hosting_data.hf_models.HfApi.model_info", _raise_not_found)

    with pytest.raises(ModelNotFoundError):
        get_model_size("org/does-not-exist")


def test_get_model_size_sums_sibling_file_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Sibling:
        def __init__(self, size: int | None) -> None:
            self.size = size

    class _Info:
        siblings: ClassVar = [_Sibling(1_000), _Sibling(2_000), _Sibling(None)]

    monkeypatch.setattr(
        "llm_hosting_data.hf_models.HfApi.model_info",
        lambda *_a, **_k: _Info(),
    )

    size = get_model_size("org/model")

    assert size.total_bytes == 3_000
    assert size.file_count == 2


def test_get_model_size_reuses_cache_on_second_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Sibling:
        def __init__(self, size: int | None) -> None:
            self.size = size

    class _Info:
        siblings: ClassVar = [_Sibling(5_000)]

    calls = []
    monkeypatch.setattr(
        "llm_hosting_data.hf_models.HfApi.model_info",
        lambda *_a, **_k: calls.append(1) or _Info(),
    )

    first = get_model_size("org/model")
    second = get_model_size("org/model")

    assert len(calls) == 1  # second call hit the cache, not the API
    assert second == first


def test_get_model_size_force_refresh_bypasses_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Sibling:
        def __init__(self, size: int | None) -> None:
            self.size = size

    class _Info:
        def __init__(self, size: int) -> None:
            self.siblings = [_Sibling(size)]

    sizes = iter([1_000, 2_000])
    monkeypatch.setattr(
        "llm_hosting_data.hf_models.HfApi.model_info",
        lambda *_a, **_k: _Info(next(sizes)),
    )

    first = get_model_size("org/model")
    second = get_model_size("org/model", force_refresh=True)

    assert first.total_bytes == 1_000
    assert second.total_bytes == 2_000  # re-fetched, not served from cache
