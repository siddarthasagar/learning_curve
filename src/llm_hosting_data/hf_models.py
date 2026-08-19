"""Discover HuggingFace model repos and compute their on-disk size in GiB.

Uses the official ``huggingface_hub`` client rather than the undocumented
``/api/models/<repo>/treesize/main`` endpoint, so a renamed or deleted repo
raises a typed error instead of being silently miscounted, and so discovery
can walk a Collection or an org's full repo list instead of a hand-maintained
URL list.

``get_model_size()`` caches its result to ``dump/cache/hf_models/`` (see
``paths.py``), one small JSON file per ``repo_id``+``revision`` — 2026-08-19,
user feedback: a specific checkpoint's file listing doesn't change once
uploaded, so re-fetching it on every run wastes an API call on data that's
already settled, the same "cache what's actually static" reasoning as the
AWS offer files and Vantage's catalog. One caveat: this project's default
``revision="main"`` is a mutable branch pointer, not a pinned commit — this
session already observed HF Collections gain members between two runs
minutes apart, so a lab re-uploading files under ``main`` is possible, if
rare. Discovery (``list_collection_model_ids``/``list_org_model_ids``) is
deliberately **not** cached, for the opposite reason — membership is exactly
the part that does change run to run. ``force_refresh`` bypasses the cache
for one call; unlike Vantage there's no cheap live signal to gate an
automatic refresh on, so it's a manual escape hatch, same as AWS's
``--refresh``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from huggingface_hub import HfApi
from huggingface_hub.utils import (
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from llm_hosting_data.paths import CACHE_DIR as _SHARED_CACHE_DIR

if TYPE_CHECKING:
    from pathlib import Path

_BYTES_PER_GIB = 1024**3
CACHE_DIR = _SHARED_CACHE_DIR / "hf_models"


class ModelNotFoundError(RuntimeError):
    """Raised when a HuggingFace repo or revision does not exist."""


class ModelAccessError(RuntimeError):
    """Raised when a HuggingFace repo exists but requires an authorized token."""


@dataclass(frozen=True)
class ModelSize:
    """Total on-disk size of a HuggingFace model repo at one revision.

    No fetch timestamp here: the enclosing snapshot filename already carries
    that, and a per-record timestamp would make every re-run's snapshot
    differ byte-for-byte even when nothing about the model actually changed
    — defeating both the diff (everything looks "changed") and the
    unchanged-run dedup in ``snapshot.save_snapshot``.
    """

    repo_id: str
    revision: str
    total_bytes: int
    file_count: int

    @property
    def total_gib(self) -> float:
        """Size in binary GiB (1024-based) — HuggingFace's web UI shows decimal GB instead."""
        return self.total_bytes / _BYTES_PER_GIB


def _cache_path(repo_id: str, revision: str) -> Path:
    safe_repo = repo_id.replace("/", "__")
    safe_revision = revision.replace("/", "__")
    return CACHE_DIR / f"{safe_repo}@{safe_revision}.json"


def get_model_size(
    repo_id: str,
    *,
    revision: str = "main",
    token: str | None = None,
    force_refresh: bool = False,
) -> ModelSize:
    """Fetch the total file size of a HuggingFace model repo at ``revision``.

    Sums the ``size`` of every file in the repo tree via ``model_info(...,
    files_metadata=True)``. Raises :class:`ModelNotFoundError` for a missing
    repo/revision and :class:`ModelAccessError` for a gated repo, instead of
    treating either as "confirmed absent" — neither error is cached, only a
    real result is.

    Cached to disk per ``repo_id``+``revision`` (see module docstring); pass
    ``force_refresh=True`` to bypass a cached value.
    """
    cache_path = _cache_path(repo_id, revision)
    if cache_path.exists() and not force_refresh:
        return ModelSize(**json.loads(cache_path.read_text()))

    api = HfApi(token=token)
    try:
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
    except RepositoryNotFoundError as exc:
        msg = f"Model repo does not exist: {repo_id}"
        raise ModelNotFoundError(msg) from exc
    except RevisionNotFoundError as exc:
        msg = f"Revision {revision!r} does not exist for {repo_id}"
        raise ModelNotFoundError(msg) from exc
    except GatedRepoError as exc:
        msg = f"Model repo is gated and requires an authorized token: {repo_id}"
        raise ModelAccessError(msg) from exc

    siblings = info.siblings or []
    sizes = [sibling.size for sibling in siblings if sibling.size is not None]
    size = ModelSize(
        repo_id=repo_id,
        revision=revision,
        total_bytes=sum(sizes),
        file_count=len(sizes),
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(asdict(size)))
    return size


def list_collection_model_ids(
    collection_slug: str,
    *,
    token: str | None = None,
) -> list[str]:
    """List the model repo IDs in a HuggingFace Collection, e.g. ``"nvidia/nemotron-v3"``."""
    api = HfApi(token=token)
    collection = api.get_collection(collection_slug)
    return [item.item_id for item in collection.items if item.item_type == "model"]


def list_org_model_ids(author: str, *, token: str | None = None) -> list[str]:
    """List every model repo ID under a HuggingFace org/user namespace."""
    api = HfApi(token=token)
    return [model.id for model in api.list_models(author=author)]
