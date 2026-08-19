"""Discover HuggingFace model repos and compute their on-disk size in GiB.

Uses the official ``huggingface_hub`` client rather than the undocumented
``/api/models/<repo>/treesize/main`` endpoint, so a renamed or deleted repo
raises a typed error instead of being silently miscounted, and so discovery
can walk a Collection or an org's full repo list instead of a hand-maintained
URL list.
"""

from __future__ import annotations

from dataclasses import dataclass

from huggingface_hub import HfApi
from huggingface_hub.utils import (
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

_BYTES_PER_GIB = 1024**3


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


def get_model_size(
    repo_id: str,
    *,
    revision: str = "main",
    token: str | None = None,
) -> ModelSize:
    """Fetch the total file size of a HuggingFace model repo at ``revision``.

    Sums the ``size`` of every file in the repo tree via ``model_info(...,
    files_metadata=True)``. Raises :class:`ModelNotFoundError` for a missing
    repo/revision and :class:`ModelAccessError` for a gated repo, instead of
    treating either as "confirmed absent".
    """
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
    return ModelSize(
        repo_id=repo_id,
        revision=revision,
        total_bytes=sum(sizes),
        file_count=len(sizes),
    )


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
