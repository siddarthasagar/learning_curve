"""GPU hardware reference data — model name and architecture per EC2 instance type.

Sourced live from Vantage's (ec2instances.info) public instance catalog at
``https://instances.vantage.sh/instances.json``, not any AWS API: no AWS API
exposes GPU model name or architecture generation at all — confirmed
2026-08-18 by dumping every attribute key on GPU instance types across both
the EC2 and SageMaker Price List Bulk API entries; only ``gpu`` (count) and
``gpuMemory`` (a size string) exist there.

Vantage/ec2instances.info is third-party, not official AWS data — but a
long-established, actively-maintained community catalog. Confirmed live
2026-08-18 that its ``GPU_model``/``gpu_architectures`` fields agree with
every value this project previously hand-maintained in a static table, for
every family ``research_topics/llm_hosting/`` tracks. Vantage's advertised
``POST /api/v1/virtual-instances`` "REST API" was also checked live and
turned out to return the website's HTML shell, not JSON (405 on POST, plain
Next.js page on GET) — not a real filtered-query alternative, so the only
way in is the full bulk feed.

That bulk feed is ~310 MB for 1,400+ instance types, almost all of it
per-region pricing data this module doesn't need. It's downloaded once (or
on ``force_refresh``) into ``dump/cache/vantage/`` (see ``paths.py`` — one
copy, not versioned, alongside the AWS offer-file cache), and only the ~84
GPU-bearing instance types get extracted into a second, tiny (~12 KB) cache
file — every other lookup reads that small file, not the raw 310 MB one.
``aws_pricing.py`` additionally only re-triggers that download when a
confirmed-priced instance type is missing from the cached catalog entirely,
rather than on every run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import ijson
import requests

from llm_hosting_data.paths import CACHE_DIR as _SHARED_CACHE_DIR

if TYPE_CHECKING:
    from pathlib import Path

CACHE_DIR = _SHARED_CACHE_DIR / "vantage"

_INSTANCES_URL = "https://instances.vantage.sh/instances.json"
_REQUEST_TIMEOUT_SECONDS = 180
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


class GpuReferenceFetchError(RuntimeError):
    """Raised when the Vantage instance catalog cannot be fetched or parsed."""


@dataclass(frozen=True)
class GpuReference:
    """GPU model, architecture, and compute capability for one EC2 instance type.

    Each field is independently best-effort: Vantage's own catalog has gaps
    per instance type (e.g. ``p5.4xlarge`` carries ``GPU_model: "NVIDIA
    H100"`` but has no ``gpu_architectures`` key at all — confirmed live
    2026-08-19). Requiring every field to be present would silently drop
    the whole record over one missing field, discarding the parts that are
    there.
    """

    instance_type: str
    gpu_model: str
    architecture: str | None
    compute_capability: str | None


def _raw_catalog_path() -> Path:
    return CACHE_DIR / "instances.json"


def _extracted_catalog_path() -> Path:
    return CACHE_DIR / "gpu_catalog.json"


def _download_raw_catalog(*, force: bool = False) -> Path:
    destination = _raw_catalog_path()
    if destination.exists() and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_destination = destination.with_suffix(".json.part")
    try:
        with requests.get(
            _INSTANCES_URL,
            stream=True,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            with tmp_destination.open("wb") as tmp_file:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    tmp_file.write(chunk)
    except requests.RequestException as exc:
        tmp_destination.unlink(missing_ok=True)
        msg = f"Failed to download the Vantage instance catalog from {_INSTANCES_URL}"
        raise GpuReferenceFetchError(msg) from exc

    tmp_destination.replace(destination)
    return destination


def _compute_capability_label(value: Any) -> str | None:  # noqa: ANN401 -- raw JSON value
    # ijson parses JSON floats as decimal.Decimal by default (not a plain
    # float), so an isinstance(value, int | float) check silently rejects
    # every real parsed value -- confirmed by a test failure, not a guess.
    # float() converts int/float/Decimal/numeric-str alike; anything else
    # (None, a non-numeric string) raises and falls through to None.
    try:
        return f"SM{round(float(value) * 10)}"
    except (TypeError, ValueError):
        return None


def _extract_gpu_catalog(raw_catalog: Path) -> dict[str, GpuReference]:
    catalog: dict[str, GpuReference] = {}
    with raw_catalog.open("rb") as handle:
        for item in ijson.items(handle, "item", use_float=True):
            if not item.get("GPU"):
                continue
            instance_type = item.get("instance_type")
            gpu_model = item.get("GPU_model")
            if not (instance_type and gpu_model):
                continue
            architectures = item.get("gpu_architectures") or []
            catalog[instance_type] = GpuReference(
                instance_type=instance_type,
                gpu_model=gpu_model,
                architecture=architectures[0] if architectures else None,
                compute_capability=_compute_capability_label(
                    item.get("compute_capability"),
                ),
            )
    return catalog


def get_gpu_catalog(*, force_refresh: bool = False) -> dict[str, GpuReference]:
    """Return a ``{instance_type: GpuReference}`` map for every GPU instance type.

    Downloads and parses Vantage's ~310 MB raw catalog only on the first
    call (or when ``force_refresh`` is set); every later call reads the
    small extracted GPU-only file instead. Raises
    :class:`GpuReferenceFetchError` if the raw catalog can't be fetched —
    callers that treat this as best-effort enrichment (see
    ``aws_pricing.py``) should catch that rather than let it fail pricing
    lookups outright.
    """
    extracted_path = _extracted_catalog_path()
    if extracted_path.exists() and not force_refresh:
        raw = json.loads(extracted_path.read_text())
        return {key: GpuReference(**value) for key, value in raw.items()}

    raw_catalog = _download_raw_catalog(force=force_refresh)
    catalog = _extract_gpu_catalog(raw_catalog)

    extracted_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_path.write_text(
        json.dumps({key: asdict(value) for key, value in catalog.items()}),
    )
    return catalog


def lookup_gpu_reference(
    instance_type: str,
    catalog: dict[str, GpuReference] | None = None,
) -> GpuReference | None:
    """Look up GPU model/architecture for one instance type.

    Strips SageMaker's "ml." prefix if present. Pass a pre-fetched
    ``catalog`` (from :func:`get_gpu_catalog`, called once) when looking up
    many instance types, to avoid re-reading the cache file per lookup.
    """
    resolved_catalog = catalog if catalog is not None else get_gpu_catalog()
    return resolved_catalog.get(instance_type.removeprefix("ml."))
