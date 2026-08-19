"""Fetch AWS on-demand instance pricing from the public Price List Bulk API.

Uses ``https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/<service>/current/<region>/index.json``,
which is unauthenticated and needs no AWS account or IAM credentials. The EC2
offer file for a single region can exceed 400 MB, so it is downloaded once to
a local cache and then stream-parsed with ``ijson`` rather than loaded whole
into memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ijson
import requests

from llm_hosting_data.gpu_reference import (
    GpuReference,
    GpuReferenceFetchError,
    get_gpu_catalog,
    lookup_gpu_reference,
)
from llm_hosting_data.instance_naming import family_of
from llm_hosting_data.paths import CACHE_DIR as _SHARED_CACHE_DIR

if TYPE_CHECKING:
    from pathlib import Path

CACHE_DIR = _SHARED_CACHE_DIR / "aws_pricing"

_BULK_API_BASE = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws"
_REQUEST_TIMEOUT_SECONDS = 60
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024

# AWS's bulk pricing schema is not uniform across services: EC2 keys each SKU's
# clean instance identifier as `instanceType`, but SageMaker's `instanceType`
# is suffixed per product family (e.g. "ml.g7e.2xlarge-hosting" for real-time
# hosting vs plain "ml.g7e.2xlarge" for Studio) while `instanceName` stays
# clean across every family. SageMaker also multiplexes one instance type
# across many unrelated product families (Hosting, Training, Studio,
# Notebook, Processing, AsyncInf, Cluster, ...) that carry different prices,
# so it additionally needs a `component` filter or results collide/duplicate.
_INSTANCE_FIELD_BY_SERVICE: dict[str, str] = {
    "AmazonEC2": "instanceType",
    "AmazonSageMaker": "instanceName",
}
# Same cross-service inconsistency for the vCPU count: EC2 keys it lowercase
# ("vcpu"), SageMaker capitalizes the C ("vCpu"). Reading the wrong key
# silently returns None instead of erroring, so this needs to be explicit
# rather than assumed -- confirmed live 2026-08-18 that ``vcpu=attributes.get
# ("vcpu")`` was silently None for every SageMaker row before this map
# existed. Physical-processor and dedicated-EBS-throughput fields have no
# SageMaker equivalent worth mapping (SageMaker's "physicalCpu"/"physicalGpu"
# were observed as literally "N/A"), so those are read from EC2's key name
# only and naturally come back None for SageMaker.
_VCPU_FIELD_BY_SERVICE: dict[str, str] = {
    "AmazonEC2": "vcpu",
    "AmazonSageMaker": "vCpu",
}
DEFAULT_SAGEMAKER_COMPONENT = "Hosting"


class PricingFetchError(RuntimeError):
    """Raised when the AWS Price List Bulk API cannot be fetched or parsed."""


@dataclass(frozen=True)
class InstancePrice:
    """On-demand hourly price and hardware specs for one instance type in one region.

    Every spec field through ``clock_speed`` comes from the same offer-file
    attributes already streamed through for pricing -- no separate API call
    or AWS credentials needed, since the Price List Bulk API already carries
    vCPU/memory/GPU/network/storage specs alongside the dollar figure.
    ``gpu_model``/``gpu_architecture``/``gpu_memory_bandwidth_gbps``/
    ``gpu_nvlink_generation``/``gpu_nvlink_bandwidth_gbps`` are the
    exception: no AWS API exposes any of those (verified 2026-08-18 for
    model/architecture, 2026-08-19 for bandwidth/NVLink), so they're joined
    in from ``gpu_reference.py`` instead -- model/architecture live from
    Vantage, bandwidth/NVLink from ``config/gpu_hardware_specs.yaml``
    (sourced from NVIDIA's own spec pages; neither AWS nor Vantage has that
    data at all).
    Any of them is ``None`` if that lookup fails or doesn't cover the
    instance type -- a best-effort enrichment, not something pricing itself
    depends on. ``gpu_memory`` (capacity, e.g. ``"48 GB"``) is a different
    thing from ``gpu_memory_bandwidth_gbps`` (throughput, e.g. ``864.0``) --
    don't conflate them.
    """

    service_code: str
    instance_type: str
    region: str
    usd_per_hour: float
    vcpu: str | None
    memory: str | None
    gpu: str | None
    gpu_model: str | None
    gpu_architecture: str | None
    gpu_compute_capability: str | None
    gpu_memory: str | None
    gpu_memory_bandwidth_gbps: float | None
    gpu_nvlink_generation: str | None
    gpu_nvlink_bandwidth_gbps: float | None
    network_performance: str | None
    dedicated_ebs_throughput: str | None
    storage: str | None
    physical_processor: str | None
    clock_speed: str | None
    usage_type: str


def _offer_file_url(service_code: str, region: str) -> str:
    return f"{_BULK_API_BASE}/{service_code}/current/{region}/index.json"


def _cache_path(service_code: str, region: str) -> Path:
    return CACHE_DIR / f"{service_code}-{region}.json"


def download_offer_file(service_code: str, region: str, *, force: bool = False) -> Path:
    """Download (and cache) the Price List Bulk API offer file for one service/region.

    Returns the cached local path. Subsequent calls reuse the cached file
    unless ``force`` is set, since the EC2 file alone is several hundred MB.
    """
    destination = _cache_path(service_code, region)
    if destination.exists() and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = _offer_file_url(service_code, region)
    tmp_destination = destination.with_suffix(".json.part")
    try:
        with requests.get(
            url,
            stream=True,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            with tmp_destination.open("wb") as tmp_file:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    tmp_file.write(chunk)
    except requests.RequestException as exc:
        tmp_destination.unlink(missing_ok=True)
        msg = f"Failed to download AWS offer file from {url}"
        raise PricingFetchError(msg) from exc

    tmp_destination.replace(destination)
    return destination


def _matches_target(
    service_code: str,
    attributes: dict[str, Any],
    instance_types: set[str],
    families: set[str],
    sagemaker_component: str,
) -> bool:
    field = _INSTANCE_FIELD_BY_SERVICE.get(service_code, "instanceType")
    identifier = attributes.get(field)
    if identifier is None:
        return False
    if identifier not in instance_types and family_of(identifier) not in families:
        return False
    if (
        service_code == "AmazonSageMaker"
        and attributes.get("component") != sagemaker_component
    ):
        return False
    return (
        attributes.get("operatingSystem", "Linux") == "Linux"
        and attributes.get("tenancy", "Shared") == "Shared"
        and attributes.get("capacitystatus", "Used") == "Used"
        and attributes.get("preInstalledSw", "NA") == "NA"
        # Some instance types also list a "CapacityBlock" SKU alongside the
        # true on-demand one; its OnDemand price term is a dummy $0 entry
        # (Capacity Blocks are purchased as reservations, not hourly).
        and attributes.get("marketoption", "OnDemand") == "OnDemand"
    )


def _read_matching_products(
    offer_file: Path,
    service_code: str,
    instance_types: set[str],
    families: set[str],
    sagemaker_component: str,
) -> dict[str, dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    with offer_file.open("rb") as handle:
        for sku, product in ijson.kvitems(handle, "products"):
            attributes = product.get("attributes", {})
            if _matches_target(
                service_code,
                attributes,
                instance_types,
                families,
                sagemaker_component,
            ):
                products[sku] = attributes
    return products


def _read_on_demand_prices(offer_file: Path, skus: set[str]) -> dict[str, float]:
    sku_to_price: dict[str, float] = {}
    with offer_file.open("rb") as handle:
        for sku, offer in ijson.kvitems(handle, "terms.OnDemand"):
            if sku not in skus:
                continue
            for term in offer.values():
                for dimension in term.get("priceDimensions", {}).values():
                    usd = dimension.get("pricePerUnit", {}).get("USD")
                    if usd is not None:
                        sku_to_price[sku] = float(usd)
    return sku_to_price


def _normalize_instance_types(
    service_code: str,
    instance_types: list[str],
) -> list[str]:
    """SageMaker's clean instance identifier is always "ml."-prefixed; accept either form.

    Lets every caller (CLI, ``fetch_combined_pricing``) pass plain names like
    ``"g7e.2xlarge"`` regardless of service, instead of each caller having to
    know and apply the SageMaker-specific prefix itself.
    """
    if service_code != "AmazonSageMaker":
        return list(instance_types)
    return [t if t.startswith("ml.") else f"ml.{t}" for t in instance_types]


def fetch_on_demand_pricing(  # noqa: PLR0913 -- all keyword-only past service/instance_types
    service_code: str,
    instance_types: list[str],
    *,
    families: list[str] | None = None,
    region: str = "us-east-1",
    force_refresh: bool = False,
    sagemaker_component: str = DEFAULT_SAGEMAKER_COMPONENT,
) -> list[InstancePrice]:
    """Fetch on-demand hourly pricing and hardware specs for the given instance types/families.

    Downloads (and caches) the offer file for ``service_code``/``region``,
    then filters it down to ``instance_types`` (exact) and ``families``
    (every size within a family, e.g. ``"g6e"`` -> g6e.xlarge, g6e.2xlarge,
    ...) in a single streaming pass, so the full offer file is never fully
    materialized in memory and family membership doesn't need a
    hand-maintained catalog of every size AWS currently ships. For
    ``AmazonSageMaker``, ``sagemaker_component`` picks which product family
    to price (default ``"Hosting"``, i.e. real-time inference endpoints —
    the same one EC2-comparable "SageMaker price" figures normally mean, as
    opposed to Training, Studio, Notebook, Processing, etc., which price the
    same instance type differently). ``instance_types`` may be given with or
    without SageMaker's "ml." prefix; it's normalized either way.
    """
    field = _INSTANCE_FIELD_BY_SERVICE.get(service_code, "instanceType")
    vcpu_field = _VCPU_FIELD_BY_SERVICE.get(service_code, "vcpu")
    target_types = set(_normalize_instance_types(service_code, instance_types))
    target_families = set(families or [])
    offer_file = download_offer_file(service_code, region, force=force_refresh)

    products = _read_matching_products(
        offer_file,
        service_code,
        target_types,
        target_families,
        sagemaker_component,
    )
    prices = _read_on_demand_prices(offer_file, set(products))

    # AWS pricing is checked first, deliberately: only instance types that
    # actually got a real price (a genuine SKU, not a typo or an
    # out-of-scope family) count as "confirmed" for the Vantage
    # staleness check below, so a bad --instance-types value can never
    # trigger a spurious 310 MB Vantage re-download.
    confirmed_instance_types = {
        attributes[field].removeprefix("ml.")
        for sku, attributes in products.items()
        if sku in prices
    }
    gpu_catalog = _gpu_catalog_for(
        confirmed_instance_types,
        force_refresh=force_refresh,
    )

    results = [
        _build_instance_price(
            service_code,
            region,
            attributes,
            prices[sku],
            field,
            vcpu_field,
            gpu_catalog,
        )
        for sku, attributes in products.items()
        if sku in prices
    ]
    results.sort(key=lambda price: (price.instance_type, price.usd_per_hour))
    return results


def _gpu_catalog_for(
    confirmed_instance_types: set[str],
    *,
    force_refresh: bool,
) -> dict[str, GpuReference]:
    """Fetch Vantage's GPU catalog, refreshing only when there's a real signal to.

    ``force_refresh=True`` (the CLI's ``--refresh``) always re-downloads, as
    for the AWS offer file. Otherwise the cached catalog is trusted as-is
    *unless* a ``confirmed_instance_types`` entry (a type AWS just priced --
    see the caller) is missing from it entirely: since every family this
    pipeline prices is GPU-bearing, that gap most likely means Vantage
    hasn't cataloged a newer instance type yet, not that this pipeline's
    filters are wrong. One retry, not a loop: if the refresh still doesn't
    have it, Vantage simply doesn't have it (yet).

    GPU model/architecture is enrichment on top of pricing either way, not
    something pricing itself should fail over -- any fetch failure here
    just means ``gpu_model``/``gpu_architecture`` come back ``None``.
    """
    try:
        catalog = get_gpu_catalog(force_refresh=force_refresh)
    except GpuReferenceFetchError:
        return {}

    if not force_refresh and confirmed_instance_types - catalog.keys():
        try:
            refreshed = get_gpu_catalog(force_refresh=True)
        except GpuReferenceFetchError:
            refreshed = None
        if refreshed:
            catalog = refreshed
    return catalog


def _build_instance_price(  # noqa: PLR0913, PLR0917 -- internal row builder, one call site
    service_code: str,
    region: str,
    attributes: dict[str, Any],
    usd_per_hour: float,
    field: str,
    vcpu_field: str,
    gpu_catalog: dict[str, GpuReference],
) -> InstancePrice:
    instance_type = attributes[field]
    gpu_reference = lookup_gpu_reference(instance_type, gpu_catalog)
    return InstancePrice(
        service_code=service_code,
        instance_type=instance_type,
        region=region,
        usd_per_hour=usd_per_hour,
        vcpu=attributes.get(vcpu_field),
        memory=attributes.get("memory"),
        gpu=attributes.get("gpu"),
        gpu_model=gpu_reference.gpu_model if gpu_reference else None,
        gpu_architecture=gpu_reference.architecture if gpu_reference else None,
        gpu_compute_capability=gpu_reference.compute_capability
        if gpu_reference
        else None,
        gpu_memory=attributes.get("gpuMemory"),
        gpu_memory_bandwidth_gbps=gpu_reference.memory_bandwidth_gbps
        if gpu_reference
        else None,
        gpu_nvlink_generation=gpu_reference.nvlink_generation
        if gpu_reference
        else None,
        gpu_nvlink_bandwidth_gbps=gpu_reference.nvlink_bandwidth_gbps
        if gpu_reference
        else None,
        network_performance=attributes.get("networkPerformance"),
        dedicated_ebs_throughput=attributes.get("dedicatedEbsThroughput"),
        storage=attributes.get("storage"),
        physical_processor=attributes.get("physicalProcessor"),
        clock_speed=attributes.get("clockSpeed"),
        usage_type=attributes.get("usagetype", ""),
    )


@dataclass(frozen=True)
class CombinedInstancePricing:
    """EC2 on-demand and SageMaker Hosting pricing for one instance type, side by side.

    ``available_on`` names which service(s) actually list this instance type
    at all — not every EC2 instance type has a SageMaker Hosting SKU (e.g.
    p5.4xlarge and p6-b300.48xlarge, confirmed live 2026-08-18, exist on EC2
    but have no SageMaker Hosting listing whatsoever), so a missing price
    there is a real "not offered," not a fetch failure.
    """

    instance_type: str
    region: str
    vcpu: str | None
    memory: str | None
    gpu: str | None
    gpu_model: str | None
    gpu_architecture: str | None
    gpu_compute_capability: str | None
    gpu_memory: str | None
    gpu_memory_bandwidth_gbps: float | None
    gpu_nvlink_generation: str | None
    gpu_nvlink_bandwidth_gbps: float | None
    network_performance: str | None
    ec2_usd_per_hour: float | None
    sagemaker_usd_per_hour: float | None
    available_on: list[str]


def fetch_combined_pricing(
    instance_types: list[str],
    *,
    families: list[str] | None = None,
    region: str = "us-east-1",
    force_refresh: bool = False,
    sagemaker_component: str = DEFAULT_SAGEMAKER_COMPONENT,
) -> list[CombinedInstancePricing]:
    """Fetch EC2 and SageMaker Hosting pricing for the same instance types/families, merged.

    One row per instance type that appears on either service, with both
    prices side by side and ``available_on`` showing which service(s)
    actually offer it — see :class:`CombinedInstancePricing`.
    """
    ec2_prices = fetch_on_demand_pricing(
        "AmazonEC2",
        instance_types,
        families=families,
        region=region,
        force_refresh=force_refresh,
    )
    sagemaker_prices = fetch_on_demand_pricing(
        "AmazonSageMaker",
        instance_types,
        families=families,
        region=region,
        force_refresh=force_refresh,
        sagemaker_component=sagemaker_component,
    )
    sagemaker_by_type = {
        price.instance_type.removeprefix("ml."): price for price in sagemaker_prices
    }

    combined: dict[str, CombinedInstancePricing] = {}
    for price in ec2_prices:
        sagemaker_price = sagemaker_by_type.get(price.instance_type)
        available_on = ["ec2", "sagemaker"] if sagemaker_price else ["ec2"]
        combined[price.instance_type] = CombinedInstancePricing(
            instance_type=price.instance_type,
            region=region,
            vcpu=price.vcpu,
            memory=price.memory,
            gpu=price.gpu,
            gpu_model=price.gpu_model,
            gpu_architecture=price.gpu_architecture,
            gpu_compute_capability=price.gpu_compute_capability,
            gpu_memory=price.gpu_memory,
            gpu_memory_bandwidth_gbps=price.gpu_memory_bandwidth_gbps,
            gpu_nvlink_generation=price.gpu_nvlink_generation,
            gpu_nvlink_bandwidth_gbps=price.gpu_nvlink_bandwidth_gbps,
            network_performance=price.network_performance,
            ec2_usd_per_hour=price.usd_per_hour,
            sagemaker_usd_per_hour=sagemaker_price.usd_per_hour
            if sagemaker_price
            else None,
            available_on=available_on,
        )
    for instance_type, sagemaker_price in sagemaker_by_type.items():
        if instance_type in combined:
            continue
        combined[instance_type] = CombinedInstancePricing(
            instance_type=instance_type,
            region=region,
            vcpu=sagemaker_price.vcpu,
            memory=sagemaker_price.memory,
            gpu=sagemaker_price.gpu,
            gpu_model=sagemaker_price.gpu_model,
            gpu_architecture=sagemaker_price.gpu_architecture,
            gpu_compute_capability=sagemaker_price.gpu_compute_capability,
            gpu_memory=sagemaker_price.gpu_memory,
            gpu_memory_bandwidth_gbps=sagemaker_price.gpu_memory_bandwidth_gbps,
            gpu_nvlink_generation=sagemaker_price.gpu_nvlink_generation,
            gpu_nvlink_bandwidth_gbps=sagemaker_price.gpu_nvlink_bandwidth_gbps,
            network_performance=sagemaker_price.network_performance,
            ec2_usd_per_hour=None,
            sagemaker_usd_per_hour=sagemaker_price.usd_per_hour,
            available_on=["sagemaker"],
        )

    results = list(combined.values())
    results.sort(key=lambda row: row.instance_type)
    return results
