"""Tests for AWS Price List Bulk API filtering.

Uses small hand-built offer-file fixtures (the real files are hundreds of MB)
to lock in schema quirks discovered while validating this against
``research_topics/llm_hosting/04-aws-gpu-capacity-quantization-pricing-matrix.md``:
EC2 lists a same-instance-type "CapacityBlock" SKU with a dummy $0 on-demand
price, SageMaker multiplexes one instance type across unrelated product
families (Hosting, Studio, Training, ...) that must not be conflated, and
family matching must not conflate e.g. "p4d" with "p4de".
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from llm_hosting_data import aws_pricing

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _no_live_gpu_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep these tests offline and independent of any locally cached Vantage data.

    Individual tests that care about the GPU join override this again with
    their own fixture data.
    """
    monkeypatch.setattr(aws_pricing, "get_gpu_catalog", lambda **_k: {})


def _write_offer_file(
    path: Path,
    products: dict[str, Any],
    on_demand: dict[str, Any],
) -> None:
    payload = {"products": products, "terms": {"OnDemand": on_demand}}
    path.write_text(json.dumps(payload))


def _on_demand_term(sku: str, usd: str) -> dict[str, Any]:
    return {
        f"{sku}.RATECODE": {
            "priceDimensions": {
                f"{sku}.RATECODE.DIM": {"unit": "Hrs", "pricePerUnit": {"USD": usd}},
            },
        },
    }


def _ec2_attributes(instance_type: str) -> dict[str, Any]:
    return {
        "instanceType": instance_type,
        "operatingSystem": "Linux",
        "tenancy": "Shared",
        "capacitystatus": "Used",
        "preInstalledSw": "NA",
        "marketoption": "OnDemand",
    }


def test_ec2_pricing_excludes_capacity_block_sku(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offer_file = tmp_path / "ec2.json"
    _write_offer_file(
        offer_file,
        products={
            "SKU-ONDEMAND": {
                "attributes": {
                    "instanceType": "p5.4xlarge",
                    "operatingSystem": "Linux",
                    "tenancy": "Shared",
                    "capacitystatus": "Used",
                    "preInstalledSw": "NA",
                    "marketoption": "OnDemand",
                },
            },
            "SKU-CAPACITYBLOCK": {
                "attributes": {
                    "instanceType": "p5.4xlarge",
                    "operatingSystem": "Linux",
                    "tenancy": "Shared",
                    "capacitystatus": "Used",
                    "preInstalledSw": "NA",
                    "marketoption": "CapacityBlock",
                },
            },
        },
        on_demand={
            "SKU-ONDEMAND": _on_demand_term("SKU-ONDEMAND", "6.8800000000"),
            "SKU-CAPACITYBLOCK": _on_demand_term("SKU-CAPACITYBLOCK", "0.0000000000"),
        },
    )
    monkeypatch.setattr(
        aws_pricing,
        "download_offer_file",
        lambda *_a, **_k: offer_file,
    )

    results = aws_pricing.fetch_on_demand_pricing("AmazonEC2", ["p5.4xlarge"])

    assert len(results) == 1
    assert results[0].usd_per_hour == pytest.approx(6.88)


def test_sagemaker_pricing_selects_hosting_component_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offer_file = tmp_path / "sagemaker.json"
    _write_offer_file(
        offer_file,
        products={
            "SKU-HOSTING": {
                "attributes": {
                    "instanceName": "ml.g7e.2xlarge",
                    "instanceType": "ml.g7e.2xlarge-hosting",
                    "component": "Hosting",
                },
            },
            "SKU-STUDIO": {
                "attributes": {
                    "instanceName": "ml.g7e.2xlarge",
                    "instanceType": "ml.g7e.2xlarge",
                    "component": "studio-jupyterlab",
                },
            },
        },
        on_demand={
            "SKU-HOSTING": _on_demand_term("SKU-HOSTING", "4.2039000000"),
            "SKU-STUDIO": _on_demand_term("SKU-STUDIO", "2.6100000000"),
        },
    )
    monkeypatch.setattr(
        aws_pricing,
        "download_offer_file",
        lambda *_a, **_k: offer_file,
    )

    results = aws_pricing.fetch_on_demand_pricing("AmazonSageMaker", ["ml.g7e.2xlarge"])

    assert len(results) == 1
    assert results[0].usd_per_hour == pytest.approx(4.2039)
    assert results[0].instance_type == "ml.g7e.2xlarge"


def test_family_expands_to_every_size_in_that_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offer_file = tmp_path / "ec2.json"
    _write_offer_file(
        offer_file,
        products={
            "SKU-XL": {"attributes": _ec2_attributes("g6e.xlarge")},
            "SKU-2XL": {"attributes": _ec2_attributes("g6e.2xlarge")},
            "SKU-OTHER-FAMILY": {"attributes": _ec2_attributes("g7e.2xlarge")},
        },
        on_demand={
            "SKU-XL": _on_demand_term("SKU-XL", "1.8610000000"),
            "SKU-2XL": _on_demand_term("SKU-2XL", "2.2421000000"),
            "SKU-OTHER-FAMILY": _on_demand_term("SKU-OTHER-FAMILY", "3.3631000000"),
        },
    )
    monkeypatch.setattr(
        aws_pricing,
        "download_offer_file",
        lambda *_a, **_k: offer_file,
    )

    results = aws_pricing.fetch_on_demand_pricing("AmazonEC2", [], families=["g6e"])

    assert {r.instance_type for r in results} == {"g6e.xlarge", "g6e.2xlarge"}


def test_family_does_not_conflate_p4d_and_p4de(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offer_file = tmp_path / "ec2.json"
    _write_offer_file(
        offer_file,
        products={
            "SKU-P4D": {"attributes": _ec2_attributes("p4d.24xlarge")},
            "SKU-P4DE": {"attributes": _ec2_attributes("p4de.24xlarge")},
        },
        on_demand={
            "SKU-P4D": _on_demand_term("SKU-P4D", "21.9576000000"),
            "SKU-P4DE": _on_demand_term("SKU-P4DE", "27.4471000000"),
        },
    )
    monkeypatch.setattr(
        aws_pricing,
        "download_offer_file",
        lambda *_a, **_k: offer_file,
    )

    results = aws_pricing.fetch_on_demand_pricing("AmazonEC2", [], families=["p4d"])

    assert {r.instance_type for r in results} == {"p4d.24xlarge"}


def test_instance_price_joins_gpu_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offer_file = tmp_path / "ec2.json"
    _write_offer_file(
        offer_file,
        products={"SKU": {"attributes": _ec2_attributes("g6e.xlarge")}},
        on_demand={"SKU": _on_demand_term("SKU", "1.8610000000")},
    )
    monkeypatch.setattr(
        aws_pricing,
        "download_offer_file",
        lambda *_a, **_k: offer_file,
    )
    monkeypatch.setattr(
        aws_pricing,
        "get_gpu_catalog",
        lambda **_k: {
            "g6e.xlarge": aws_pricing.GpuReference(
                "g6e.xlarge",
                "L40S",
                "Ada Lovelace",
                "SM89",
            ),
        },
    )

    results = aws_pricing.fetch_on_demand_pricing("AmazonEC2", ["g6e.xlarge"])

    assert len(results) == 1
    assert results[0].gpu_model == "L40S"
    assert results[0].gpu_architecture == "Ada Lovelace"


def test_gpu_catalog_refreshes_once_when_confirmed_instance_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offer_file = tmp_path / "ec2.json"
    _write_offer_file(
        offer_file,
        products={"SKU": {"attributes": _ec2_attributes("g6e.xlarge")}},
        on_demand={"SKU": _on_demand_term("SKU", "1.8610000000")},
    )
    monkeypatch.setattr(
        aws_pricing,
        "download_offer_file",
        lambda *_a, **_k: offer_file,
    )

    calls: list[bool] = []

    def _fake_get_gpu_catalog(*, force_refresh: bool = False) -> dict[str, object]:
        calls.append(force_refresh)
        if force_refresh:
            return {
                "g6e.xlarge": aws_pricing.GpuReference(
                    "g6e.xlarge",
                    "L40S",
                    "Ada Lovelace",
                    "SM89",
                ),
            }
        return {}  # cold/stale cache -- doesn't have g6e.xlarge yet

    monkeypatch.setattr(aws_pricing, "get_gpu_catalog", _fake_get_gpu_catalog)

    results = aws_pricing.fetch_on_demand_pricing("AmazonEC2", ["g6e.xlarge"])

    assert calls == [False, True]  # tried the cache first, refreshed exactly once
    assert results[0].gpu_model == "L40S"


def test_gpu_catalog_not_refreshed_when_already_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offer_file = tmp_path / "ec2.json"
    _write_offer_file(
        offer_file,
        products={"SKU": {"attributes": _ec2_attributes("g6e.xlarge")}},
        on_demand={"SKU": _on_demand_term("SKU", "1.8610000000")},
    )
    monkeypatch.setattr(
        aws_pricing,
        "download_offer_file",
        lambda *_a, **_k: offer_file,
    )

    calls: list[bool] = []

    def _fake_get_gpu_catalog(*, force_refresh: bool = False) -> dict[str, object]:
        calls.append(force_refresh)
        return {
            "g6e.xlarge": aws_pricing.GpuReference(
                "g6e.xlarge",
                "L40S",
                "Ada Lovelace",
                "SM89",
            ),
        }

    monkeypatch.setattr(aws_pricing, "get_gpu_catalog", _fake_get_gpu_catalog)

    aws_pricing.fetch_on_demand_pricing("AmazonEC2", ["g6e.xlarge"])

    assert calls == [False]  # already had it -- no refresh needed


def test_gpu_catalog_never_refreshed_for_an_invalid_instance_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that an invalid instance type can't trigger a Vantage refresh.

    A typo'd/invalid instance type never gets priced by AWS, so it never
    reaches the "confirmed instance types" set and can't trigger a spurious
    Vantage refresh -- AWS pricing is checked first, deliberately.
    """
    offer_file = tmp_path / "ec2.json"
    _write_offer_file(offer_file, products={}, on_demand={})
    monkeypatch.setattr(
        aws_pricing,
        "download_offer_file",
        lambda *_a, **_k: offer_file,
    )

    calls: list[bool] = []
    monkeypatch.setattr(
        aws_pricing,
        "get_gpu_catalog",
        lambda **kwargs: calls.append(kwargs.get("force_refresh", False)) or {},
    )

    results = aws_pricing.fetch_on_demand_pricing("AmazonEC2", ["g6e.xlarge-typo"])

    assert results == []
    assert calls == [False]
