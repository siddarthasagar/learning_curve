"""Command-line entry points for the AWS pricing / HuggingFace size pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_hosting_data.aws_pricing import (
    PricingFetchError,
    fetch_combined_pricing,
    fetch_on_demand_pricing,
)
from llm_hosting_data.config import (
    ConfigError,
    load_aws_targets,
    load_benchmark_targets,
    load_hf_targets,
)
from llm_hosting_data.hf_models import (
    ModelAccessError,
    ModelNotFoundError,
    get_model_size,
    list_collection_model_ids,
    list_org_model_ids,
)
from llm_hosting_data.openrouter_benchmarks import (
    OpenRouterAuthError,
    OpenRouterFetchError,
    fetch_benchmarks,
    filter_by_model,
    filter_by_type,
)
from llm_hosting_data.paths import DUMP_DIR
from llm_hosting_data.snapshot import (
    diff_by_key,
    load_latest_snapshot,
    save_snapshot,
    to_jsonable,
)

# Output snapshots (aws-ec2-..., aws-sagemaker-..., hf-model-sizes-...,
# openrouter-benchmarks-...) land directly under DUMP_DIR; raw source-data
# caches (the AWS offer files, Vantage's catalog) live under DUMP_DIR/cache
# -- see paths.py. Both are gitignored: regenerable, not source.


def _print_diff(delta: dict[str, list[object]], key_label: str) -> None:
    if not (delta["added"] or delta["removed"] or delta["changed"]):
        print("No change since last snapshot.")
        return
    for item in delta["added"]:
        print(f"  + added: {item.get(key_label)}")
    for item in delta["removed"]:
        print(f"  - removed: {item.get(key_label)}")
    for entry in delta["changed"]:
        print(f"  ~ changed: {entry['key']}: {entry['old']} -> {entry['new']}")


def _resolve_aws_targets(
    args: argparse.Namespace,
) -> tuple[list[str], list[str]] | None:
    instance_types = list(args.instance_types or [])
    families = list(args.family or [])

    if args.config:
        try:
            targets = load_aws_targets(args.config)
        except ConfigError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return None
        instance_types.extend(targets.instance_types)
        families.extend(targets.families)

    if not instance_types and not families:
        print(
            "No instance types or families given — pass --instance-types, --family, "
            "or --config.",
            file=sys.stderr,
        )
        return None
    return instance_types, families


def _run_aws_pricing_combined(
    args: argparse.Namespace,
    instance_types: list[str],
    families: list[str],
) -> int:
    try:
        rows = fetch_combined_pricing(
            instance_types,
            families=families,
            region=args.region,
            force_refresh=args.refresh,
        )
    except PricingFetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print("No matching SKUs found — check instance type/family names and region.")
        return 1

    for row in rows:
        ec2 = (
            f"${row.ec2_usd_per_hour:.4f}/hr"
            if row.ec2_usd_per_hour is not None
            else "n/a"
        )
        sagemaker = (
            f"${row.sagemaker_usd_per_hour:.4f}/hr"
            if row.sagemaker_usd_per_hour is not None
            else "n/a"
        )
        print(
            f"{row.instance_type:<22} EC2 {ec2:>14}   SageMaker {sagemaker:>14}   "
            f"[{', '.join(row.available_on)}]",
        )

    snapshot_name = f"aws-combined-{args.region}"
    previous = load_latest_snapshot(snapshot_name, DUMP_DIR)
    save_snapshot(snapshot_name, rows, DUMP_DIR)
    if previous is not None:
        delta = diff_by_key(previous, to_jsonable(rows), key="instance_type")
        _print_diff(delta, key_label="instance_type")
    return 0


def _run_aws_pricing(args: argparse.Namespace) -> int:
    targets = _resolve_aws_targets(args)
    if targets is None:
        return 1
    instance_types, families = targets

    if args.service == "both":
        return _run_aws_pricing_combined(args, instance_types, families)

    service_code = "AmazonEC2" if args.service == "ec2" else "AmazonSageMaker"
    try:
        prices = fetch_on_demand_pricing(
            service_code,
            instance_types,
            families=families,
            region=args.region,
            force_refresh=args.refresh,
        )
    except PricingFetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not prices:
        print("No matching SKUs found — check instance type/family names and region.")
        return 1

    for price in prices:
        print(
            f"{price.instance_type:<22} ${price.usd_per_hour:>10.4f}/hr  ({price.region})",
        )

    snapshot_name = f"aws-{args.service}-{args.region}"
    previous = load_latest_snapshot(snapshot_name, DUMP_DIR)
    save_snapshot(snapshot_name, prices, DUMP_DIR)
    if previous is not None:
        delta = diff_by_key(previous, to_jsonable(prices), key="instance_type")
        _print_diff(delta, key_label="instance_type")
    return 0


def _run_hf_size(args: argparse.Namespace) -> int:
    repo_ids = list(args.repo_ids)
    collections = [args.collection] if args.collection else []
    authors = [args.author] if args.author else []

    if args.config:
        try:
            targets = load_hf_targets(args.config)
        except ConfigError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        repo_ids.extend(targets.repo_ids)
        collections.extend(targets.collections)
        authors.extend(targets.authors)

    for collection in collections:
        repo_ids.extend(list_collection_model_ids(collection, token=args.token))
    for author in authors:
        repo_ids.extend(list_org_model_ids(author, token=args.token))

    if not repo_ids:
        print(
            "No repo IDs given — pass repo IDs, --collection, --author, or --config.",
            file=sys.stderr,
        )
        return 1

    results = []
    had_error = False
    for repo_id in dict.fromkeys(repo_ids):
        try:
            size = get_model_size(repo_id, token=args.token)
        except (ModelNotFoundError, ModelAccessError) as exc:
            print(f"{repo_id}: ERROR - {exc}", file=sys.stderr)
            had_error = True
            continue
        print(f"{repo_id}: {size.total_gib:.2f} GiB ({size.file_count} files)")
        results.append(size)

    if results:
        snapshot_name = "hf-model-sizes"
        previous = load_latest_snapshot(snapshot_name, DUMP_DIR)
        save_snapshot(snapshot_name, results, DUMP_DIR)
        if previous is not None:
            delta = diff_by_key(previous, to_jsonable(results), key="repo_id")
            _print_diff(delta, key_label="repo_id")

    return 1 if had_error and not results else 0


def _benchmark_diff_key(item: dict[str, object]) -> str:
    fields = item.get("fields")
    fields = fields if isinstance(fields, dict) else {}
    discriminator = fields.get("benchmark_type") or fields.get("category") or ""
    return f"{item.get('source')}|{item.get('model_permaslug')}|{discriminator}"


def _with_diff_key(items: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{**item, "_diff_key": _benchmark_diff_key(item)} for item in items]


def _run_benchmarks(args: argparse.Namespace) -> int:
    model_slugs = list(args.models)
    if args.config:
        try:
            model_slugs.extend(load_benchmark_targets(args.config))
        except ConfigError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if not model_slugs:
        print("No model slugs given — pass slugs or --config.", file=sys.stderr)
        return 1

    try:
        entries = fetch_benchmarks(token=args.token, force_refresh=args.refresh)
    except (OpenRouterAuthError, OpenRouterFetchError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    entries = filter_by_model(entries, model_slugs)
    entries = filter_by_type(
        entries,
        source=args.source,
        benchmark_type=args.benchmark_type,
    )

    if not entries:
        print(
            "No matching benchmark entries — check model slugs / --source / --benchmark-type.",
        )
        return 1

    skip_verbose_fields = {"pricing", "tournament_stats"}
    for entry in entries:
        summary = ", ".join(
            f"{key}={value}"
            for key, value in entry.fields.items()
            if key not in skip_verbose_fields
        )
        print(f"{entry.source:18} {entry.model_permaslug:45} {summary}")

    snapshot_name = "openrouter-benchmarks"
    previous = load_latest_snapshot(snapshot_name, DUMP_DIR)
    save_snapshot(snapshot_name, entries, DUMP_DIR)
    if previous is not None:
        delta = diff_by_key(
            _with_diff_key(previous),
            _with_diff_key(to_jsonable(entries)),
            key="_diff_key",
        )
        _print_diff(delta, key_label="model_permaslug")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-hosting-data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    aws_parser = subparsers.add_parser(
        "aws-pricing",
        help="Fetch AWS on-demand pricing.",
    )
    aws_parser.add_argument(
        "--service",
        choices=["ec2", "sagemaker", "both"],
        default="ec2",
        help='"both" merges EC2 and SageMaker Hosting pricing per instance type.',
    )
    aws_parser.add_argument("--region", default="us-east-1")
    aws_parser.add_argument(
        "--instance-types",
        nargs="*",
        default=None,
        help='Exact instance types, e.g. "g7e.2xlarge".',
    )
    aws_parser.add_argument(
        "--family",
        nargs="*",
        default=None,
        help='Instance families to expand, e.g. "g6e" -> every g6e size.',
    )
    aws_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help='YAML file of families/instance_types, e.g. "config/aws_instances.yaml"',
    )
    aws_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass the local cache.",
    )
    aws_parser.set_defaults(handler=_run_aws_pricing)

    hf_parser = subparsers.add_parser("hf-size", help="Fetch HuggingFace model sizes.")
    hf_parser.add_argument("repo_ids", nargs="*", default=[])
    hf_parser.add_argument(
        "--collection",
        default=None,
        help='e.g. "nvidia/nemotron-v3"',
    )
    hf_parser.add_argument("--author", default=None, help='e.g. "Qwen"')
    hf_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help='YAML file of repo_ids/collections/authors, e.g. "config/hf_models.yaml"',
    )
    hf_parser.add_argument("--token", default=None, help="HuggingFace access token.")
    hf_parser.set_defaults(handler=_run_hf_size)

    bench_parser = subparsers.add_parser(
        "benchmarks",
        help="Fetch model benchmark scores from OpenRouter.",
    )
    bench_parser.add_argument("models", nargs="*", default=[])
    bench_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help='YAML file of model slug substrings, e.g. "config/benchmark_models.yaml"',
    )
    bench_parser.add_argument(
        "--source",
        choices=["artificial-analysis", "design-arena", "openrouter"],
        default=None,
    )
    bench_parser.add_argument(
        "--benchmark-type",
        default=None,
        help='openrouter-native rows only, e.g. "tau_bench_verified_airline"',
    )
    bench_parser.add_argument(
        "--token",
        default=None,
        help="OpenRouter API token (defaults to OPENROUTER_API_TOKEN env var).",
    )
    bench_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass the local cache.",
    )
    bench_parser.set_defaults(handler=_run_benchmarks)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the matching subcommand handler."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
