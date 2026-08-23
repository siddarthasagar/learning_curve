"""Command-line entry points for the AWS pricing / HuggingFace size pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from llm_hosting_data.aws_pricing import (
    PricingFetchError,
    fetch_combined_pricing,
    fetch_on_demand_pricing,
)
from llm_hosting_data.config import (
    BenchmarkComparisonTargets,
    ConfigError,
    load_aws_targets,
    load_benchmark_targets,
    load_gpu_hardware_specs,
    load_hf_targets,
    load_kernel_support_matrix,
)
from llm_hosting_data.hf_models import (
    ModelAccessError,
    ModelNotFoundError,
    get_model_size,
    list_collection_model_ids,
    list_org_model_ids,
)
from llm_hosting_data.model_instance_fit import (
    CandidateComparison,
    CapacityFit,
    ComparisonDataset,
    InstanceCandidateFit,
    KernelCompatibleFit,
    build_baseline_comparisons,
    build_capacity_pass,
    build_instance_reports,
    build_kernel_filter_pass,
    discover_candidate_slugs,
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
from llm_hosting_data.vllm_registry import (
    VllmRegistryFetchError,
    fetch_quant_registry,
    write_registry_yaml,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm_hosting_data.hf_models import ModelSize

_SubParsers = argparse._SubParsersAction  # noqa: SLF001 -- argparse has no public alias

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
            size = get_model_size(repo_id, token=args.token, force_refresh=args.refresh)
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
            targets = load_benchmark_targets(args.config)
        except ConfigError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        model_slugs.extend(targets.baseline)

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


def _kernel_caveat(support: str | None) -> str:
    """Return a short suffix noting non-native kernel support, or nothing when native/unset.

    "blocked" never reaches here (excluded upstream in
    ``model_instance_fit.py`` before a hosting recommendation is built).
    """
    if support == "fallback":
        return " [runs via kernel fallback, not native]"
    if support == "unknown":
        return " [kernel support unverified]"
    return ""


def _render_compare_report(
    comparisons: list[CandidateComparison],
    instance_report: dict[str, list[InstanceCandidateFit]],
    benchmark_type: str | None,
) -> str:
    lines = [
        "# Model <-> instance fit report",
        "",
        f"Benchmark: {benchmark_type or '(unfiltered)'}",
        "",
    ]

    lines.append("## Candidates that meet or beat the baseline")
    lines.append("")
    if not comparisons:
        lines.append("No candidate met or beat any baseline on this benchmark.")
    for comparison in comparisons:
        lines.append(
            f"### {comparison.candidate_slug} ({comparison.candidate_accuracy:.4f}) "
            f"vs {comparison.baseline_slug} ({comparison.baseline_accuracy:.4f})",
        )
        if not comparison.checkpoints:
            lines.append("- no candidate-pool checkpoints found for this family")
        for checkpoint in comparison.checkpoints:
            hosting = checkpoint.hosting
            tight = (
                f"{hosting.tight_fit_instance} (${hosting.tight_fit_usd_per_hour:.4f}/hr)"
                f"{_kernel_caveat(hosting.tight_fit_kernel_support)}"
                if hosting.tight_fit_instance
                else "none"
            )
            best = (
                f"{hosting.best_fit_instance} (${hosting.best_fit_usd_per_hour:.4f}/hr)"
                f"{_kernel_caveat(hosting.best_fit_kernel_support)}"
                if hosting.best_fit_instance
                else "none"
            )
            lines.append(
                f"- `{checkpoint.repo_id}` ({checkpoint.total_gib:.2f} GiB) "
                f"— tight fit: {tight}; best fit: {best}",
            )
        lines.append("")

    unscored_preview_count = 5
    lines.append("## Best-fit candidates per instance type")
    lines.append("")
    lines.append(
        "Scored candidates (known OpenRouter benchmark family) shown in full; "
        f"unscored candidate-pool checkpoints are previewed (first {unscored_preview_count}) "
        "and counted, not all listed.",
    )
    lines.append("")
    for instance_type, fits in instance_report.items():
        lines.append(f"### {instance_type}")
        if not fits:
            lines.append("- no candidate-pool checkpoint fits")
        scored = [f for f in fits if f.accuracy is not None]
        unscored = [f for f in fits if f.accuracy is None]
        lines.extend(
            f"- [{fit.fit_tier}] `{fit.repo_id}` ({fit.total_gib:.2f} GiB) — "
            f"{fit.accuracy:.4f}{_kernel_caveat(fit.kernel_support)}"
            for fit in scored
        )
        lines.extend(
            f"- [{fit.fit_tier}] `{fit.repo_id}` ({fit.total_gib:.2f} GiB) — "
            f"no benchmark data{_kernel_caveat(fit.kernel_support)}"
            for fit in unscored[:unscored_preview_count]
        )
        remaining = len(unscored) - unscored_preview_count
        if remaining > 0:
            lines.append(f"- ... and {remaining} more without benchmark data")
        lines.append("")

    return "\n".join(lines)


def _load_config_or_report[T](
    loader: Callable[[Path], T],
    config_path: Path,
) -> T | None:
    try:
        return loader(config_path)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return None


def _resolve_model_sizes(
    hf_config: Path,
    *,
    token: str | None,
    force_refresh: bool,
) -> list[ModelSize] | None:
    hf_targets = _load_config_or_report(load_hf_targets, hf_config)
    if hf_targets is None:
        return None
    repo_ids = list(hf_targets.repo_ids)
    for collection in hf_targets.collections:
        repo_ids.extend(list_collection_model_ids(collection, token=token))
    for author in hf_targets.authors:
        repo_ids.extend(list_org_model_ids(author, token=token))

    model_sizes = []
    for repo_id in dict.fromkeys(repo_ids):
        try:
            model_sizes.append(
                get_model_size(repo_id, token=token, force_refresh=force_refresh),
            )
        except (ModelNotFoundError, ModelAccessError) as exc:
            print(f"{repo_id}: ERROR - {exc}", file=sys.stderr)
    return model_sizes


def _resolve_compare_inputs(
    args: argparse.Namespace,
) -> (
    tuple[
        list[ModelSize],
        BenchmarkComparisonTargets,
        dict[str, dict[str, str]],
        dict[str, dict[str, object]],
    ]
    | None
):
    model_sizes = _resolve_model_sizes(
        args.hf_config,
        token=args.hf_token,
        force_refresh=args.refresh,
    )
    if model_sizes is None:
        return None
    comparison_targets = _load_config_or_report(
        load_benchmark_targets,
        args.benchmark_config,
    )
    if comparison_targets is None:
        return None
    kernel_matrix = _load_config_or_report(
        load_kernel_support_matrix,
        args.kernel_config,
    )
    if kernel_matrix is None:
        return None
    gpu_hardware_specs = _load_config_or_report(
        load_gpu_hardware_specs,
        args.gpu_specs_config,
    )
    if gpu_hardware_specs is None:
        return None
    return model_sizes, comparison_targets, kernel_matrix, gpu_hardware_specs


def _render_capacity_pass_markdown(pass1: dict[str, list[CapacityFit]]) -> str:
    lines = [
        "# Rank pipeline — pass 1: raw GPU-memory capacity fit",
        "",
        (
            "Every candidate-pool checkpoint that fits an instance by GPU memory "
            "alone (tight or best tier) — no kernel/quantization-support check "
            "yet, that's pass 2."
        ),
        "",
    ]
    for instance_type, fits in pass1.items():
        lines.append(f"## {instance_type}")
        if not fits:
            lines.append("- no candidate-pool checkpoint fits")
        lines.extend(
            f"- [{fit.fit_tier}] `{fit.repo_id}` ({fit.total_gib:.2f} GiB)"
            for fit in fits
        )
        lines.append("")
    return "\n".join(lines)


def _render_kernel_filter_pass_markdown(
    pass2: dict[str, list[KernelCompatibleFit]],
) -> str:
    lines = [
        "# Rank pipeline — pass 2: kernel/quantization-support filter",
        "",
        (
            "Pass 1 survivors whose quantization format is not `blocked` on "
            "that instance's GPU architecture (config/kernel_support_matrix.yaml). "
            "Pass 3 (rank survivors by benchmark score) is deliberately not "
            "run yet — held off pending active-parameter data, see the "
            "tracking doc §6."
        ),
        "",
    ]
    for instance_type, fits in pass2.items():
        lines.append(f"## {instance_type}")
        if not fits:
            lines.append("- no pass-1 survivor has working kernel support here")
        lines.extend(
            f"- [{fit.fit_tier}] `{fit.repo_id}` ({fit.total_gib:.2f} GiB)"
            f"{_kernel_caveat(fit.kernel_support)}"
            for fit in fits
        )
        lines.append("")
    return "\n".join(lines)


def _run_rank(args: argparse.Namespace) -> int:
    """3-pass iterative model<->instance ranking pipeline (2026-08-22, user request).

    Pass 1: raw GPU-memory capacity filter (tight/best tier) per instance.
    Pass 2: pass-1 survivors filtered by kernel/quantization support
    (excludes ``"blocked"`` combinations). Pass 3 (rank survivors by
    benchmark score) is deliberately not built yet — held off per the
    2026-08-22 decision, rather than shipping a ranking that only ever
    answers doc `04`'s "Smartest" category and silently skips "Fastest
    decode"/"Best overall" (both need active-parameter data this pipeline
    doesn't have yet, tracking doc §6 gap 2).

    Each pass writes its own JSON snapshot (``dump/``, via
    ``save_snapshot()`` — dated history + a ``-latest.json`` pointer) and
    Markdown report (``dump/reports/``) — separate files per pass, never
    overwriting an earlier pass's output.
    """
    aws_args = argparse.Namespace(
        region=args.region,
        instance_types=None,
        family=None,
        config=args.aws_config,
        refresh=args.refresh,
    )
    aws_targets = _resolve_aws_targets(aws_args)
    if aws_targets is None:
        return 1
    instance_types, families = aws_targets
    try:
        instances = fetch_combined_pricing(
            instance_types,
            families=families,
            region=args.region,
            force_refresh=args.refresh,
        )
    except PricingFetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    model_sizes = _resolve_model_sizes(
        args.hf_config,
        token=args.hf_token,
        force_refresh=args.refresh,
    )
    if model_sizes is None:
        return 1
    kernel_matrix = _load_config_or_report(
        load_kernel_support_matrix,
        args.kernel_config,
    )
    if kernel_matrix is None:
        return 1
    gpu_hardware_specs = _load_config_or_report(
        load_gpu_hardware_specs,
        args.gpu_specs_config,
    )
    if gpu_hardware_specs is None:
        return 1

    reports_dir = DUMP_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=== Pass 1: raw GPU-memory capacity fit ===")
    pass1 = build_capacity_pass(
        model_sizes=model_sizes,
        instances=instances,
        gpu_hardware_specs=gpu_hardware_specs,
    )
    pass1_markdown = _render_capacity_pass_markdown(pass1)
    print(pass1_markdown)
    save_snapshot("rank-pass1-capacity", pass1, DUMP_DIR)
    (reports_dir / "rank-pass1-capacity.md").write_text(pass1_markdown)
    print(
        f"Written to {DUMP_DIR / 'rank-pass1-capacity-latest.json'} and dump/reports/rank-pass1-capacity.md",
    )

    print("\n=== Pass 2: kernel/quantization-support filter ===")
    pass2 = build_kernel_filter_pass(
        pass1,
        instances=instances,
        kernel_matrix=kernel_matrix,
    )
    pass2_markdown = _render_kernel_filter_pass_markdown(pass2)
    print(pass2_markdown)
    save_snapshot("rank-pass2-kernel-filtered", pass2, DUMP_DIR)
    (reports_dir / "rank-pass2-kernel-filtered.md").write_text(pass2_markdown)
    print(
        f"Written to {DUMP_DIR / 'rank-pass2-kernel-filtered-latest.json'} "
        "and dump/reports/rank-pass2-kernel-filtered.md",
    )

    print(
        "\nPass 3 (rank survivors by metric) not run — held off "
        "2026-08-22 pending active-parameter data (tracking doc §6 gap 2).",
    )
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    """Model<->instance fit + baseline comparison report (see model_instance_fit.py).

    Fetches all three underlying data sources live (cache-backed, same as
    ``pipeline``) rather than reading a possibly differently-scoped prior
    snapshot, then writes a standalone Markdown report under
    ``dump/reports/`` — 2026-08-19, user clarification: a new document to
    compare by hand, not an edit to the existing research docs.
    """
    aws_args = argparse.Namespace(
        region=args.region,
        instance_types=None,
        family=None,
        config=args.aws_config,
        refresh=args.refresh,
    )
    aws_targets = _resolve_aws_targets(aws_args)
    if aws_targets is None:
        return 1
    instance_types, families = aws_targets
    try:
        instances = fetch_combined_pricing(
            instance_types,
            families=families,
            region=args.region,
            force_refresh=args.refresh,
        )
    except PricingFetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    compare_inputs = _resolve_compare_inputs(args)
    if compare_inputs is None:
        return 1
    model_sizes, comparison_targets, kernel_matrix, gpu_hardware_specs = compare_inputs

    try:
        raw_benchmarks = fetch_benchmarks(
            token=args.openrouter_token,
            force_refresh=args.refresh,
        )
    except (OpenRouterAuthError, OpenRouterFetchError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    benchmarks = filter_by_type(
        raw_benchmarks,
        source="openrouter",
        benchmark_type=args.benchmark_type,
    )

    candidate_slugs = discover_candidate_slugs(benchmarks, model_sizes)
    print(
        f"Auto-discovered {len(candidate_slugs)} candidate families with "
        f"OpenRouter coverage: {', '.join(candidate_slugs) or '(none)'}\n",
    )

    dataset = ComparisonDataset(
        model_sizes=model_sizes,
        instances=instances,
        benchmarks=benchmarks,
        kernel_matrix=kernel_matrix,
        gpu_hardware_specs=gpu_hardware_specs,
    )
    comparisons = build_baseline_comparisons(
        baseline_slugs=comparison_targets.baseline,
        candidate_slugs=candidate_slugs,
        dataset=dataset,
    )
    instance_report = build_instance_reports(
        candidate_slugs=candidate_slugs,
        dataset=dataset,
    )

    report_text = _render_compare_report(
        comparisons,
        instance_report,
        args.benchmark_type,
    )
    print(report_text)

    report_path = DUMP_DIR / "reports" / "model-instance-fit-report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text)
    print(f"\nWritten to {report_path}")
    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    """Run AWS pricing, HF sizes, and OpenRouter benchmarks in one pass.

    Single entry point over the three data sources this project connects —
    2026-08-19, user request: a way to see all of it together instead of
    invoking three subcommands by hand. Each phase reuses its standalone
    subcommand's own handler verbatim, so it still saves its own snapshot
    under ``dump/`` and prints its own diff exactly as running it alone
    would; this just sequences all three off the default config files and
    reports one consolidated exit status.
    """
    print("=== AWS pricing (EC2 + SageMaker, combined) ===")
    aws_args = argparse.Namespace(
        region=args.region,
        instance_types=None,
        family=None,
        config=args.aws_config,
        refresh=args.refresh,
    )
    aws_targets = _resolve_aws_targets(aws_args)
    aws_status = 1
    if aws_targets is not None:
        instance_types, families = aws_targets
        aws_status = _run_aws_pricing_combined(aws_args, instance_types, families)

    print("\n=== HuggingFace model sizes ===")
    hf_args = argparse.Namespace(
        repo_ids=[],
        collection=None,
        author=None,
        config=args.hf_config,
        token=args.hf_token,
        refresh=args.refresh,
    )
    hf_status = _run_hf_size(hf_args)

    print("\n=== OpenRouter benchmarks ===")
    bench_args = argparse.Namespace(
        models=[],
        config=args.benchmark_config,
        source=args.source,
        benchmark_type=args.benchmark_type,
        token=args.openrouter_token,
        refresh=args.refresh,
    )
    bench_status = _run_benchmarks(bench_args)

    statuses = {
        "aws-pricing": aws_status,
        "hf-size": hf_status,
        "benchmarks": bench_status,
    }
    print("\n=== Pipeline summary ===")
    for label, status in statuses.items():
        print(f"  {label}: {'ok' if status == 0 else 'FAILED'}")

    return 0 if all(status == 0 for status in statuses.values()) else 1


def _run_quant_registry(args: argparse.Namespace) -> int:
    """Fetch vLLM's quantization-method registry and diff it against the last fetch.

    See ``vllm_registry.py``'s module docstring: this is a change-detection
    signal for ``config/kernel_support_matrix.yaml``, not a replacement for
    it — writes to its own file (``--out``), never the real matrix, per the
    user's explicit "let's see how stable this is... before directly making
    use of it" (2026-08-19).
    """
    try:
        snapshot = fetch_quant_registry(ref=args.ref)
    except VllmRegistryFetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    previous_methods: set[str] = set()
    if args.out.exists():
        previous = yaml.safe_load(args.out.read_text()) or {}
        previous_methods = set(previous.get("methods") or [])

    print(
        f"vLLM quantization registry @ {snapshot.ref} ({len(snapshot.methods)} methods):",
    )
    for method in snapshot.methods:
        flag = " (deprecated)" if method in snapshot.deprecated_methods else ""
        print(f"  {method}{flag}")

    if args.out.exists():
        current_methods = set(snapshot.methods)
        added = sorted(current_methods - previous_methods)
        removed = sorted(previous_methods - current_methods)
        if added or removed:
            print("\nChanged since last fetch:")
            for method in added:
                print(f"  + added: {method}")
            for method in removed:
                print(f"  - removed: {method}")
        else:
            print("\nNo change since last fetch.")

    write_registry_yaml(snapshot, args.out)
    print(f"\nWritten to {args.out}")
    return 0


def _add_aws_pricing_subparser(subparsers: _SubParsers) -> None:
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


def _add_hf_size_subparser(subparsers: _SubParsers) -> None:
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
    hf_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass the local per-model-size cache.",
    )
    hf_parser.set_defaults(handler=_run_hf_size)


def _add_benchmarks_subparser(subparsers: _SubParsers) -> None:
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


def _add_pipeline_subparser(subparsers: _SubParsers) -> None:
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help=(
            "Run AWS pricing + HF sizes + OpenRouter benchmarks in one pass, "
            "off the default configs."
        ),
    )
    pipeline_parser.add_argument("--region", default="us-east-1")
    pipeline_parser.add_argument(
        "--aws-config",
        type=Path,
        default=Path("config/aws_instances.yaml"),
    )
    pipeline_parser.add_argument(
        "--hf-config",
        type=Path,
        default=Path("config/hf_models.yaml"),
    )
    pipeline_parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=Path("config/benchmark_models.yaml"),
    )
    pipeline_parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace access token.",
    )
    pipeline_parser.add_argument(
        "--openrouter-token",
        default=None,
        help="OpenRouter API token (defaults to OPENROUTER_API_TOKEN env var).",
    )
    pipeline_parser.add_argument(
        "--source",
        choices=["artificial-analysis", "design-arena", "openrouter"],
        default=None,
    )
    pipeline_parser.add_argument(
        "--benchmark-type",
        default=None,
        help='openrouter-native rows only, e.g. "tau_bench_verified_airline"',
    )
    pipeline_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass every local cache (AWS offer files, Vantage, HF sizes, OpenRouter).",
    )
    pipeline_parser.set_defaults(handler=_run_pipeline)


def _add_compare_subparser(subparsers: _SubParsers) -> None:
    compare_parser = subparsers.add_parser(
        "compare",
        help=(
            "Model<->instance fit report: candidates beating a baseline, "
            "plus best-fit candidates per AWS instance."
        ),
    )
    compare_parser.add_argument("--region", default="us-east-1")
    compare_parser.add_argument(
        "--aws-config",
        type=Path,
        default=Path("config/aws_instances.yaml"),
    )
    compare_parser.add_argument(
        "--hf-config",
        type=Path,
        default=Path("config/hf_models.yaml"),
    )
    compare_parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=Path("config/benchmark_models.yaml"),
    )
    compare_parser.add_argument(
        "--kernel-config",
        type=Path,
        default=Path("config/kernel_support_matrix.yaml"),
        help="SM x format kernel-support matrix, see the file's own comments.",
    )
    compare_parser.add_argument(
        "--gpu-specs-config",
        type=Path,
        default=Path("config/gpu_hardware_specs.yaml"),
        help="Per-family GPU memory/bandwidth/NVLink specs, see the file's own comments.",
    )
    compare_parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace access token.",
    )
    compare_parser.add_argument(
        "--openrouter-token",
        default=None,
        help="OpenRouter API token (defaults to OPENROUTER_API_TOKEN env var).",
    )
    compare_parser.add_argument(
        "--benchmark-type",
        default="tau_bench_verified_airline",
        help="The one apples-to-apples metric this report compares candidates on.",
    )
    compare_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass every local cache (AWS offer files, Vantage, HF sizes, OpenRouter).",
    )
    compare_parser.set_defaults(handler=_run_compare)


def _add_rank_subparser(subparsers: _SubParsers) -> None:
    rank_parser = subparsers.add_parser(
        "rank",
        help=(
            "3-pass model<->instance ranking pipeline: capacity filter, then "
            "kernel-support filter, per instance (pass 3 ranking not yet built)."
        ),
    )
    rank_parser.add_argument("--region", default="us-east-1")
    rank_parser.add_argument(
        "--aws-config",
        type=Path,
        default=Path("config/aws_instances.yaml"),
    )
    rank_parser.add_argument(
        "--hf-config",
        type=Path,
        default=Path("config/hf_models.yaml"),
    )
    rank_parser.add_argument(
        "--kernel-config",
        type=Path,
        default=Path("config/kernel_support_matrix.yaml"),
        help="SM x format kernel-support matrix, see the file's own comments.",
    )
    rank_parser.add_argument(
        "--gpu-specs-config",
        type=Path,
        default=Path("config/gpu_hardware_specs.yaml"),
        help="Per-family GPU memory/bandwidth/NVLink specs, see the file's own comments.",
    )
    rank_parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace access token.",
    )
    rank_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass every local cache (AWS offer files, Vantage, HF sizes).",
    )
    rank_parser.set_defaults(handler=_run_rank)


def _add_quant_registry_subparser(subparsers: _SubParsers) -> None:
    registry_parser = subparsers.add_parser(
        "quant-registry",
        help=(
            "Fetch vLLM's own quantization-method registry (GitHub API) -- a "
            "change-detection signal for config/kernel_support_matrix.yaml, "
            "not a support-level source of truth. See vllm_registry.py."
        ),
    )
    registry_parser.add_argument(
        "--ref",
        default="main",
        help="vLLM git ref (branch/tag/commit) to fetch from.",
    )
    registry_parser.add_argument(
        "--out",
        type=Path,
        default=Path("config/kernel_support_matrix_vllm.yaml"),
        help=(
            "Standalone, checked-in YAML file this writes to -- kept separate from "
            "config/kernel_support_matrix.yaml, which stays hand-verified-only."
        ),
    )
    registry_parser.set_defaults(handler=_run_quant_registry)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-hosting-data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_aws_pricing_subparser(subparsers)
    _add_hf_size_subparser(subparsers)
    _add_benchmarks_subparser(subparsers)
    _add_pipeline_subparser(subparsers)
    _add_compare_subparser(subparsers)
    _add_rank_subparser(subparsers)
    _add_quant_registry_subparser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the matching subcommand handler."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
