"""Tests for the model<->instance fit and baseline-comparison report logic.

Stays offline: builds small hand-constructed CombinedInstancePricing/
ModelSize/BenchmarkEntry fixtures rather than live data -- the live
p4d/p4de/p5en cases this module's design is grounded in are documented (and
were verified live 2026-08-19) in
documents/llm-hosting-data-pipeline-plan.md §5.
"""

from __future__ import annotations

from llm_hosting_data import model_instance_fit as fit
from llm_hosting_data.aws_pricing import CombinedInstancePricing
from llm_hosting_data.hf_models import ModelSize
from llm_hosting_data.openrouter_benchmarks import BenchmarkEntry

_BYTES_PER_GIB = 1024**3

# Small subsets of config/kernel_support_matrix.yaml's and
# config/gpu_hardware_specs.yaml's real cells -- just what these tests
# exercise, kept local so editing either checked-in YAML can't silently
# change what this file is asserting.
_KERNEL_MATRIX: dict[str, dict[str, str]] = {
    "SM80": {
        "bf16": "native",
        "int4_w4a16": "native",
        "nvfp4_dense": "fallback",
        "nvfp4_moe": "fallback",
    },
    "SM90": {
        "bf16": "native",
    },
    "SM120": {
        "int8_w8a8": "blocked",
    },
}

_GPU_HARDWARE_SPECS: dict[str, dict[str, object]] = {
    "g6e": {"memory_gb_per_gpu": 48.0},
    "g7e": {"memory_gb_per_gpu": 96.0},
    "p4d": {"memory_gb_per_gpu": 40.0},
    "p4de": {"memory_gb_per_gpu": 80.0},
    "p5": {"memory_gb_per_gpu": 80.0},
    "p5en": {"memory_gb_per_gpu": 141.0},
}


def _model_size(repo_id: str, total_gib: float) -> ModelSize:
    return ModelSize(
        repo_id=repo_id,
        revision="main",
        total_bytes=round(total_gib * _BYTES_PER_GIB),
        file_count=1,
    )


def _instance(
    instance_type: str,
    gpu: str,
    usd: float,
    *,
    compute_capability: str | None = None,
) -> CombinedInstancePricing:
    return CombinedInstancePricing(
        instance_type=instance_type,
        region="us-east-1",
        vcpu=None,
        memory=None,
        gpu=gpu,
        gpu_model=None,
        gpu_architecture=None,
        gpu_compute_capability=compute_capability,
        gpu_memory=None,
        gpu_memory_bandwidth_gbps=None,
        gpu_nvlink_generation=None,
        gpu_nvlink_bandwidth_gbps=None,
        network_performance=None,
        ec2_usd_per_hour=usd,
        sagemaker_usd_per_hour=None,
        available_on=["ec2"],
    )


def test_instance_capacity_uses_gpu_count_not_the_buggy_p5en_gpu_memory_string() -> (
    None
):
    # p5en.48xlarge: AWS's own gpuMemory attribute reports a single GPU's
    # 141 GB, not the true 8x141=1128 GB total -- confirmed live 2026-08-19.
    # instance_capacity() never reads that string at all, so it isn't affected.
    capacity = fit.instance_capacity("p5en.48xlarge", "8", _GPU_HARDWARE_SPECS)

    assert capacity is not None
    assert capacity.gpu_count == 8
    assert (
        capacity.total_nameplate_gib > 1000
    )  # would be ~131 GiB if it trusted "141 GB"


def test_instance_capacity_returns_none_for_unknown_family_or_missing_gpu() -> None:
    assert fit.instance_capacity("m5.xlarge", "0", _GPU_HARDWARE_SPECS) is None
    assert fit.instance_capacity("m5.xlarge", None, _GPU_HARDWARE_SPECS) is None
    assert (
        fit.instance_capacity("totally-unknown.2xlarge", "1", _GPU_HARDWARE_SPECS)
        is None
    )


def test_classify_fit_tight_vs_best_vs_does_not_fit() -> None:
    capacity = fit.instance_capacity("g7e.2xlarge", "1", _GPU_HARDWARE_SPECS)
    assert capacity is not None  # usable_gib ~= 96/1.073741824 * 0.9 ~= 80.5 GiB

    assert (
        fit.classify_fit(90.0, capacity) is None
    )  # too big even for the raw usable budget
    assert fit.classify_fit(78.0, capacity) == "tight"  # fits, but under 15% headroom
    assert fit.classify_fit(50.0, capacity) == "best"  # comfortably under budget


def test_hf_repo_matches_candidate_across_orgs_and_avoids_neighboring_sizes() -> None:
    slug = "qwen/qwen3.5-397b-a17b"

    assert fit.hf_repo_matches_candidate("Qwen/Qwen3.5-397B-A17B-GPTQ-Int4", slug)
    assert fit.hf_repo_matches_candidate("nvidia/Qwen3.5-397B-A17B-NVFP4-V2", slug)
    assert not fit.hf_repo_matches_candidate("Qwen/Qwen3.5-122B-A10B-GPTQ-Int4", slug)


def test_discover_candidate_slugs_needs_no_manual_list() -> None:
    model_sizes = [
        _model_size("Qwen/Qwen3.5-397B-A17B-GPTQ-Int4", 219.58),
        _model_size("nvidia/Qwen3.5-397B-A17B-NVFP4", 233.99),
        _model_size("Qwen/Qwen3.5-4B", 8.70),  # no OpenRouter coverage at all
    ]
    benchmarks = [
        BenchmarkEntry(
            source="openrouter",
            model_permaslug="qwen/qwen3.5-397b-a17b-20260216",
            display_name="Qwen3.5 397B A17B",
            fields={"accuracy": 0.79},
        ),
        BenchmarkEntry(
            source="openrouter",
            model_permaslug="deepseek/deepseek-v3.1-terminus",  # non-date suffix, not stripped
            display_name="DeepSeek V3.1 Terminus",
            fields={"accuracy": 0.5},
        ),
        BenchmarkEntry(
            source="artificial-analysis",  # wrong source, must be ignored
            model_permaslug="qwen/qwen3.5-397b-a17b",
            display_name="Qwen3.5 397B A17B",
            fields={"intelligence_index": 70},
        ),
    ]

    discovered = fit.discover_candidate_slugs(benchmarks, model_sizes)

    assert discovered == ["qwen/qwen3.5-397b-a17b"]


def test_kernel_support_matches_doc_04_worked_cases() -> None:
    # INT4 W4A16 is native everywhere (doc 04 §5 fact 5).
    assert (
        fit.kernel_support("Qwen/Qwen3.5-27B-GPTQ-Int4", "SM80", _KERNEL_MATRIX)
        == "native"
    )
    # NVFP4 dense on A100 (SM80): no FP4 silicon at all -- falls back to W4A16.
    assert (
        fit.kernel_support("nvidia/Qwen3.5-397B-A17B-NVFP4", "SM80", _KERNEL_MATRIX)
        == "fallback"
    )
    # INT8 W8A8 is the one cell doc 04 marks genuinely blocked -- SM120/RTX PRO 6000.
    assert fit.kernel_support("org/model-Int8", "SM120", _KERNEL_MATRIX) == "blocked"
    # No recognized format token in the repo_id -- unverified, not guessed.
    assert (
        fit.kernel_support("deepseek-ai/DeepSeek-V4-Pro", "SM90", _KERNEL_MATRIX)
        == "unknown"
    )
    # No compute_capability known for the instance -- also unverified.
    assert (
        fit.kernel_support("Qwen/Qwen3.5-27B-GPTQ-Int4", None, _KERNEL_MATRIX)
        == "unknown"
    )


def test_cheapest_fit_prefers_cheaper_instance_and_separates_tiers() -> None:
    instances = [
        _instance("g7e.2xlarge", "1", 3.36),  # best fit for a small model
        _instance("p5.4xlarge", "1", 6.88),  # also fits, but pricier
    ]

    hosting = fit.cheapest_fit(
        "org/model",
        20.0,
        instances,
        _KERNEL_MATRIX,
        _GPU_HARDWARE_SPECS,
    )

    assert hosting.best_fit_instance == "g7e.2xlarge"
    assert hosting.tight_fit_instance == "g7e.2xlarge"


def test_cheapest_fit_excludes_a_blocked_instance_even_when_capacity_fits() -> None:
    # g7e.2xlarge has ample raw capacity for a 20 GiB model, but INT8 W8A8
    # is blocked on SM120 (RTX PRO 6000) per doc 04's own table -- it must
    # never be recommended for this checkpoint, capacity aside.
    instances = [_instance("g7e.2xlarge", "1", 3.36, compute_capability="SM120")]

    hosting = fit.cheapest_fit(
        "org/model-Int8",
        20.0,
        instances,
        _KERNEL_MATRIX,
        _GPU_HARDWARE_SPECS,
    )

    assert hosting.tight_fit_instance is None
    assert hosting.best_fit_instance is None


def test_build_baseline_comparisons_skips_non_winning_and_uncomparable_candidates() -> (
    None
):
    model_sizes = [
        _model_size("Qwen/Qwen3.5-397B-A17B-GPTQ-Int4", 219.58),
    ]
    instances = [_instance("p5.48xlarge", "8", 55.04)]
    benchmarks = [
        BenchmarkEntry(
            source="openrouter",
            model_permaslug="anthropic/claude-sonnet-5-20260630",
            display_name="Claude Sonnet 5",
            fields={"accuracy": 0.77, "benchmark_type": "tau_bench_verified_airline"},
        ),
        BenchmarkEntry(
            source="openrouter",
            model_permaslug="qwen/qwen3.5-397b-a17b-20260216",
            display_name="Qwen3.5 397B A17B",
            fields={"accuracy": 0.79, "benchmark_type": "tau_bench_verified_airline"},
        ),
        BenchmarkEntry(
            source="openrouter",
            model_permaslug="moonshotai/kimi-k3-20260715",
            display_name="Kimi K3",
            fields={"accuracy": None, "benchmark_type": "tau_bench_verified_airline"},
        ),
    ]

    dataset = fit.ComparisonDataset(
        model_sizes=model_sizes,
        instances=instances,
        benchmarks=benchmarks,
        kernel_matrix=_KERNEL_MATRIX,
        gpu_hardware_specs=_GPU_HARDWARE_SPECS,
    )
    comparisons = fit.build_baseline_comparisons(
        baseline_slugs=["anthropic/claude-sonnet-5"],
        candidate_slugs=["qwen/qwen3.5-397b-a17b", "moonshotai/kimi-k3"],
        dataset=dataset,
    )

    assert (
        len(comparisons) == 1
    )  # kimi-k3 has no numeric accuracy -- not "beats", not included
    assert comparisons[0].candidate_slug == "qwen/qwen3.5-397b-a17b"
    assert comparisons[0].beats_baseline is True
    assert len(comparisons[0].checkpoints) == 1
    assert comparisons[0].checkpoints[0].repo_id == "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"


def test_build_instance_reports_sorts_best_fit_first_then_by_accuracy() -> None:
    model_sizes = [
        _model_size("Qwen/Qwen3.5-397B-A17B-GPTQ-Int4", 219.58),
        _model_size("moonshotai/Kimi-K3", 1454.0),
    ]
    instances = [_instance("p5.48xlarge", "8", 55.04)]
    benchmarks = [
        BenchmarkEntry(
            source="openrouter",
            model_permaslug="qwen/qwen3.5-397b-a17b-20260216",
            display_name="Qwen3.5 397B A17B",
            fields={"accuracy": 0.79, "benchmark_type": "tau_bench_verified_airline"},
        ),
    ]
    # Kimi K3 (1,454 GiB) doesn't fit an 8x H100 (640 GiB nameplate) box --
    # only the Qwen checkpoint should show up for this instance.

    dataset = fit.ComparisonDataset(
        model_sizes=model_sizes,
        instances=instances,
        benchmarks=benchmarks,
        kernel_matrix=_KERNEL_MATRIX,
        gpu_hardware_specs=_GPU_HARDWARE_SPECS,
    )
    report = fit.build_instance_reports(
        candidate_slugs=["qwen/qwen3.5-397b-a17b", "moonshotai/kimi-k3"],
        dataset=dataset,
    )

    fits = report["p5.48xlarge"]
    assert [f.repo_id for f in fits] == ["Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"]
    assert fits[0].accuracy == 0.79
