"""Match open-weight candidate models to AWS instances, and compare them to a baseline.

Builds two views entirely from data this pipeline already fetches live —
AWS combined pricing/specs, HF model sizes, and OpenRouter benchmarks — no
new data source. See ``documents/llm-hosting-data-pipeline-plan.md`` §5 for
the full design writeup and sourcing.

One thing here is deliberately **not** derived from a live source or from
external data, because it's a genuine policy choice, not a fact: **the
fit-tier headroom convention** (``GPU_MEMORY_UTILIZATION``,
``BEST_FIT_MIN_HEADROOM_FRACTION``). Doc `04`'s own worked examples don't
reduce to one fixed formula — "~13 GiB of 86 GiB usable" is called
comfortable/BEST OVERALL (~15% headroom) and "~2 GiB after overhead" is
called marginal (~2.5% headroom), with doc `04` §6 itself calling the whole
thing "a planning heuristic, not a hard limit." These two numbers encode the
convention picked with the user 2026-08-19 (vLLM's own
``--gpu-memory-utilization 0.9`` default, ≥15% of the usable budget left
over to call something "best fit" — the threshold doc `04`'s own 122B-on-g7e
"BEST OVERALL" call sits right at).

Two other things are static facts, not policy — both live in **external
YAML under ``config/``**, not Python constants (2026-08-19, user request,
applied consistently across this package: "follow the pattern of
externalizing... static mapping data" wherever it appears, not just where
first asked):

1. **Per-GPU nameplate memory** (``config/gpu_hardware_specs.yaml``, loaded
   via ``config.load_gpu_hardware_specs()`` and passed in explicitly as
   part of :class:`ComparisonDataset`). AWS's own ``gpu_memory`` attribute
   is usually the instance *total* (count x per-GPU size), but is confirmed
   live (2026-08-19) to be wrong for ``p5en.48xlarge``: it reports a single
   GPU's memory ("141 GB HBM3e") even though ``gpu`` correctly says 8,
   while every other multi-GPU family reports the true total. Computing the
   total from ``gpu`` (always reliable) x this table — sourced from doc `04`
   §3's own "Memory/GPU" pricing column — sidesteps that bug instead of
   needing to special-case one instance type's string. The same YAML file
   also carries the bandwidth/NVLink specs ``gpu_reference.py`` merges into
   the GPU catalog — one shared source of truth for both, previously two
   independent tables that could drift apart.

2. **Kernel/format support per GPU architecture**
   (``config/kernel_support_matrix.yaml``, loaded via
   ``config.load_kernel_support_matrix()``) — 2026-08-19, user feedback: raw
   capacity alone isn't enough of an elimination step -- "does the hardware
   and kernel support exist for the given quantized variant on that
   hardware" also has to pass. Unlike GPU hardware specs, this one was
   flagged as likely to need updating as kernels mature ("hardware support
   anyway is not going to change [but kernel support might, so] keep it as
   external metadata... i'll try to build a pipeline that can update this
   file when kernel updates happen") — that's why it also gets its own
   ``compare --kernel-config`` override, where the hardware-specs file
   doesn't. Transcribed from doc `04` §5's own "Quantization by
   architecture" table (SM80/A100 through SM120/RTX PRO 6000 x
   BF16/FP8/INT8/INT4/NVFP4, dense vs MoE where the table itself splits
   them).

Both are passed through explicitly (``instance_capacity()``,
``kernel_support()``, ``cheapest_fit()``, ``build_baseline_comparisons()``,
``build_instance_reports()`` all take them as parameters, bundled into
:class:`ComparisonDataset`) rather than this module reading either file
itself — every function here stays a pure function over explicit inputs,
easy to test without file IO.

A checkpoint's own quantization format is separately inferred from its
repo_id (``_classify_checkpoint_format()``) only from an *explicit* token
(``-FP8``, ``-GPTQ-Int4``, ``-NVFP4``, ...) -- a repo with no such token is
left ``None`` (format unknown) rather than assumed BF16, since default
precision genuinely varies by lab (DeepSeek ships FP8-native by default;
Qwen/Llama ship BF16) and guessing wrong here would silently misclassify
real checkpoints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_hosting_data.instance_naming import family_of

if TYPE_CHECKING:
    from llm_hosting_data.aws_pricing import CombinedInstancePricing
    from llm_hosting_data.hf_models import ModelSize
    from llm_hosting_data.openrouter_benchmarks import BenchmarkEntry

GPU_MEMORY_UTILIZATION = 0.9
BEST_FIT_MIN_HEADROOM_FRACTION = 0.15

_GB_PER_GIB = (
    1.073741824  # decimal GB -> binary GiB -- the "GB/GiB trap" doc 04 documents
)


@dataclass(frozen=True)
class InstanceCapacity:
    """Usable GPU-memory budget for one AWS instance type."""

    instance_type: str
    gpu_count: int
    total_nameplate_gib: float
    usable_gib: float


def instance_capacity(
    instance_type: str,
    gpu_count: str | int | None,
    gpu_hardware_specs: dict[str, dict[str, object]],
) -> InstanceCapacity | None:
    """Compute one instance's usable GPU-memory budget from its family and GPU count.

    ``gpu_hardware_specs`` is the per-family table loaded from
    ``config/gpu_hardware_specs.yaml`` via
    ``config.load_gpu_hardware_specs()`` — see the module docstring's point
    1. Returns ``None`` for a non-GPU instance or an unrecognized family —
    never guesses at a number.
    """
    if not gpu_count:
        return None
    try:
        count = int(gpu_count)
    except (TypeError, ValueError):
        return None
    per_gpu_gb = gpu_hardware_specs.get(family_of(instance_type), {}).get(
        "memory_gb_per_gpu",
    )
    if not isinstance(per_gpu_gb, int | float):
        return None
    total_nameplate_gib = (per_gpu_gb * count) / _GB_PER_GIB
    return InstanceCapacity(
        instance_type=instance_type,
        gpu_count=count,
        total_nameplate_gib=total_nameplate_gib,
        usable_gib=total_nameplate_gib * GPU_MEMORY_UTILIZATION,
    )


def classify_fit(model_gib: float, capacity: InstanceCapacity) -> str | None:
    """Return ``"best"``, ``"tight"``, or ``None`` (does not fit) for one model/instance pair."""
    if model_gib > capacity.usable_gib:
        return None
    headroom_fraction = (capacity.usable_gib - model_gib) / capacity.usable_gib
    if headroom_fraction >= BEST_FIT_MIN_HEADROOM_FRACTION:
        return "best"
    return "tight"


_MOE_ACTIVE_PARAMS_RE = re.compile(r"-a\d+b", re.IGNORECASE)
# Ordered most-specific-first: "-NVFP4" must be checked before a bare "-FP4"
# would ever be added, and GGUF is deliberately unclassified -- it's a
# llama.cpp format, not one doc 04's vLLM-oriented table covers at all.
_FORMAT_TOKEN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"nvfp4|mxfp4", re.IGNORECASE), "nvfp4"),
    (re.compile(r"fp8", re.IGNORECASE), "fp8"),
    (
        re.compile(r"gptq.?int4|awq|w4a16|(?<![a-z])int4(?![a-z])", re.IGNORECASE),
        "int4_w4a16",
    ),
    (re.compile(r"w8a8|(?<![a-z])int8(?![a-z])", re.IGNORECASE), "int8_w8a8"),
    (re.compile(r"bf16", re.IGNORECASE), "bf16"),
]


def _classify_checkpoint_format(repo_id: str) -> str | None:
    """Infer one checkpoint's kernel-support format key from an explicit repo_id token.

    Returns ``None`` (format unknown) rather than guessing when no
    recognized token is present — see the module docstring's point 3 for
    why "no suffix" can't safely default to any one format.
    """
    is_moe = bool(_MOE_ACTIVE_PARAMS_RE.search(repo_id))
    for pattern, base_format in _FORMAT_TOKEN_PATTERNS:
        if not pattern.search(repo_id):
            continue
        if base_format == "fp8":
            return "fp8_moe" if is_moe else "fp8_dense"
        if base_format == "nvfp4":
            return "nvfp4_moe" if is_moe else "nvfp4_dense"
        return base_format
    return None


def kernel_support(
    repo_id: str,
    compute_capability: str | None,
    kernel_matrix: dict[str, dict[str, str]],
) -> str:
    """Return ``"native"``/``"fallback"``/``"blocked"``/``"unknown"`` for one checkpoint on one GPU.

    ``kernel_matrix`` is the SM -> format -> level table loaded from
    ``config/kernel_support_matrix.yaml`` via
    ``config.load_kernel_support_matrix()`` — see the module docstring's
    point 3 for why it's external data, not a constant here. ``"unknown"``
    covers both an unrecognized checkpoint format and a missing/unrecognized
    ``compute_capability`` (e.g. Vantage doesn't have it for that instance
    type) — treated as includable-but-unverified by callers, never as
    "blocked."
    """
    if compute_capability is None:
        return "unknown"
    format_key = _classify_checkpoint_format(repo_id)
    if format_key is None:
        return "unknown"
    return kernel_matrix.get(compute_capability, {}).get(format_key, "unknown")


_NORMALIZE_RE = re.compile(r"[^a-z0-9]")


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", text.lower())


def candidate_family_key(benchmark_slug: str) -> str:
    """Return the normalized token matched against HF repo_ids for one candidate slug.

    Uses the slug's last ``/``-separated segment (the model name, not the
    org) — e.g. ``"qwen/qwen3.5-397b-a17b"`` -> ``"qwen35397ba17b"`` — so the
    same key matches every quantization checkpoint of that family regardless
    of which HF org hosts it (Qwen's own org for BF16/FP8/GPTQ-Int4,
    ``nvidia/`` for NVFP4 — see the tracking doc's NVFP4 findings).
    """
    return _normalize(benchmark_slug.rsplit("/", 1)[-1])


def hf_repo_matches_candidate(repo_id: str, benchmark_slug: str) -> bool:
    """Whether an HF repo_id belongs to one candidate family's checkpoints."""
    return candidate_family_key(benchmark_slug) in _normalize(repo_id)


_TRAILING_DATE_RE = re.compile(r"-\d{8}$")


def _strip_release_date(model_permaslug: str) -> str:
    """Strip an OpenRouter release-date suffix (e.g. ``-20260216``), if present.

    Not every permaslug has one (``"openai/gpt-4o"``, ``"deepseek/deepseek-chat-v3"``),
    and some have a non-date trailing token that must *not* be stripped
    (``"deepseek/deepseek-v3.1-terminus"`` is a distinct named checkpoint, not
    a dated re-eval) — only a trailing, exactly-8-digit block is removed.
    """
    return _TRAILING_DATE_RE.sub("", model_permaslug)


def discover_candidate_slugs(
    benchmarks: list[BenchmarkEntry],
    model_sizes: list[ModelSize],
) -> list[str]:
    """Auto-derive candidate family slugs — every OpenRouter family with a matching HF repo.

    No manual list: for every distinct ``openrouter``-source
    ``model_permaslug`` (release-date suffix stripped), keep it only if it
    matches at least one repo already in the resolved HF candidate pool
    (``config/hf_models.yaml``). This is how "every model in
    ``hf_models.yaml``" (2026-08-19, user request) gets checked against
    OpenRouter automatically, without a second hand-typed list that has to
    stay in sync as collections change — only the baseline (closed-source,
    not part of ``hf_models.yaml`` at all) still needs a manual slug.
    """
    stable_slugs = {
        _strip_release_date(entry.model_permaslug)
        for entry in benchmarks
        if entry.source == "openrouter"
    }
    return sorted(
        slug
        for slug in stable_slugs
        if any(hf_repo_matches_candidate(size.repo_id, slug) for size in model_sizes)
    )


@dataclass(frozen=True)
class HostingFit:
    """Cheapest instance that tight-fits and cheapest that best-fits one model.

    ``*_kernel_support`` is the level (see :func:`kernel_support`) for the
    checkpoint on the *chosen* instance's GPU architecture — an instance
    where support is ``"blocked"`` is never chosen at all (skipped during
    the search, not just flagged), so this field is always one of
    ``"native"``/``"fallback"``/``"unknown"`` when the corresponding
    instance is not ``None``.
    """

    tight_fit_instance: str | None
    tight_fit_usd_per_hour: float | None
    tight_fit_kernel_support: str | None
    best_fit_instance: str | None
    best_fit_usd_per_hour: float | None
    best_fit_kernel_support: str | None


def cheapest_fit(
    repo_id: str,
    model_gib: float,
    instances: list[CombinedInstancePricing],
    kernel_matrix: dict[str, dict[str, str]],
    gpu_hardware_specs: dict[str, dict[str, object]],
) -> HostingFit:
    """Find the cheapest EC2-priced instance that fits ``model_gib``, per tier.

    Three-step elimination per instance, matching doc `04`'s own process
    (2026-08-19, user feedback): raw GPU-memory capacity
    (:func:`instance_capacity`/:func:`classify_fit`), then kernel/format
    support (:func:`kernel_support` — an instance where this checkpoint's
    format is ``"blocked"`` on that GPU architecture is skipped entirely,
    same as a capacity or price miss), *then* price to pick the cheapest
    survivor. "Tight" tracks the cheapest instance in *either* fit tier
    (best fit is also a tight fit); "best" tracks only the cheapest among
    best-fit instances.
    """
    tight: tuple[str, float, str] | None = None
    best: tuple[str, float, str] | None = None
    for instance in instances:
        price = instance.ec2_usd_per_hour
        if price is None:
            continue
        capacity = instance_capacity(
            instance.instance_type,
            instance.gpu,
            gpu_hardware_specs,
        )
        if capacity is None:
            continue
        tier = classify_fit(model_gib, capacity)
        if tier is None:
            continue
        support = kernel_support(
            repo_id,
            instance.gpu_compute_capability,
            kernel_matrix,
        )
        if support == "blocked":
            continue
        if tight is None or price < tight[1]:
            tight = (instance.instance_type, price, support)
        if tier == "best" and (best is None or price < best[1]):
            best = (instance.instance_type, price, support)
    return HostingFit(
        tight_fit_instance=tight[0] if tight else None,
        tight_fit_usd_per_hour=tight[1] if tight else None,
        tight_fit_kernel_support=tight[2] if tight else None,
        best_fit_instance=best[0] if best else None,
        best_fit_usd_per_hour=best[1] if best else None,
        best_fit_kernel_support=best[2] if best else None,
    )


@dataclass(frozen=True)
class CandidateCheckpointFit:
    """One HF checkpoint of a candidate family, and where it can be hosted."""

    repo_id: str
    total_gib: float
    hosting: HostingFit


@dataclass(frozen=True)
class CandidateComparison:
    """One candidate family versus one baseline, on one benchmark."""

    candidate_slug: str
    candidate_accuracy: float | None
    baseline_slug: str
    baseline_accuracy: float | None
    beats_baseline: bool | None
    checkpoints: list[CandidateCheckpointFit]


def _best_accuracy(entries: list[BenchmarkEntry], slug: str) -> float | None:
    lowered = slug.lower()
    accuracies = [
        entry.fields["accuracy"]
        for entry in entries
        if lowered in entry.model_permaslug.lower()
        and isinstance(entry.fields.get("accuracy"), int | float)
    ]
    return max(accuracies) if accuracies else None


@dataclass(frozen=True)
class ComparisonDataset:
    """The shared inputs both reports are built from — one fetch, two views.

    2026-08-19, user feedback: "we build the data for report pattern 1 and
    reuse the data for report pattern 2" — bundling the five inputs every
    report needs (model sizes, instance pricing/specs, benchmark scores,
    kernel-support matrix, GPU hardware specs) into one object is what makes
    that reuse explicit, rather than each report function taking its own
    loose copy of the same five parameters.
    """

    model_sizes: list[ModelSize]
    instances: list[CombinedInstancePricing]
    benchmarks: list[BenchmarkEntry]
    kernel_matrix: dict[str, dict[str, str]]
    gpu_hardware_specs: dict[str, dict[str, object]]


def build_baseline_comparisons(
    *,
    baseline_slugs: list[str],
    candidate_slugs: list[str],
    dataset: ComparisonDataset,
) -> list[CandidateComparison]:
    """Report 1: candidates that meet or beat each baseline, with hosting requirements.

    One row per (baseline, candidate) pair whose candidate accuracy is known
    and ``>=`` the baseline's. A candidate with no benchmark entry for the
    filtered-in benchmark type is skipped here, not scored as 0 — see
    ``beats_baseline`` semantics: ``None`` means "not comparable," never
    treated as "loses."
    """
    results: list[CandidateComparison] = []
    for baseline_slug in baseline_slugs:
        baseline_accuracy = _best_accuracy(dataset.benchmarks, baseline_slug)
        for candidate_slug in candidate_slugs:
            candidate_accuracy = _best_accuracy(dataset.benchmarks, candidate_slug)
            beats_baseline = (
                candidate_accuracy >= baseline_accuracy
                if candidate_accuracy is not None and baseline_accuracy is not None
                else None
            )
            if beats_baseline is not True:
                continue
            checkpoints = [
                CandidateCheckpointFit(
                    repo_id=size.repo_id,
                    total_gib=size.total_gib,
                    hosting=cheapest_fit(
                        size.repo_id,
                        size.total_gib,
                        dataset.instances,
                        dataset.kernel_matrix,
                        dataset.gpu_hardware_specs,
                    ),
                )
                for size in dataset.model_sizes
                if hf_repo_matches_candidate(size.repo_id, candidate_slug)
            ]
            results.append(
                CandidateComparison(
                    candidate_slug=candidate_slug,
                    candidate_accuracy=candidate_accuracy,
                    baseline_slug=baseline_slug,
                    baseline_accuracy=baseline_accuracy,
                    beats_baseline=beats_baseline,
                    checkpoints=checkpoints,
                ),
            )
    return results


@dataclass(frozen=True)
class InstanceCandidateFit:
    """One candidate checkpoint that fits a given instance, for the per-instance view."""

    repo_id: str
    total_gib: float
    fit_tier: str
    kernel_support: str
    candidate_slug: str | None
    accuracy: float | None


def build_instance_reports(
    *,
    candidate_slugs: list[str],
    dataset: ComparisonDataset,
) -> dict[str, list[InstanceCandidateFit]]:
    """Report 2: for each AWS instance, every candidate checkpoint that fits it.

    "Fits" means capacity *and* kernel support both pass — a checkpoint
    whose format is ``"blocked"`` on that instance's GPU architecture (see
    :func:`kernel_support`) is excluded here, not just flagged, the same
    elimination :func:`cheapest_fit` applies for Report 1. Sorted scored
    candidates first (by accuracy descending, best-fit tier ahead of
    tight-fit at equal accuracy), then unscored checkpoints — included, not
    dropped, so a family with no OpenRouter coverage is still visible, just
    after the comparable ones. The inverse view of
    :func:`build_baseline_comparisons`.
    """
    accuracy_by_slug = {
        slug: _best_accuracy(dataset.benchmarks, slug) for slug in candidate_slugs
    }

    def _matching_slug(repo_id: str) -> str | None:
        return next(
            (
                slug
                for slug in candidate_slugs
                if hf_repo_matches_candidate(repo_id, slug)
            ),
            None,
        )

    report: dict[str, list[InstanceCandidateFit]] = {}
    for instance in dataset.instances:
        capacity = instance_capacity(
            instance.instance_type,
            instance.gpu,
            dataset.gpu_hardware_specs,
        )
        if capacity is None:
            continue
        fits: list[InstanceCandidateFit] = []
        for size in dataset.model_sizes:
            tier = classify_fit(size.total_gib, capacity)
            if tier is None:
                continue
            support = kernel_support(
                size.repo_id,
                instance.gpu_compute_capability,
                dataset.kernel_matrix,
            )
            if support == "blocked":
                continue
            slug = _matching_slug(size.repo_id)
            fits.append(
                InstanceCandidateFit(
                    repo_id=size.repo_id,
                    total_gib=size.total_gib,
                    fit_tier=tier,
                    kernel_support=support,
                    candidate_slug=slug,
                    accuracy=accuracy_by_slug.get(slug) if slug else None,
                ),
            )
        fits.sort(
            key=lambda fit: (
                fit.accuracy is None,  # scored candidates first
                fit.fit_tier != "best",
                -(fit.accuracy if fit.accuracy is not None else 0.0),
            ),
        )
        report[instance.instance_type] = fits
    return report
