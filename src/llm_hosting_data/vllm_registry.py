"""Fetch vLLM's own quantization-method registry via the GitHub REST API.

Evaluates "Approach B" from ``documents/fetch_info.md``: querying vLLM's
source directly for which quantization formats it currently registers, as
one candidate signal for keeping ``config/kernel_support_matrix.yaml``
current — 2026-08-19, user request: build a pipeline around this, writing
to its own separate output file first ("let's see how stable the approach
is... before directly making use of it") rather than feeding the trusted
matrix directly. That file is ``config/kernel_support_matrix_vllm.yaml``
(2026-08-20, user request — moved from a gitignored ``dump/`` snapshot to a
checked-in ``config/`` file specifically so it sits next to, and stays
git-diffable against, the hand-verified matrix; ``config/kernel_support_matrix.yaml``
itself stays reserved for manually-verified sources only).

Confirmed live 2026-08-19 that ``fetch_info.md``'s own example output is
already stale relative to the real vLLM ``main`` branch: the live registry
has six online-quant shorthand entries (``fp8_per_tensor``,
``fp8_per_block``, ...) the doc's example doesn't show, and is *missing*
``bitsandbytes``, which the doc's example claims is registered. That's
exactly the kind of drift this module exists to observe.

This only covers ``fetch_info.md``'s "Approach B, part A" (the registry
list) — not part B (per-file SM-gate assertions in
``fp8.py``/``modelopt.py``/``moe_wna16.py``). That part is structurally
harder to extract reliably (arbitrary Python source, not a clean
``Literal[...]`` list) and is out of scope for this first pass.

Does **not** feed ``config/kernel_support_matrix.yaml`` automatically, and
doesn't claim to know the native/fallback/blocked support *level* for any
of these formats — see ``fetch_info.md``'s own §3: source-registry
membership can't detect silent runtime fallbacks, missing PTX compilation
flags, or SM gates that live in a delegated kernel library outside vLLM's
own repo. This is a change-detection signal (did the registry gain/lose a
method since it was last checked), not a support-level source of truth.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import requests
import yaml

if TYPE_CHECKING:
    from pathlib import Path

_REGISTRY_FILE_PATH = "vllm/model_executor/layers/quantization/__init__.py"
_CONTENTS_URL = (
    f"https://api.github.com/repos/vllm-project/vllm/contents/{_REGISTRY_FILE_PATH}"
)
_REQUEST_TIMEOUT_SECONDS = 30

_METHODS_RE = re.compile(r"QuantizationMethods\s*=\s*Literal\[(.*?)\]", re.DOTALL)
_DEPRECATED_RE = re.compile(
    r"DEPRECATED_QUANTIZATION_METHODS\s*=\s*\[(.*?)\]",
    re.DOTALL,
)
_QUOTED_RE = re.compile(r"""["'](.+?)["']""")


class VllmRegistryFetchError(RuntimeError):
    """Raised when vLLM's registry file can't be fetched or parsed."""


@dataclass(frozen=True)
class QuantRegistrySnapshot:
    """One fetch of vLLM's registered-quantization-method list."""

    ref: str
    fetched_at: str
    methods: list[str]
    deprecated_methods: list[str]


def _extract_string_list(source: str, pattern: re.Pattern[str]) -> list[str] | None:
    """Pull the quoted entries out of a matched ``[...]``/``Literal[...]`` block.

    Splits on *lines*, not commas: a standalone comment line between two
    entries (e.g. vLLM's own "# Below are online quant shorthand names.")
    has no comma of its own, so naive comma-splitting glues it to the
    following entry and both get dropped as "starts with #" -- caught by a
    failing test, not by inspection. Line-based splitting plus a per-line
    quoted-string regex sidesteps that regardless of comma placement.
    """
    match = pattern.search(source)
    if not match:
        return None
    entries = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        quoted = _QUOTED_RE.search(stripped)
        if quoted:
            entries.append(quoted.group(1))
    return entries


def fetch_quant_registry(ref: str = "main") -> QuantRegistrySnapshot:
    """Fetch and parse vLLM's ``QuantizationMethods``/``DEPRECATED_QUANTIZATION_METHODS``.

    ``ref`` is any git ref vLLM's GitHub repo accepts (branch, tag, commit
    SHA) — e.g. a specific release tag, to check what changed between
    versions. Unauthenticated GitHub REST API calls are rate-limited to
    60/hour; fine for occasional manual runs, not for tight polling.
    """
    try:
        response = requests.get(
            _CONTENTS_URL,
            params={"ref": ref},
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        msg = f"Failed to fetch {_CONTENTS_URL} (ref={ref})"
        raise VllmRegistryFetchError(msg) from exc

    payload = response.json()
    content_b64 = payload.get("content")
    if not content_b64:
        msg = f"GitHub response had no file content for ref={ref}: {payload}"
        raise VllmRegistryFetchError(msg)
    source = base64.b64decode(content_b64).decode("utf-8")

    methods = _extract_string_list(source, _METHODS_RE)
    if methods is None:
        msg = "Could not find a QuantizationMethods Literal[...] block in the fetched file"
        raise VllmRegistryFetchError(msg)
    deprecated = _extract_string_list(source, _DEPRECATED_RE) or []

    return QuantRegistrySnapshot(
        ref=ref,
        fetched_at=datetime.now(UTC).isoformat(),
        methods=methods,
        deprecated_methods=deprecated,
    )


def write_registry_yaml(snapshot: QuantRegistrySnapshot, out_path: Path) -> None:
    """Write one fetch to a standalone, checked-in YAML file — deliberately not the hand-verified matrix.

    Default ``out_path`` (see ``cli.py``) is
    ``config/kernel_support_matrix_vllm.yaml`` — checked into git, sitting
    next to ``config/kernel_support_matrix.yaml`` for direct comparison, but
    never merged into it. 2026-08-19/20, user request: evaluate stability
    across runs/refs before trusting this as an input to the real matrix,
    which stays reserved for manually-verified sources only.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(
            {
                "ref": snapshot.ref,
                "fetched_at": snapshot.fetched_at,
                "methods": snapshot.methods,
                "deprecated_methods": snapshot.deprecated_methods,
            },
            sort_keys=False,
        ),
    )
