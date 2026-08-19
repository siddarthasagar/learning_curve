"""Write timestamped JSON snapshots and diff against the previous run.

Both AWS pricing and HuggingFace model availability are moving targets, so a
refresh should record what changed, not just overwrite the last answer. But
"moving target" doesn't mean "changes every run" — most re-runs of the same
query return identical data, so ``save_snapshot`` skips writing a new
timestamped file when the content matches the last one (only ``-latest.json``
exists to diff against, no dated file), and keeps only the most recent
``keep`` timestamped snapshots per name so re-running this regularly doesn't
pile up an ever-growing pile of near-duplicate files in ``dump/``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

DEFAULT_SNAPSHOT_KEEP = 2


def to_jsonable(value: Any) -> Any:  # noqa: ANN401
    """Recursively convert dataclasses (and containers of them) into plain JSON data."""
    if is_dataclass(value) and not isinstance(value, type):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def save_snapshot(
    name: str,
    data: Any,  # noqa: ANN401
    out_dir: Path,
    *,
    keep: int = DEFAULT_SNAPSHOT_KEEP,
) -> Path:
    """Write ``data`` as a timestamped JSON snapshot and refresh ``<name>-latest.json``.

    A no-op beyond refreshing ``<name>-latest.json`` (mtime aside) if the
    content is identical to the last run — no new dated file, since nothing
    changed. Otherwise writes a new dated file and prunes older dated
    snapshots for this ``name`` down to the ``keep`` most recent.

    Returns the path written: the new timestamped snapshot, or the existing
    ``<name>-latest.json`` when the content was unchanged.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = to_jsonable(data)
    serialized = json.dumps(payload, indent=2, sort_keys=True)

    latest_path = out_dir / f"{name}-latest.json"
    if latest_path.exists() and latest_path.read_text() == serialized:
        return latest_path

    timestamp = datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)
    snapshot_path = out_dir / f"{name}-{timestamp}.json"
    snapshot_path.write_text(serialized)
    latest_path.write_text(serialized)

    _prune_old_snapshots(name, out_dir, keep=keep)
    return snapshot_path


def _prune_old_snapshots(name: str, out_dir: Path, *, keep: int) -> None:
    latest_name = f"{name}-latest.json"
    dated_snapshots = sorted(
        path for path in out_dir.glob(f"{name}-*.json") if path.name != latest_name
    )
    stale = dated_snapshots[:-keep] if keep > 0 else dated_snapshots
    for path in stale:
        path.unlink()


def load_latest_snapshot(name: str, out_dir: Path) -> list[Any] | None:
    """Load the most recent snapshot for ``name``, or ``None`` if none exists yet."""
    latest_path = out_dir / f"{name}-latest.json"
    if not latest_path.exists():
        return None
    return json.loads(latest_path.read_text())


def diff_by_key(
    old_items: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
    key: str,
) -> dict[str, list[Any]]:
    """Compare two lists of dicts by a shared identifying field.

    Returns ``{"added": [...], "removed": [...], "changed": [...]}``, where
    each "changed" entry is ``{"key": ..., "old": {...}, "new": {...}}``.
    """
    old_by_key = {item[key]: item for item in old_items}
    new_by_key = {item[key]: item for item in new_items}

    added = [
        item for item_key, item in new_by_key.items() if item_key not in old_by_key
    ]
    removed = [
        item for item_key, item in old_by_key.items() if item_key not in new_by_key
    ]
    changed = [
        {"key": item_key, "old": old_by_key[item_key], "new": new_by_key[item_key]}
        for item_key in old_by_key.keys() & new_by_key.keys()
        if old_by_key[item_key] != new_by_key[item_key]
    ]
    return {"added": added, "removed": removed, "changed": changed}
