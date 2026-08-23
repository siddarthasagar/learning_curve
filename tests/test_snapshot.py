"""Tests for timestamped snapshotting and diffing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_hosting_data.snapshot import (
    diff_by_key,
    load_latest_snapshot,
    save_snapshot,
    to_jsonable,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class _Item:
    name: str
    value: int


def test_to_jsonable_converts_nested_dataclasses() -> None:
    payload = to_jsonable([_Item(name="a", value=1), _Item(name="b", value=2)])
    assert payload == [{"name": "a", "value": 1}, {"name": "b", "value": 2}]


def test_save_and_load_latest_snapshot_roundtrip(tmp_path: Path) -> None:
    data = [_Item(name="a", value=1)]
    save_snapshot("widgets", data, tmp_path)

    loaded = load_latest_snapshot("widgets", tmp_path)

    assert loaded == [{"name": "a", "value": 1}]


def test_load_latest_snapshot_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_latest_snapshot("missing", tmp_path) is None


def test_save_snapshot_skips_dated_file_when_content_unchanged(tmp_path: Path) -> None:
    data = [_Item(name="a", value=1)]

    save_snapshot("widgets", data, tmp_path)
    latest_after_first = set(tmp_path.glob("widgets-*.json"))
    dated_after_first = set((tmp_path / "backup").glob("widgets-*.json"))

    save_snapshot("widgets", data, tmp_path)
    latest_after_second = set(tmp_path.glob("widgets-*.json"))
    dated_after_second = set((tmp_path / "backup").glob("widgets-*.json"))

    # "widgets-latest.json" lives directly under tmp_path; the one dated
    # file from the first, genuinely-new write lives under tmp_path/backup/
    # -- the second, identical write adds nothing to either.
    assert len(latest_after_first) == 1
    assert len(dated_after_first) == 1
    assert latest_after_second == latest_after_first
    assert dated_after_second == dated_after_first


def test_save_snapshot_prunes_dated_files_beyond_keep(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    for fake_timestamp in ("20200101T000000Z", "20200102T000000Z", "20200103T000000Z"):
        (backup_dir / f"widgets-{fake_timestamp}.json").write_text("[]")
    (tmp_path / "widgets-latest.json").write_text("[]")

    save_snapshot("widgets", [_Item(name="a", value=1)], tmp_path, keep=2)

    dated = sorted(p.name for p in backup_dir.glob("widgets-*.json"))
    assert len(dated) == 2
    assert "widgets-20200101T000000Z.json" not in dated
    assert "widgets-20200102T000000Z.json" not in dated
    assert "widgets-20200103T000000Z.json" in dated


def test_diff_by_key_reports_added_removed_and_changed() -> None:
    old_items = [
        {"id": "a", "price": 1},
        {"id": "b", "price": 2},
        {"id": "c", "price": 3},
    ]
    new_items = [
        {"id": "a", "price": 1},
        {"id": "b", "price": 99},
        {"id": "d", "price": 4},
    ]

    delta = diff_by_key(old_items, new_items, key="id")

    assert delta["added"] == [{"id": "d", "price": 4}]
    assert delta["removed"] == [{"id": "c", "price": 3}]
    assert delta["changed"] == [
        {"key": "b", "old": {"id": "b", "price": 2}, "new": {"id": "b", "price": 99}},
    ]
