"""Exact one-to-one matching of stim files to sync vsync blocks (gh #8). Synthetic files."""

from __future__ import annotations

import datetime
import pathlib
import types

import h5py
import numpy as np
import pytest

import npc_stim

SYNC_START = datetime.datetime(2026, 1, 1, 10, 0, 0)


def _sync(*blocks: tuple[float, int]) -> types.SimpleNamespace:
    """(first vsync time, n vsyncs) per block."""
    return types.SimpleNamespace(
        start_time=SYNC_START,
        vsync_times_in_blocks=[t0 + np.arange(n) / 60 for t0, n in blocks],
    )


def _stim(tmp: pathlib.Path, name: str, start_sec: int, n_frames: int) -> pathlib.Path:
    path = tmp / f"{name}.hdf5"
    start = SYNC_START + datetime.timedelta(seconds=start_sec)
    with h5py.File(path, "w") as f:
        f["frameIntervals"] = np.full(n_frames - 1, 1 / 60)
        f["startTime"] = start.strftime("%Y%m%d_%H%M%S").encode()
    return path


def test_exact_match(tmp_path) -> None:
    a = _stim(tmp_path, "a", 10, 600)
    b = _stim(tmp_path, "b", 100, 1200)
    matches, errors = npc_stim.get_stim_block_matches([b, a], _sync((11, 600), (101, 1200)))
    assert {i: p and p.name for i, p in matches.items()} == {0: "a.hdf5", 1: "b.hdf5"}
    assert not errors


@pytest.mark.parametrize("diff", [-1, 1, 30])
def test_count_mismatch_is_not_absorbed(tmp_path, diff) -> None:
    a = _stim(tmp_path, "a", 10, 600)
    matches, errors = npc_stim.get_stim_block_matches([a], _sync((11, 600 + diff)))
    assert matches == {0: None}
    (err,) = errors.values()
    assert isinstance(err, IndexError) and "mismatch" in str(err)


def test_block_claimed_twice_matches_none(tmp_path) -> None:
    a = _stim(tmp_path, "a", 10, 600)
    b = _stim(tmp_path, "b", 12, 600)
    matches, errors = npc_stim.get_stim_block_matches([a, b], _sync((11, 600), (500, 600)))
    assert matches == {0: None, 1: None}
    assert len(errors) == 2 and all("claimed by 2" in str(e) for e in errors.values())


def test_count_match_on_other_block_is_not_used(tmp_path) -> None:
    # closest block by start time has the wrong count: not matched to the block with the
    # right count
    a = _stim(tmp_path, "a", 10, 600)
    matches, errors = npc_stim.get_stim_block_matches([a], _sync((11, 601), (500, 600)))
    assert matches == {0: None, 1: None}
    assert "[1]" in str(errors[next(iter(errors))])


def test_frame_times_report_mismatch(tmp_path, monkeypatch) -> None:
    a = _stim(tmp_path, "a", 10, 600)
    b = _stim(tmp_path, "b", 100, 1200)
    sync = _sync((11, 601), (101, 1200))
    monkeypatch.setattr(npc_stim.stim.npc_sync, "get_sync_data", lambda s: s)
    times = npc_stim.get_stim_frame_times(a, b, sync=sync, frame_time_type="vsync")
    assert isinstance(times[a], IndexError)
    assert len(times[b]) == 1200
