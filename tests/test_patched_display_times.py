"""Display times on a sync file with a replayed interval, with and without a `SyncPatch`
(npc_sync >= the `sync-patch` branch). Synthetic sync + stim files."""

from __future__ import annotations

import pathlib

import h5py
import npc_sync
import numpy as np
import pytest
from npc_sync import patch as P

import npc_stim

FS = 100_000
BUFFER = 50_000
LABELS = [""] * 32
LABELS[2], LABELS[4], LABELS[5], LABELS[31] = "vsync_stim", "stim_photodiode", "stim_running", "lick_sensor"
pytestmark = pytest.mark.skipif(
    not hasattr(npc_sync, "SyncPatch"), reason="npc_sync without SyncPatch"
)


def _files(tmp: pathlib.Path, replay: bool):
    rng = np.random.default_rng(0)
    n = 3000
    fi = np.full(n - 1, 1 / 60) + rng.normal(0, 2e-5, n - 1)
    for i in (700, 1500, 2400):
        fi[i] = 3 / 60
    frames = 5.0 + np.r_[0, np.cumsum(fi)]
    edges = []
    for t in frames:
        edges += [(t - 0.002, 2, 1), (t, 2, 0)]
    pol = 1
    for f in range(0, n, 3):
        edges.append((frames[f] + (0.020 if pol else 0.024), 4, pol))
        pol ^= 1
    edges += [(4.95, 5, 1), (frames[-1] + 0.05, 5, 0)]
    for t in np.sort(rng.uniform(6, 50, 60)):
        edges += [(t, 31, 1), (t + 0.03, 31, 0)]
    items = sorted((int(round(t * FS)), b, v) for t, b, v in edges)
    times, states, state = [0], [0], 0
    for s, b, v in items:
        state = (state | (1 << b)) if v else (state & ~(1 << b))
        if times[-1] == s:
            states[-1] = state
        else:
            times.append(s)
            states.append(state)
    t, s = np.asarray(times, np.int64), np.asarray(states, np.int64)
    if replay:  # buffers 36-39 are copies of buffer 40
        g0, first = 40 * BUFFER, 36 * BUFFER
        g = (t >= g0) & (t < g0 + BUFFER)
        before, after = t < first, t >= g0
        t, s = (
            np.r_[t[before], *[first + c * BUFFER + t[g] - g0 for c in range(4)], t[after]],
            np.r_[s[before], *[s[g] for _ in range(4)], s[after]],
        )
    sync = tmp / ("replayed.h5" if replay else "clean.h5")
    meta = {
        "ni_daq": {"sample_freq": float(FS), "counter_bits": 32, "event_bits": 32},
        "start_time": "2026-01-01 10:00:00.000000",
        "stop_time": "2026-01-01 10:01:00.000000",
        "line_labels": LABELS,
        "timeouts": [],
        "version": "synthetic",
        "total_samples": 60 * FS,
    }
    with h5py.File(sync, "w") as f:
        f["data"] = np.c_[t.astype(np.uint32), s.astype(np.uint32)]
        f["meta"] = str(meta)
    stim = tmp / "DynamicRouting1_000000_20260101_100004.hdf5"
    with h5py.File(stim, "w") as f:
        f["frameIntervals"] = fi
        f["startTime"] = b"20260101_100004"
        f["diodeBoxFrameInterval"] = 3
    return sync, stim, fi, frames


def test_unpatched_replay_raises(tmp_path) -> None:
    sync, stim, *_ = _files(tmp_path, replay=True)
    with pytest.raises(AssertionError, match="replayed"):
        npc_stim.get_display_times(npc_sync.SyncDataset(sync), [stim])


def test_patched_replay_gives_display_times(tmp_path) -> None:
    sync, stim, fi, frames = _files(tmp_path, replay=True)
    _, _, _, clean_frames = _files(tmp_path, replay=False)
    clean = npc_stim.get_display_times(npc_sync.SyncDataset(tmp_path / "clean.h5"), [stim])
    raw = npc_sync.SyncDataset(sync, on_invalid="ignore")
    assert raw.invalid_intervals == ((18.0, 20.0),)
    patch = P.build_sync_patch(raw, [P.StimBlock(0, fi, 3)], builder="test")
    (block,) = npc_stim.get_display_times(npc_sync.SyncDataset(sync, patch=patch), [stim])
    (ref,) = clean
    assert ref.source == "photodiode" and ref.frame_source is None
    assert block.source == "photodiode"
    assert len(block.display_times) == len(frames)
    # display times equal the clean recording's to within a few samples
    assert np.max(np.abs(block.display_times - ref.display_times)) < 5 / FS
    inside = P.in_intervals(block.vsync_times, raw.invalid_intervals)
    assert (block.frame_source[inside] == 1).all()
    assert (block.frame_source[block.vsync_times < 17.9] == 0).all()


def test_patch_without_stim_block_rejects_only_that_block(tmp_path) -> None:
    sync, stim, fi, frames = _files(tmp_path, replay=True)
    raw = npc_sync.SyncDataset(sync, on_invalid="ignore")
    patch = P.build_sync_patch(raw, [], builder="test")  # vsync can't be reconstructed
    (block,) = npc_stim.get_display_times(npc_sync.SyncDataset(sync, patch=patch), [stim])
    assert block.source == "unavailable" and "couldn't reconstruct" in block.reason
