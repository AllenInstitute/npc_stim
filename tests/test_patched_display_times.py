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


# phantom flicker (diodeBoxFrameInterval 1): the SyncPatch removes it, npc_stim consumes the
# cleaned line without removing edges itself ------------------------------------------------- #

R = 1 / 60
LAT_R, LAT_F = 0.020, 0.024


def _box1_files(
    tmp: pathlib.Path,
    phantom_pairs: dict[int, list[int]],
    name: str,
    first_fall_delay: float = 0.0,
    vsync_pulse_width: float = 0.015,
):
    """Box-1 block: 4000 frames with held frames; phantom flicker = one-refresh flashes of the
    other buffer during holds (TTL: first edge 3 ms late, restoring edge on time).
    `first_fall_delay`: the first vsync's falling edge is this late (its rising edge on time)."""
    n = 4000
    holds = {700: 3, 1500: 2, 1520: 4, 2400: 11, 3000: 3, 3300: 3, 3600: 2}
    n_ref = np.ones(n - 1, int)
    for k, h in holds.items():
        n_ref[k] = h
    display = 5.0 + np.r_[0, np.cumsum(n_ref)] * R
    rng = np.random.default_rng(1)
    fall = display + 0.0002 + rng.normal(0, 2e-5, n)
    fi = np.diff(fall)
    edges = []
    for i, t in enumerate(fall):
        edges += [(t - vsync_pulse_width, 2, 1), (t + (first_fall_delay if i == 0 else 0), 2, 0)]
    pol = 1
    for k in range(n):
        edges.append((display[k] + (LAT_R if pol else LAT_F), 4, pol))
        for r in phantom_pairs.get(k, []):
            edges.append((display[k] + r * R + (LAT_R if 1 - pol else LAT_F) + 0.003, 4, 1 - pol))
            edges.append((display[k] + (r + 1) * R + (LAT_R if pol else LAT_F), 4, pol))
        pol ^= 1
    edges += [(4.95, 5, 1), (fall[-1] + 0.05, 5, 0)]
    items = sorted((int(round(t * FS)), b, v) for t, b, v in edges)
    times, states, state = [0], [0], 0
    for s, b, v in items:
        state = (state | (1 << b)) if v else (state & ~(1 << b))
        if times[-1] == s:
            states[-1] = state
        else:
            times.append(s)
            states.append(state)
    sync = tmp / f"{name}.h5"
    meta = {
        "ni_daq": {"sample_freq": float(FS), "counter_bits": 32, "event_bits": 32},
        "start_time": "2026-01-01 10:00:00.000000",
        "stop_time": "2026-01-01 10:01:40.000000",
        "line_labels": LABELS,
        "timeouts": [],
        "version": "synthetic",
        "total_samples": 100 * FS,
    }
    with h5py.File(sync, "w") as f:
        f["data"] = np.c_[np.asarray(times, np.uint32), np.asarray(states, np.uint32)]
        f["meta"] = str(meta)
    stim = tmp / f"DynamicRouting1_000000_20260101_100004_{name}.hdf5"
    with h5py.File(stim, "w") as f:
        f["frameIntervals"] = fi
        f["startTime"] = b"20260101_100004"
        f["diodeBoxFrameInterval"] = 1
    return sync, stim, fi, display


PAIRS = {700: [1], 2400: [1, 3, 5, 7], 3000: [1], 3300: [1]}

needs_flicker_patch = pytest.mark.skipif(
    not hasattr(npc_sync, "align_diode_toggles"), reason="npc_sync without phantom-flicker removal"
)


@needs_flicker_patch
def test_phantom_flicker_rejects_block_without_patch(tmp_path) -> None:
    sync, stim, fi, display = _box1_files(tmp_path, PAIRS, "flicker")
    (block,) = npc_stim.get_display_times(npc_sync.SyncDataset(sync), [stim])
    assert block.source == "unavailable" and "diode flips" in block.reason


@needs_flicker_patch
def test_patch_cleaned_line_is_consumed_without_further_removal(tmp_path, monkeypatch) -> None:
    sync, stim, fi, display = _box1_files(tmp_path, PAIRS, "flicker")
    raw = npc_sync.SyncDataset(sync)
    patch = P.build_sync_patch(raw, [P.StimBlock(0, fi, 1, "synthetic")], builder="test")
    (rec,) = patch.diode_blocks
    assert rec.status == "cleaned" and rec.n_phantom_flicker == 2 * sum(len(v) for v in PAIRS.values())

    def no_removal(*args, **kwargs):
        raise AssertionError("npc_stim must not remove diode edges from a line the SyncPatch cleaned")

    monkeypatch.setattr(npc_stim.display_times, "anomalous_interval_indices", no_removal)
    (block,) = npc_stim.get_display_times(npc_sync.SyncDataset(sync, patch=patch), [stim])
    assert block.source == "photodiode", block.reason
    assert len(block.display_times) == len(display)
    # toggle frames ("on" flips) are exact; the others are interpolated by the estimator
    on = np.arange(0, len(display), 2)
    expected = display + LAT_R + npc_stim.display_times.MONITOR_CENTER_REFRESH_TIME
    assert np.max(np.abs(block.display_times[on] - expected[on])) < 2e-5


@needs_flicker_patch
def test_patch_rejected_block_is_unavailable(tmp_path) -> None:
    sync, stim, fi, display = _box1_files(tmp_path, PAIRS, "flicker")
    raw = npc_sync.SyncDataset(sync)
    patch = P.build_sync_patch(raw, [P.StimBlock(0, fi, 1, "synthetic")], builder="test")
    patch.diode_blocks[0].status = "rejected"
    patch.diode_blocks[0].reason = "test"
    patch.removed = {}
    (block,) = npc_stim.get_display_times(npc_sync.SyncDataset(sync, patch=patch), [stim])
    assert block.source == "unavailable" and "SyncPatch rejected" in block.reason


@needs_flicker_patch
def test_clean_block_is_unchanged_by_the_patch(tmp_path) -> None:
    sync, stim, fi, display = _box1_files(tmp_path, {}, "clean")
    raw = npc_sync.SyncDataset(sync)
    patch = P.build_sync_patch(raw, [P.StimBlock(0, fi, 1, "synthetic")], builder="test")
    assert patch.diode_blocks[0].status == "cleaned" and patch.removed == {}
    (a,) = npc_stim.get_display_times(raw, [stim])
    (b,) = npc_stim.get_display_times(npc_sync.SyncDataset(sync, patch=patch), [stim])
    assert a.source == b.source == "photodiode"
    assert np.array_equal(a.display_times, b.display_times)


def test_late_first_vsync_falling_edge_keeps_first_flip(tmp_path) -> None:
    """666986_2023-08-16 RFMapping: the first vsync's falling edge was 12 ms late and the first
    diode flip came 13 ms after it (normal latency from the rising edge). Flips were filtered from
    the first falling edge + MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC, so the first flip was dropped,
    every frame looked one frame off and the block had no display times."""
    # first flip at display + LAT_R ~ falling edge + 20 ms: make it 8 ms after the falling edge
    sync, stim, _, display = _box1_files(
        tmp_path, {}, "late_first_fall", first_fall_delay=0.012, vsync_pulse_width=0.002
    )
    ref_sync, ref_stim, *_ = _box1_files(tmp_path, {}, "ref", vsync_pulse_width=0.002)
    (ref,) = npc_stim.get_display_times(npc_sync.SyncDataset(ref_sync), [ref_stim])
    assert ref.source == "photodiode", ref.reason
    (block,) = npc_stim.get_display_times(npc_sync.SyncDataset(sync), [stim])
    assert block.source == "photodiode", block.reason
    assert len(block.display_times) == len(display)
    assert np.max(np.abs(block.display_times - ref.display_times)) < 2e-5
