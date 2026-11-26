"""Frame display-time estimation (moved from npc_sync <= 0.2.9, where these tests lived in
tests/test_known_files.py, test_plots.py, test_core.py, test_replay.py and
test_discontinuities.py)."""

from __future__ import annotations

import pathlib
import threading
import time

import h5py
import npc_sync
import numpy as np
import pytest
import upath

import npc_stim
import npc_stim.display_times

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def _stim_paths(sync_path: str) -> tuple[upath.UPath, ...]:
    return tuple(upath.UPath(sync_path).parent.glob("*.hdf5"))


def _display_times(sync_path: str, **kwargs) -> tuple[npc_stim.BlockDisplayTimes, ...]:
    return npc_stim.get_display_times(sync_path, _stim_paths(sync_path), **kwargs)


# fail-fast on faulty sync data ------------------------------------------------ #


def test_get_display_times_fails_fast_on_replay() -> None:
    sync = object.__new__(npc_sync.SyncDataset)
    sync.path = pathlib.Path("replayed.h5")
    sync.invalid_intervals = ((5247.5, 5257.0),)
    with pytest.raises(AssertionError, match="5247.500-5257.000 s"):
        _ = npc_stim.get_display_times(sync, {})


def test_get_display_times_fails_fast_on_discontinuity() -> None:
    sync = object.__new__(npc_sync.SyncDataset)
    sync.path = pathlib.Path("dropped.h5")
    sync.invalid_intervals = ()
    sync.timebase_discontinuities = ((2458.56, 2488.58, -1.0),)
    with pytest.raises(AssertionError, match="2458.560-2488.580 s"):
        _ = npc_stim.get_display_times(sync, {})


# stim-file parameters ------------------------------------------------------------ #


def _write_stim(path: pathlib.Path, start: str, interval: int | None) -> pathlib.Path:
    with h5py.File(path, "w") as f:
        f["startTime"] = start.encode()
        if interval is not None:
            f["diodeBoxFrameInterval"] = interval
    return path


def test_diode_box_frame_interval_is_read_from_stim_file(tmp_path: pathlib.Path) -> None:
    current = _write_stim(tmp_path / "current.hdf5", "20250801_120000", 3)
    assert npc_stim.get_diode_box_frame_interval(current) == 3


def test_missing_diode_box_frame_interval_is_1(tmp_path: pathlib.Path) -> None:
    for start in ("20230101_120000", "20250714_175002", "20260101_120000"):
        path = _write_stim(tmp_path / f"{start}.hdf5", start, None)
        assert npc_stim.get_diode_box_frame_interval(path) == 1


# estimator ------------------------------------------------------------------------ #


@pytest.mark.parametrize("raise_on_error", (False, True))
def test_missing_flips_not_at_end_of_block_fails_instead_of_hanging(
    raise_on_error: bool,
) -> None:
    """60 s (3600 frames) of the 712815_2024-05-23 task from 1143.6 s, times relative to the first
    vsync: the estimator returns 3597 display times and the last flip is already later than the
    last vsync + the mean latency, so `add_missing_diode_flips_for_truncated_sync` added nothing
    and npc_stim <= 0.1.28 looped forever. Since 0.1.32 `reconcile_box1_frame_count` sees that the
    last frames' latency is far from the block median (the missing flips aren't at the end)."""
    data = np.load(FIXTURES_DIR / "chunk_missing_flips_not_at_end.npz")
    result: list = []

    def helper() -> None:
        try:
            result.append(
                npc_stim.display_times._estimate_block_display_times(
                    sync_data_name="712815_2024-05-23",
                    block_idx=0,
                    stim_id="712815_2024-05-23_chunk",
                    vsyncs=data["vsyncs"],
                    vsync_rising_edges=data["vsync_rising_edges"],
                    rising=data["rising"],
                    falling=data["falling"],
                    first_frame_start=float(data["vsyncs"][0]),
                    vsyncs_per_diode_flip=1,
                    expected_vsyncs=len(data["vsyncs"]),
                    diode_cleaned=False,
                    raise_on_error=raise_on_error,
                )
            )
        except BaseException as exc:  # an exception in a thread is otherwise swallowed
            result.append(exc)

    t = threading.Thread(target=helper, daemon=True)  # daemon: a hang mustn't block exit
    t.start()
    t.join(60)
    assert not t.is_alive(), "estimator is looping forever"
    expected = "3597 display times for 3600 vsyncs, but the latency of the last 50 frames"
    if raise_on_error:
        assert isinstance(result[0], AssertionError) and expected in str(result[0])
    else:
        assert result[0].times is None
        assert expected in result[0].reason
        assert "the missing flip(s) are not at the end of the block" in result[0].reason
        assert result[0].retryable  # so that `get_display_times` still tries the rising-edge window


# frame index check (constant-lag blocks) ------------------------------------------ #


def _vsyncs_with_long_frames(
    n: int = 2000, long_frames: tuple[int, ...] = (500, 1200)
) -> np.ndarray:
    intervals = np.full(n - 1, 1 / 60)
    intervals[list(long_frames)] *= 2
    return 100 + np.concatenate(([0], np.cumsum(intervals)))


def test_frame_index_check_passes_with_clock_drift_and_jitter() -> None:
    v = _vsyncs_with_long_frames()
    stim = np.diff(v) * (1 + 5e-6)  # stim PC clock 5 ppm slow
    v = v.copy()
    v[800] += 0.0047  # one late vsync falling edge: +4.7, -4.7 ms on sync only
    v[1201] += 0.010  # ... on the frame after a long frame
    v[300] += 0.009  # ... that takes an interval over the threshold (25.7 ms)
    check = npc_stim.check_frame_indices(v, stim)
    assert check.passed is True and check.reason is None
    assert (check.n_long_sync, check.n_long_stim, check.n_same_index) == (3, 2, 2)
    assert (check.n_adjacent_index, check.n_unpaired, check.n_unmatched) == (0, 1, 0)
    assert abs(check.max_step) < 1e-4


def test_frame_index_check_tolerates_long_frame_split_across_adjacent_frames() -> None:
    v = _vsyncs_with_long_frames()
    stim = np.diff(v)
    v = v.copy()
    v[501] -= 0.015  # intervals 500, 501 on sync: 18.3, 31.7 ms; in the stim file: 33.3, 16.7 ms
    check = npc_stim.check_frame_indices(v, stim)
    assert check.passed is True
    assert (check.n_same_index, check.n_adjacent_index, check.n_unmatched) == (1, 1, 0)


@pytest.mark.parametrize(
    "stim_long_frames, n_unmatched",
    (
        ((500,), 1),  # dropped frame on sync that the stim file doesn't have
        ((500, 1200, 1500), 1),  # ... and the reverse
        ((500, 1202), 2),  # 2 frames apart
    ),
)
def test_frame_index_check_fails(stim_long_frames: tuple[int, ...], n_unmatched: int) -> None:
    v = _vsyncs_with_long_frames()
    stim = np.diff(_vsyncs_with_long_frames(long_frames=stim_long_frames))
    check = npc_stim.check_frame_indices(v, stim)
    assert check.passed is False
    assert check.n_unmatched == n_unmatched
    assert "don't match" in check.reason
    assert abs(abs(check.max_step) - 1 / 60) < 1e-4


def test_frame_index_check_fails_on_different_number_of_dropped_refreshes() -> None:
    v = _vsyncs_with_long_frames()
    stim = np.diff(v).copy()
    stim[500] += 1 / 60  # 2 refreshes dropped in the stim file, 1 on sync, at the same index
    check = npc_stim.check_frame_indices(v, stim)
    assert check.passed is False
    assert (check.n_same_index, check.n_unmatched, check.unmatched_frames) == (2, 1, (500,))


def test_frame_index_check_fails_on_duration() -> None:
    v = _vsyncs_with_long_frames()
    check = npc_stim.check_frame_indices(v, np.diff(v) * 1.001)
    assert check.passed is False
    assert check.n_unmatched == 0
    assert "duration" in check.reason


def test_frame_index_check_not_run_without_stim_file(tmp_path: pathlib.Path) -> None:
    v = _vsyncs_with_long_frames()
    assert npc_stim.get_frame_index_check(v, None).passed is None
    no_intervals = _write_stim(tmp_path / "no_intervals.hdf5", "20240101_120000", 1)
    check = npc_stim.get_frame_index_check(v, no_intervals)
    assert check.passed is None and "frameIntervals" in check.reason
    path = tmp_path / "stim.hdf5"
    with h5py.File(path, "w") as f:
        f["frameIntervals"] = np.diff(v)
    assert npc_stim.get_frame_index_check(v, path).passed is True
    assert npc_stim.get_frame_index_check(v[:-1], path).passed is False  # frame count


# known files (S3) ----------------------------------------------------------------- #


def test_frame_index_check_only_for_constant_lag_blocks() -> None:
    """699847_2024-04-18 (no SyncPatch): the photodiode is absent for half of the task."""
    sync = "s3://aind-private-data-prod-o5171v/ecephys_699847_2024-04-18_13-42-34/behavior/20240418T134234.h5"
    blocks = _display_times(sync, fallback="constant_lag")
    assert {b.source for b in blocks} == {"photodiode", "constant_lag"}
    for b in blocks:
        if b.source == "photodiode":
            assert b.frame_index_check is None
            continue
        assert b.frame_index_check == npc_stim.get_frame_index_check(b.vsync_times, b.stim_path)
    task = next(b for b in blocks if b.stim_path and "DynamicRouting1" in b.stim_path.name)
    assert task.source == "constant_lag"
    assert task.frame_index_check.passed is True
    assert (task.frame_index_check.n_long_sync, task.frame_index_check.n_same_index) == (221, 221)
    # no fallback: only stims without display-timed visual stims (here SpontaneousRewards,
    # whose photodiode is unusable) still get constant lag, with the check
    no_fallback = _display_times(sync)
    for b in no_fallback:
        if b.stim_path is not None and not npc_stim.display_times.has_display_timed_visual_stims(b.stim_path):
            assert b.source in ("photodiode", "constant_lag")
            assert (b.frame_index_check is None) == (b.source == "photodiode")
        else:
            assert b.frame_index_check is None
    assert any(b.source == "constant_lag" for b in no_fallback)
    assert next(b for b in no_fallback if b.stim_path and "DynamicRouting1" in b.stim_path.name).source == "unavailable"


def test_slow_processing_file() -> None:
    """File with many missing diode flips, caused processing to be slow by trying
    to fill-in the missing diode flips. This is a rare case, but it's good to have
    a check for it.

    - was resolved by checking number of missing flips, and if above a threshold
      using vsync times + constant monitor lag
    """
    t0 = time.time()
    _ = _display_times(
        "s3://aind-private-data-prod-o5171v/ecephys_699847_2024-04-18_13-42-34/behavior/20240418T134234.h5"
    )
    t = time.time() - t0
    assert t < 60, f"file with many missing diode flips is being processed too slowly: {t = }"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="666986_20230814 has replayed sync data (1273.0-1273.5 s), so get_display_times raises; "
    "before the thread's exception was re-raised this test passed without checking anything. "
    "Needs a replacement file with missing diode flips at the start of a block.",
)
def test_missing_diode_flips_at_start_of_block() -> None:
    errors: list[BaseException] = []

    def helper() -> None:
        try:
            _ = _display_times(
                "s3://aind-ephys-data/ecephys_666986_2023-08-14_11-38-55/behavior/20230814T113855.h5"
            )
        except BaseException as exc:  # an exception in a thread is otherwise swallowed
            errors.append(exc)

    # launch job in a separate thread, then kill if it takes too long
    import threading

    t = threading.Thread(target=helper)
    t.start()
    t.join(20)
    assert not t.is_alive(), "file with missing diode flips at start of block is taking too long to process"
    if errors:
        raise errors[0]


def test_flips_outside_stim_running() -> None:
    """A file with 3 diode flips per vsync interval. The last pair of flips for the task occur after
    stim running goes low"""
    blocks = _display_times(
        "s3://aind-open-data/ecephys_795555_2025-08-25_12-01-16/behavior/20250825T120116.h5"
    )
    assert blocks[3].display_times is not None
    assert len(blocks[3].display_times) == 216896, "Handling of flips outside stim running is incorrect"


def test_standard_file() -> None:
    """A standard file with no issues."""
    sync = "s3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/20230821T124345.h5"
    blocks = _display_times(sync)
    assert len(blocks) == 7
    assert all(b.is_photodiode for b in blocks)
    assert len(np.concatenate([b.vsync_times for b in blocks])) == 414763
    assert len(np.concatenate([b.display_times for b in blocks])) == 414763


def test_plots(tmp_path: pathlib.Path) -> None:
    sync = "s3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/20230821T124345.h5"
    blocks = _display_times(sync)
    npc_stim.plot_stim_onsets(sync, blocks).savefig(tmp_path / "test_stim_starts.png")
    npc_stim.plot_stim_offsets(sync, blocks).savefig(tmp_path / "test_stim_ends.png")
    npc_stim.plot_diode_measured_sync_square_flips(sync).savefig(tmp_path / "diode.png")


if __name__ == "__main__":
    pytest.main(["-v", __file__])


# end-of-block reconciliation (npc_stim#9) ----------------------------------------- #


def test_extra_flips_mid_block_are_not_truncated_at_end() -> None:
    """Two extra flips in the middle of a box-1 block put every later frame 2 refreshes early
    (733891_2024-09-18 task, npc_stim#9). npc_stim <= 0.1.31 truncated the 2 extra display times
    at the end of the block; now the block is rejected."""
    v = np.arange(5000) / 59.95
    lag = npc_stim.display_times.AVERAGE_VSYNC_TO_DIODE_FLIP_TIME
    t = v + lag
    # frames from 4000 get the flips of frames 2 earlier, and 2 frames' flips are left at the end
    times = np.concatenate([t[:4000], t[3998:4000] + 0.002, t[4000:]])
    times[4000:4002] = t[4000:4002] - 2 / 59.95  # the 2 extra flips
    times = np.sort(times)
    _, _, problem = npc_stim.display_times.reconcile_box1_frame_count(times, v)
    assert problem is not None and "not at the end of the block" in problem


def test_extra_flip_after_last_frame_is_dropped() -> None:
    v = np.arange(5000) / 59.95
    lag = npc_stim.display_times.AVERAGE_VSYNC_TO_DIODE_FLIP_TIME
    times = np.append(v + lag, v[-1] + lag + 1 / 59.95)  # transition to the grey screen
    out, invented, problem = npc_stim.display_times.reconcile_box1_frame_count(times, v)
    assert problem is None and len(invented) == 0
    assert np.array_equal(out, v + lag)


def test_missing_and_extra_flip_mid_block_fail_step_check() -> None:
    """Counts match, but frames 1000-2999 are a refresh late: only the running-median check sees it"""
    v = np.arange(5000) / 59.95
    lag = npc_stim.display_times.AVERAGE_VSYNC_TO_DIODE_FLIP_TIME
    t = v + lag
    t[1000:3000] += 1 / 59.95
    problem = npc_stim.display_times.get_residual_step_problem(t, v)
    assert problem is not None and "1000-2999" in problem
