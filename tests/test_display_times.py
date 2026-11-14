"""Frame display-time estimation (moved from npc_sync <= 0.2.9, where these tests lived in
tests/test_known_files.py, test_plots.py, test_core.py, test_replay.py and
test_discontinuities.py)."""

from __future__ import annotations

import pathlib
import time

import h5py
import npc_sync
import numpy as np
import pytest
import upath

import npc_stim


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


def test_missing_diode_box_frame_interval_only_accepted_before_key_existed(
    tmp_path: pathlib.Path,
) -> None:
    old = _write_stim(tmp_path / "old.hdf5", "20250101_120000", None)
    assert npc_stim.get_diode_box_frame_interval(old) == 1
    last = _write_stim(tmp_path / "last.hdf5", "20250714_175002", None)
    assert npc_stim.get_diode_box_frame_interval(last) == 1
    new = _write_stim(tmp_path / "new.hdf5", "20250801_120000", None)
    with pytest.raises(KeyError):
        npc_stim.get_diode_box_frame_interval(new)


# known files (S3) ----------------------------------------------------------------- #


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


def test_missing_diode_flips_at_start_of_block() -> None:
    def helper() -> None:
        _ = _display_times(
            "s3://aind-ephys-data/ecephys_666986_2023-08-14_11-38-55/behavior/20230814T113855.h5"
        )

    # launch job in a separate thread, then kill if it takes too long
    import threading

    t = threading.Thread(target=helper)
    t.start()
    t.join(20)
    assert not t.is_alive(), "file with missing diode flips at start of block is taking too long to process"


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
