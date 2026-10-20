import numpy as np
import npc_stim

from npc_stim import running


def test_import_package():
    pass


def test_running_speed_timestamps_are_interval_midpoints(monkeypatch):
    monkeypatch.setattr(
        running,
        "get_running_speed_from_hdf5",
        lambda _: np.array([np.nan, 1, 2, 3]),
    )
    monkeypatch.setattr(
        running,
        "get_frame_times_from_stim_file",
        lambda _: np.arange(4) / running.RUNNING_SAMPLE_RATE,
    )

    _, timestamps = running.get_running_speed_from_stim_files("stim.hdf5")

    np.testing.assert_allclose(
        timestamps,
        (np.arange(3) + 0.5) / running.RUNNING_SAMPLE_RATE,
    )
