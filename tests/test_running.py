"""Tests for running speed glitch masking (gh #2) and NaN-aware filtering (gh #4).

Event patterns below are taken from raw OptoTagging hdf5 files; see
experiments/issue2_encoder/raw_*.json.
"""

import numpy as np
import pytest

import npc_stim.running as running

CPR = 8192
RADIUS = 4.69
CM_S_PER_COUNT = 2 * np.pi * RADIUS * running.RUNNING_SAMPLE_RATE / CPR


@pytest.fixture(autouse=True)
def _cm_s(monkeypatch):
    monkeypatch.setattr(running, "RUNNING_SPEED_UNITS", "cm/s")


def _speed(count, index=None):
    return running.get_running_speed_from_encoder_count(count, CPR, RADIUS, index=index)


def _steady(n=50, counts_per_frame=140.0, start=1e6):
    """~30 cm/s steady running."""
    return start + counts_per_frame * np.arange(n), 1000 + np.arange(n, dtype=float)


def test_threshold_from_literature() -> None:
    # fastest recorded house mouse (3.34 m/s) + 30%
    assert running.RUNNING_SPEED_MAX_CM_S == pytest.approx(434.2)


def test_clean_data_unchanged() -> None:
    count, index = _steady()
    s = _speed(count, index)
    assert np.isnan(s[0])
    assert np.allclose(s[1:], 140 * CM_S_PER_COUNT)


def test_skip_step_and_following_read_masked() -> None:
    # 668755_2023-08-31 OptoTagging_..._124903 i=12449: index +3 with 2517-count step,
    # next read (index +1) still 1616 counts (~349 cm/s, below threshold)
    count = np.array([1238461.0, 1238497, 1238541, 1241058, 1242674, 1242656, 1242638])
    index = np.array(
        [482792305.0, 482792306, 482792307, 482792310, 482792311, 482792312, 482792313]
    )
    s = _speed(count, index)
    assert np.isnan(s[[0, 3, 4]]).all()
    assert not np.isnan(s[[1, 2, 5, 6]]).any()
    assert np.nanmax(np.abs(s)) < 10


def test_garbage_read_spike_masked() -> None:
    # 668755_2023-08-31 OptoTagging_..._142538 i=12404: garbage 1.3e13 count read, then index +3
    count = np.array(
        [1326378.0, 1326576, 1326772, 13269651341114, 1348324, 1348477, 1348628]
    )
    index = np.array(
        [483061082.0, 483061083, 483061084, 483061085, 483061088, 483061089, 483061090]
    )
    s = _speed(count, index)
    assert np.isnan(s[[0, 3, 4, 5]]).all()
    assert not np.isnan(s[[1, 2, 6]]).any()
    assert np.nanmax(np.abs(s)) < 50


def test_garbage_read_masked_without_index() -> None:
    count, _ = _steady()
    count[20] = 1.3e13
    s = _speed(count)
    assert np.isnan(s[[20, 21]]).all()
    assert np.isnan(s).sum() == 3  # + leading sample
    assert np.nanmax(np.abs(s)) < running.RUNNING_SPEED_MAX_CM_S


def test_counter_wrap_masked_without_index() -> None:
    count, _ = _steady()
    count[30:] -= 2**32
    s = _speed(count)
    assert np.isnan(s[30])
    assert np.isnan(s).sum() == 2


def test_nan_read_masked() -> None:
    # NaN count and index rows, with a step across them (668755 OptoTagging_..._142538 i=7094)
    count = np.array([559196.0, 559490, np.nan, 597404, 597687, 597970])
    index = np.array([483055769.0, 483055770, np.nan, 483055773, 483055774, 483055775])
    s = _speed(count, index)
    assert np.isnan(s[[0, 2, 3, 4]]).all()
    assert not np.isnan(s[[1, 5]]).any()


def test_duplicate_read_masked() -> None:
    count, index = _steady()
    index[10:] -= 1  # index repeats: diff == 0
    count[10:] -= 140  # count repeats too
    s = _speed(count, index)
    assert np.isnan(s[[10, 11]]).all()
    assert np.isnan(s).sum() == 3


def test_index_shape_mismatch_raises() -> None:
    count, index = _steady()
    with pytest.raises(ValueError):
        _speed(count, index[:-1])


def test_m_per_s(monkeypatch) -> None:
    monkeypatch.setattr(running, "RUNNING_SPEED_UNITS", "m/s")
    count, index = _steady()
    count[20] = 1.3e13
    s = _speed(count, index)
    assert np.allclose(s[1:20], 140 * CM_S_PER_COUNT / 100)
    assert np.isnan(s[[20, 21]]).all()


def test_lowpass_no_dip_at_nan() -> None:
    x = np.full(200, 30.0)
    x[[0, 100, 101, 199]] = np.nan
    y = running.lowpass_filter(x)
    assert np.isnan(y[[0, 100, 101, 199]]).all()
    assert np.allclose(y[~np.isnan(x)], 30)
    y2 = running.lowpass_filter(x, restore_nan=False)
    assert np.allclose(y2, 30)


def test_lowpass_matches_old_behaviour_without_nan() -> None:
    import scipy.signal

    rng = np.random.default_rng(0)
    x = rng.normal(20, 5, 500)
    b, a = scipy.signal.butter(3, Wn=4, fs=60, btype="lowpass")
    assert np.allclose(running.lowpass_filter(x), scipy.signal.filtfilt(b, a, x))


def test_lowpass_all_nan() -> None:
    assert np.isnan(running.lowpass_filter(np.full(50, np.nan))).all()
