from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

import npc_io
import npc_sync
import numpy as np
import numpy.typing as npt
import scipy.signal

import npc_stim.stim

logger = logging.getLogger(__name__)

RUNNING_SAMPLE_RATE = 60
"""Visual stim f.p.s - assumed to equal running wheel sampling rate. i.e. one
running wheel sample per camstim vsync"""

RUNNING_SPEED_UNITS: Literal["cm/s", "m/s"] = "cm/s"
"""How to report in NWB - NWB expects SI, SDK might have previously reported cm/s"""

RUNNING_LOWPASS_FILTER_HZ = 4
"""Frequency for filtering running speed - filtered data stored in NWB `processing`, unfiltered
in `acquisition`"""

MAX_MOUSE_SPRINT_SPEED_CM_S = 334
"""Fastest recorded maximal sprint speed of any individual house mouse (wild-caught Mus
domesticus, forced sprint over 1 m on a racetrack: range 2.38-3.34 m/s; laboratory Hsd:ICR mice
0.89-1.68 m/s). Garland et al. 1995, Physiol Behav 58(5):869-876,
https://doi.org/10.1016/0031-9384(95)00148-c"""

RUNNING_SPEED_MAX_CM_S = 1.3 * MAX_MOUSE_SPRINT_SPEED_CM_S
"""Samples with |speed| above this (fastest recorded mouse + 30%) are physically implausible and
are set to NaN (in cm/s, regardless of `RUNNING_SPEED_UNITS`)."""


def get_running_speed_from_stim_files(
    *stim_path: npc_io.PathLike,
    sync: npc_sync.SyncPathOrDataset | None = None,
    filt: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Pools running speeds across files. Returns arrays of running speed and
    corresponding timestamps."""
    if not sync and len(stim_path) > 1:
        raise ValueError(
            "Must pass sync file to coordinate data from multiple stim files."
        )

    _running_speed_blocks = []
    _timestamps_blocks = []

    def _append(
        values: npt.NDArray[np.floating], times: npt.NDArray[np.floating]
    ) -> None:
        if len(times) + 1 == len(values):
            values = values[1:]
        elif len(values) + 1 == len(times):
            times = times[:-1]

        if len(times) == len(values):
            times = times[1:]
            values = values[1:]
            times = times + 0.5 * np.median(np.diff(times))
        else:
            raise ValueError(
                f"Length mismatch between running speed ({len(values)}) and timestamps ({len(times)})"
            )
        # we need to filter before pooling discontiguous blocks of samples
        values = filt(values) if filt else values
        _running_speed_blocks.append(values)
        _timestamps_blocks.append(times)

    if sync is None:
        assert len(stim_path) == 1
        _append(
            get_running_speed_from_hdf5(*stim_path),
            get_frame_times_from_stim_file(*stim_path),
        )
    else:
        # we need timestamps for each frame's nidaq-read time (wheel encoder is read before frame's
        # flip time)
        # there may be multiple h5 files with encoder
        # data per sync file: vsyncs are in blocks with a separating gap
        stim_path = sorted(set(stim_path))  # type: ignore[invalid-assignment]
        for hdf5 in stim_path:
            try:
                read_times = npc_stim.stim.get_input_data_times(hdf5, sync)
            except (ValueError, IndexError) as exc:
                logger.warning(f"Skipping {hdf5}: {exc!r}")
                continue
            h5_data = get_running_speed_from_hdf5(hdf5)
            if h5_data.size == 0:
                continue
            _append(h5_data, read_times)
    assert len(_running_speed_blocks) == len(_timestamps_blocks)
    if not _running_speed_blocks:
        return np.array([]), np.array([])
    sorted_block_indices = np.argsort([block[0] for block in _timestamps_blocks])
    running_speed = np.concatenate(
        [_running_speed_blocks[i] for i in sorted_block_indices]
    )
    timestamps = np.concatenate([_timestamps_blocks[i] for i in sorted_block_indices])
    assert np.all(np.diff(timestamps) > 0)
    return running_speed, timestamps


def get_frame_times_from_stim_file(
    stim_path: npc_io.PathLike,
) -> npt.NDArray:
    return np.concatenate(
        (
            [0],
            np.cumsum(npc_stim.stim.get_stim_data(stim_path)["frameIntervals"][:]),
        )
    )


def get_running_speed_from_hdf5(
    stim_path: npc_io.PathLike,
) -> npt.NDArray[np.floating]:
    """
    Running speed in m/s or cm/s (see `UNITS`).


    To align with timestamps, remove timestamp[0] and sample[0] and shift
    timestamps by half a frame (speed is estimated from difference between
    timestamps)

    See
    https://github.com/samgale/DynamicRoutingTask/blob/main/Analysis/DynamicRoutingAnalysisUtils.py

    >>> get_running_speed_from_hdf5('s3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5')
    array([        nan, 26.76305601, 26.33139382, ..., 37.12294866, 38.20210414, 39.49709072])
    """
    stim_path = npc_io.from_pathlike(stim_path)
    d = npc_stim.stim.get_stim_data(stim_path)
    if not (
        "rotaryEncoder" in d
        and isinstance(d["rotaryEncoder"][()], bytes)
        and d["rotaryEncoder"].asstr()[()] == "digital"
    ):
        raise ValueError(
            f"No rotary encoder data found (or not the expected format) in {stim_path}"
        )
    if "frameRate" in d:
        assert d["frameRate"][()] == RUNNING_SAMPLE_RATE
    count = d["rotaryEncoderCount"][:]
    if not np.any(np.nan_to_num(count)):
        logger.warning(f"No wheel revolutions found in {stim_path}")
        return np.array([])
    index = d["rotaryEncoderIndex"][:] if "rotaryEncoderIndex" in d else None
    if index is None:
        logger.warning(
            f"No rotaryEncoderIndex in {stim_path}: skipped encoder reads can only be "
            "detected by speed threshold"
        )
    elif index.shape != count.shape:
        logger.warning(
            f"rotaryEncoderIndex shape {index.shape} does not match rotaryEncoderCount "
            f"shape {count.shape} in {stim_path}: realigning index to count"
        )
        index = align_encoder_index_to_count(index, count)
    speed = get_running_speed_from_encoder_count(
        count,
        counts_per_rev=d["rotaryEncoderCountsPerRev"][()],
        wheel_radius_cm=d["wheelRadius"][()],
        index=index,
    )
    if (n := np.isnan(speed[1:]).sum()) > 0:
        logger.info(f"{n} running speed samples are NaN in {stim_path}")
    return speed


def align_encoder_index_to_count(
    index: npt.ArrayLike, count: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Remove spurious entries from `rotaryEncoderIndex` so it aligns with `rotaryEncoderCount`.

    TaskControl.readDigitalEncoder appends the parsed index, then the parsed count, and on any
    exception appends NaN to both. If the serial line is garbled such that the index parses but
    the count doesn't, `index` gets [value, NaN] for that frame while `count` gets [NaN]: `index`
    accumulates one extra entry per occurrence, always immediately before one of its NaNs.
    `count` has one entry per frame and is the reference.

    >>> index = [0, 1, 2, 3, np.nan, 5, 6]
    >>> count = [0, 1, 2, np.nan, 5, 6]
    >>> align_encoder_index_to_count(index, count)
    array([ 0.,  1.,  2., nan,  5.,  6.])

    Ambiguous cases raise:
    >>> align_encoder_index_to_count([0, 1, 2, 3], [0, 1, 2])
    Traceback (most recent call last):
    ...
    ValueError: Cannot align rotaryEncoderIndex (4,) to rotaryEncoderCount (3,)
    """
    index = np.asarray(index, dtype=np.float64)
    count = np.asarray(count, dtype=np.float64)
    if index.shape == count.shape:
        return index
    error = ValueError(
        f"Cannot align rotaryEncoderIndex {index.shape} to rotaryEncoderCount {count.shape}"
    )
    n_extra = index.size - count.size
    index_nan = np.flatnonzero(np.isnan(index))
    count_nan = np.flatnonzero(np.isnan(count))
    if n_extra < 0 or index_nan.size != count_nan.size or index_nan.size == 0:
        raise error
    offsets = index_nan - count_nan
    steps = np.diff(offsets, prepend=0)
    if not np.all((steps == 0) | (steps == 1)) or offsets[-1] != n_extra:
        raise error
    keep = np.ones(index.size, dtype=bool)
    keep[index_nan[steps == 1] - 1] = False
    aligned = index[keep]
    if aligned.shape != count.shape or not np.array_equal(
        np.isnan(aligned), np.isnan(count)
    ):
        raise error
    return aligned


def get_running_speed_from_encoder_count(
    count: npt.ArrayLike,
    counts_per_rev: float,
    wheel_radius_cm: float,
    index: npt.ArrayLike | None = None,
) -> npt.NDArray[np.floating]:
    """Running speed (`RUNNING_SPEED_UNITS`) from cumulative rotary encoder counts, one read per
    frame. Output has the same length as `count`, with NaN as the first sample (lost to diff).

    Unreliable samples are set to NaN:
    - missing reads (NaN count)
    - skipped encoder reads, where `rotaryEncoderIndex` does not increment by exactly 1: the
      count steps by a non-physical amount at the skip, and the following read is also
      contaminated (elevated speed), so both samples are masked
    - |speed| > `RUNNING_SPEED_MAX_CM_S` (e.g. garbage reads of ~1e13 counts)

    See https://github.com/AllenInstitute/npc_stim/issues/2

    >>> get_running_speed_from_encoder_count([0, 10, 20, 30], 8192, wheel_radius_cm=4.69)
    array([       nan, 2.15831097, 2.15831097, 2.15831097])

    Skipped read (index increments by 3): the step and the following sample are masked:
    >>> count, index = [0, 10, 20, 9000, 9010, 9020], [0, 1, 2, 5, 6, 7]
    >>> get_running_speed_from_encoder_count(count, 8192, 4.69, index=index)
    array([       nan, 2.15831097, 2.15831097,        nan,        nan, 2.15831097])
    """
    count = np.asarray(count, dtype=np.float64)
    speed_cm_s = np.concatenate(
        [
            [np.nan],
            np.diff(count / counts_per_rev)
            * 2
            * np.pi
            * wheel_radius_cm
            * RUNNING_SAMPLE_RATE,
        ]
    )
    bad = np.abs(np.nan_to_num(speed_cm_s, nan=0)) > RUNNING_SPEED_MAX_CM_S
    if index is not None:
        index = np.asarray(index, dtype=np.float64)
        if index.shape != count.shape:
            raise ValueError(
                f"rotaryEncoderIndex shape {index.shape} does not match "
                f"rotaryEncoderCount shape {count.shape}"
            )
        skipped = np.concatenate([[False], np.diff(index) != 1])  # NaN != 1 is True
        bad |= skipped
        bad[1:] |= skipped[:-1]  # read following a skip is also contaminated
    speed_cm_s[bad] = np.nan
    if RUNNING_SPEED_UNITS == "m/s":
        return speed_cm_s / 100
    elif RUNNING_SPEED_UNITS == "cm/s":
        return speed_cm_s
    raise ValueError(f"Unexpected units for running speed: {RUNNING_SPEED_UNITS}")


def lowpass_filter(running_speed: npt.NDArray, restore_nan: bool = True) -> npt.NDArray:
    """
    Careful not to filter discontiguous blocks of samples.
    See
    https://github.com/AllenInstitute/AllenSDK/blob/36e784d007aed079e3cad2b255ca83cdbbeb1330/allensdk/brain_observatory/behavior/data_objects/running_speed/running_processing.py

    NaN samples (missed or glitched encoder reads) are linearly interpolated before filtering
    (leading/trailing NaNs take the nearest valid value) rather than zero-filled, so they don't
    create artificial dips. If `restore_nan`, they are set back to NaN in the output.
    See https://github.com/AllenInstitute/npc_stim/issues/4

    >>> x = np.full(100, 20.0); x[50] = np.nan
    >>> y = lowpass_filter(x)
    >>> bool(np.isnan(y[50])), bool(np.allclose(np.delete(y, 50), 20))
    (True, True)
    """
    running_speed = np.asarray(running_speed, dtype=np.float64)
    is_nan = np.isnan(running_speed)
    if is_nan.all():
        return running_speed.copy()
    filled = running_speed.copy()
    if is_nan.any():
        x = np.arange(len(running_speed))
        filled[is_nan] = np.interp(x[is_nan], x[~is_nan], running_speed[~is_nan])
    b, a = scipy.signal.butter(
        3, Wn=RUNNING_LOWPASS_FILTER_HZ, fs=RUNNING_SAMPLE_RATE, btype="lowpass"
    )
    filtered = scipy.signal.filtfilt(b, a, filled)
    if restore_nan:
        filtered[is_nan] = np.nan
    return filtered


if __name__ == "__main__":
    from npc_stim import testmod

    testmod()
