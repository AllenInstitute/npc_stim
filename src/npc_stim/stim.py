from __future__ import annotations

import datetime
import io
import logging
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Literal, SupportsFloat

import h5py
import npc_io
import npc_session
import npc_sync
import numpy as np
import numpy.typing as npt
import upath

from npc_stim.types import StimPathOrDataset

if TYPE_CHECKING:
    import npc_stim.display_times

logger = logging.getLogger(__name__)


def get_stim_data(stim_path: StimPathOrDataset, **kwargs) -> h5py.File | dict:
    if isinstance(stim_path, h5py.File):
        return stim_path
    if isinstance(stim_path, Mapping):  # ie. from pkl file
        return dict(stim_path)
    path = npc_io.from_pathlike(stim_path)
    if path.suffix in (".hdf5", ".h5"):
        return get_h5_stim_data(path, **kwargs)
    if path.suffix == ".pkl":
        return get_pkl_stim_data(path, **kwargs)
    raise ValueError(f"Unknown stim file type: {path}")


def get_h5_stim_data(stim_path: StimPathOrDataset, **kwargs) -> h5py.File:
    if isinstance(stim_path, h5py.File):
        return stim_path
    if isinstance(stim_path, Mapping):
        raise ValueError("Susepcted pickle date encountered: use `get_pkl_stim_data`")
    kwargs.setdefault("mode", "r")
    path = npc_io.from_pathlike(stim_path)
    if path.protocol in ("", "file"):
        return h5py.File(io.BytesIO(path.read_bytes()), **kwargs)
    else:
        return h5py.File(path.open(mode="rb", cache_type="first"), **kwargs)


def get_pkl_stim_data(stim_path: StimPathOrDataset, **kwargs) -> dict:
    if isinstance(stim_path, Mapping):
        return dict(stim_path)
    if isinstance(stim_path, h5py.File):
        raise TypeError(
            "hdf5 data encountered when pkl data expected: use `get_h5_stim_data` instead"
        )
    kwargs.setdefault("encoding", "latin1")
    return pickle.loads(npc_io.from_pathlike(stim_path).read_bytes())


def get_input_data_times(
    stim: npc_io.PathLike,
    sync: npc_sync.SyncPathOrDataset | None = None,
) -> npt.NDArray[np.float64]:
    """Best-estimate time of `getInputData()` in psychopy event loop, in seconds, from start
    of experiment. Uses preceding frame's vsync time if sync provided"""
    stim_data = get_stim_data(stim)
    assert isinstance(stim_data, h5py.File), "Only hdf5 stim files supported for now"
    if not sync:
        return np.concatenate(([0], np.cumsum(stim_data["frameIntervals"][:])))
    return np.concatenate(
        [
            [np.nan],
            assert_stim_times(
                get_stim_frame_times(stim, sync=sync, frame_time_type="vsync")[stim]
            )[:-1],
        ]
    )


def get_flip_times(
    stim: npc_io.PathLike,
    sync: npc_sync.SyncPathOrDataset | None = None,
) -> npt.NDArray[np.float64]:
    """Best-estimate time of `flip()` at end of psychopy event loop, in seconds, from start
    of experiment. Uses frame's vsync time sync provided."""
    stim_data = get_stim_data(stim)
    assert isinstance(stim_data, h5py.File), "Only hdf5 stim files supported for now"
    if not sync:
        return np.concatenate(
            (
                (c := np.cumsum(f := stim_data["frameIntervals"][:])),
                [c[-1] + np.median(np.diff(f))],
            )
        )
    return assert_stim_times(
        get_stim_frame_times(stim, sync=sync, frame_time_type="vsync")[stim]
    )


def get_vis_display_times(
    stim: npc_io.PathLike,
    sync: npc_sync.SyncPathOrDataset | None = None,
) -> npt.NDArray[np.float64]:
    """Best-estimate time of monitor update. Uses photodiode if sync provided. Without sync, this equals frame times."""
    stim_data = get_stim_data(stim)
    assert isinstance(stim_data, h5py.File), "Only hdf5 stim files supported for now "
    if not sync:
        return get_flip_times(stim_data)
    return assert_stim_times(
        get_stim_frame_times(stim, sync=sync, frame_time_type="display_time")[stim]
    )


def get_stim_block_to_path(
    stim_paths: Iterable[npc_io.PathLike],
    sync_data: npc_sync.SyncDataset,
) -> dict[int, upath.UPath | None]:
    """
    Map sync block indices to stim file paths.

    Each stim file is matched to the vsync block whose first frame is closest to the stim
    file's start time, and only if the block has exactly as many vsyncs as the stim file has
    frames. A block claimed by more than one stim file matches none of them. See
    `get_stim_block_matches` for the reason each unmatched stim file wasn't matched.

    Returns a dict where keys are block indices and values are either stim paths or None
    if no stim file matches that block.
    """
    block_to_stim, _ = get_stim_block_matches(stim_paths, sync_data)
    return block_to_stim


def get_stim_block_matches(
    stim_paths: Iterable[npc_io.PathLike],
    sync_data: npc_sync.SyncDataset,
) -> tuple[dict[int, upath.UPath | None], dict[upath.UPath, Exception]]:
    """
    Exact one-to-one matching of stim files to sync vsync blocks.

    - a stim file is matched to the block whose first vsync is closest to its start time
    - the block's vsync count must equal the stim file's frame count exactly: a mismatch is
      reported, not absorbed
    - a block claimed by more than one stim file matches none of them

    Returns (block index -> stim path or None, unmatched stim path -> exception explaining
    why it wasn't matched). Stim files that can't be read, have no frames, or are from a
    different day are in neither.
    """
    blocks = sync_data.vsync_times_in_blocks
    block_to_stim: dict[int, upath.UPath | None] = dict.fromkeys(range(len(blocks)))
    errors: dict[upath.UPath, Exception] = {}
    if not blocks:
        return block_to_stim, errors

    n_frames_per_block = np.asarray([len(x) for x in blocks])
    first_frame_per_block = np.asarray([x[0] for x in blocks])

    claims: dict[int, list[upath.UPath]] = {}
    for stim_path in dict.fromkeys(npc_io.from_pathlike(p) for p in stim_paths):
        try:
            stim_data = get_h5_stim_data(stim_path)
        except OSError:
            continue  # Skip files that can't be opened

        n_stim_frames = get_total_stim_frames(stim_data)
        if n_stim_frames == 0:
            continue  # Skip files with no frames

        stim_start_time = get_stim_start_time(stim_data)
        stim_start_time_on_sync = (
            stim_start_time - sync_data.start_time
        ).total_seconds()
        if abs(stim_start_time_on_sync) >= 24 * 60 * 60:
            continue  # Skip files from different days

        block_idx = int(np.argmin(abs(first_frame_per_block - stim_start_time_on_sync)))
        n_block_frames = int(n_frames_per_block[block_idx])
        if n_block_frames != n_stim_frames:
            time_diff = first_frame_per_block[block_idx] - stim_start_time_on_sync
            exact_len_blocks = np.flatnonzero(
                n_frames_per_block == n_stim_frames
            ).tolist()
            errors[stim_path] = IndexError(
                f"Frame count mismatch for {stim_path!r}: closest sync block by start time "
                f"(block {block_idx}, starting {time_diff:.1f} s after the stim file) has "
                f"{n_block_frames} vsyncs, stim file has {n_stim_frames} frames "
                f"(difference {n_stim_frames - n_block_frames}). "
                f"Blocks with exactly {n_stim_frames} vsyncs: {exact_len_blocks}"
            )
            continue
        claims.setdefault(block_idx, []).append(stim_path)

    for block_idx, paths in claims.items():
        if len(paths) == 1:
            block_to_stim[block_idx] = paths[0]
            continue
        for stim_path in paths:
            errors[stim_path] = IndexError(
                f"Sync block {block_idx} is claimed by {len(paths)} stim files, so matches none "
                f"of them: {[p.name for p in paths]}"
            )
    return block_to_stim, errors


def get_stim_display_times(
    *stim_paths: npc_io.PathLike,
    sync: npc_sync.SyncPathOrDataset,
    fallback: Literal["constant_lag"] | None = None,
    raise_on_error: bool = False,
) -> dict[npc_io.PathLike, Exception | npc_stim.display_times.BlockDisplayTimes]:
    """As `get_stim_frame_times(..., frame_time_type='display_time')`, but values are
    `BlockDisplayTimes` (display times with their source and the matched sync block)
    instead of arrays. Stims without display times get a `ValueError`, as there."""
    return _get_stim_frame_times(
        *stim_paths,
        sync=sync,
        frame_time_type="display_time",
        fallback=fallback,
        raise_on_error=raise_on_error,
        return_blocks=True,
    )


def get_stim_frame_times(
    *stim_paths: npc_io.PathLike,
    sync: npc_sync.SyncPathOrDataset,
    frame_time_type: Literal["display_time", "vsync"] = "display_time",
    fallback: Literal["constant_lag"] | None = None,
    raise_on_error: bool = False,
) -> dict[npc_io.PathLike, Exception | npt.NDArray[np.float64]]:
    """
    - keys are the stim paths provided as inputs
    - for `frame_time_type='display_time'`, see `npc_stim.get_display_times` for
      `fallback` and `raise_on_error` (the deprecated env vars `SYNC_FALLBACK_CONSTANT_LAG`
      and `SYNC_DEBUG` are still honored if the arguments aren't given). A stim whose block
      has no display times gets a `ValueError`.

    >>> bad_stim = 's3://aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/DynamicRouting1_670248_20230802_120703.hdf5'
    >>> good_stim_1 = 's3://aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/Spontaneous_670248_20230802_114611.hdf5'
    >>> good_stim_2 = 's3://aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/SpontaneousRewards_670248_20230802_130736.hdf5'
    >>> sync = 's3://aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/20230802T113053.h5'

    Returns the frame times for each stim file based on start time and number
    of frames - can be provided in any order:
    >>> frame_times = get_stim_frame_times(good_stim_2, good_stim_1, sync=sync)
    >>> len(frame_times[good_stim_2])
    36000

    Returns Exception if the stim file can't be opened, or it has no frames.
    Should be used with `assert_stim_times` to raise a possible exception:
    >>> frame_times = get_stim_frame_times(bad_stim, sync=sync)
    >>> assert_stim_times(frame_times[bad_stim])
    Traceback (most recent call last):
    ...
    FileNotFoundError: aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/DynamicRouting1_670248_20230802_120703.hdf5
    """
    return _get_stim_frame_times(
        *stim_paths,
        sync=sync,
        frame_time_type=frame_time_type,
        fallback=fallback,
        raise_on_error=raise_on_error,
    )


def _get_stim_frame_times(
    *stim_paths: npc_io.PathLike,
    sync: npc_sync.SyncPathOrDataset,
    frame_time_type: Literal["display_time", "vsync"] = "display_time",
    fallback: Literal["constant_lag"] | None = None,
    raise_on_error: bool = False,
    return_blocks: bool = False,
) -> dict:
    import npc_stim.display_times

    if fallback is None and os.getenv("SYNC_FALLBACK_CONSTANT_LAG"):
        logger.warning(
            "SYNC_FALLBACK_CONSTANT_LAG env var is deprecated: pass fallback='constant_lag'"
        )
        fallback = "constant_lag"
    raise_on_error = raise_on_error or bool(os.getenv("SYNC_DEBUG"))

    # load sync file once
    sync_data = npc_sync.get_sync_data(sync)

    # Get mapping of block indices to stim paths
    block_to_stim, match_errors = get_stim_block_matches(stim_paths, sync_data)
    stim_to_block = {p: i for i, p in block_to_stim.items() if p is not None}

    stim_frame_times: dict[npc_io.PathLike, Exception | npt.NDArray] = {}
    stim_blocks: dict[npc_io.PathLike, npc_stim.display_times.BlockDisplayTimes] = {}

    display_time_blocks: tuple[npc_stim.display_times.BlockDisplayTimes, ...] | None = (
        None  # computed on first use
    )
    # Process each stim file and generate appropriate result (frame times or exception)
    for stim_path in stim_paths:

        try:
            stim_data = get_h5_stim_data(stim_path)
        except OSError as exc:
            stim_frame_times[stim_path] = exc
            continue

        n_stim_frames = get_total_stim_frames(stim_data)
        if n_stim_frames == 0:
            stim_frame_times[stim_path] = ValueError(
                f"No frames found in {stim_path = }"
            )
            continue

        stim_start_time = get_stim_start_time(stim_data)
        if (
            abs((stim_start_time - sync_data.start_time).total_seconds())
            >= 24 * 60 * 60
        ):
            logger.error(
                f"Skipping {stim_path =}, sync data is from a different day: {stim_start_time = }, {sync_data.start_time = }"
            )
            continue

        upath_ = npc_io.from_pathlike(stim_path)
        matched_block_idx = stim_to_block.get(upath_)
        if matched_block_idx is None:
            stim_frame_times[stim_path] = match_errors.get(
                upath_, IndexError(f"No sync block matched {stim_path!r}")
            )
            continue
        if frame_time_type == "display_time":
            if display_time_blocks is None:
                display_time_blocks = npc_stim.display_times.get_display_times(
                    sync_data,
                    block_to_stim,
                    fallback=fallback,
                    raise_on_error=raise_on_error,
                )
            block = display_time_blocks[matched_block_idx]
            if block.display_times is None:
                stim_frame_times[stim_path] = ValueError(
                    f"No frame display times for {stim_path!r} (sync block {matched_block_idx}): {block.reason}"
                )
            else:
                stim_frame_times[stim_path] = block.display_times
                if return_blocks:
                    stim_blocks[stim_path] = block
        elif frame_time_type == "vsync":
            stim_frame_times[stim_path] = sync_data.vsync_times_in_blocks[
                matched_block_idx
            ]
        else:
            raise ValueError(
                f"Unknown frame time type: {frame_time_type}. Expected 'display_time' or 'vsync'."
            )
    # Sort keys by first frame time, or by exception if present
    sorted_keys = sorted(
        stim_frame_times.keys(),
        key=lambda x: (
            0 if isinstance(stim_frame_times[x], Exception) else stim_frame_times[x][0]  # type: ignore[index]
        ),
    )
    if return_blocks:
        return {k: stim_blocks.get(k, stim_frame_times[k]) for k in sorted_keys}
    return {k: stim_frame_times[k] for k in sorted_keys}


def assert_stim_times(result: Exception | npt.NDArray) -> npt.NDArray:
    """Raise exception if result is an exception, otherwise return result"""
    if isinstance(result, Exception):
        raise result from None
    return result


def get_num_trials(
    stim_path_or_data: npc_io.PathLike | h5py.File,
) -> int:
    """
    >>> get_num_trials('s3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5')
    524
    """
    stim_data = get_h5_stim_data(stim_path_or_data)
    return len(
        stim_data.get("trialEndFrame")
        or stim_data.get("stimStartFrame")
        or stim_data.get("trialOptoOnsetFrame")  # for optoTagging
    )


def get_stim_start_time(
    stim_path_or_data: npc_io.PathLike | h5py.File,
) -> datetime.datetime:
    """Absolute datetime of the first frame, according to the stim file
    >>> get_stim_start_time('s3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5')
    datetime.datetime(2023, 8, 31, 13, 14, 18)
    """
    # TODO make compatible with pkl files
    stim_data = get_h5_stim_data(stim_path_or_data)
    # get stim start time & convert to datetime
    return npc_session.DatetimeRecord(stim_data["startTime"][()].decode()).dt


def get_total_stim_frames(stim_path_or_data: npc_io.PathLike | h5py.File) -> int:
    """
    >>> get_total_stim_frames('s3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5')
    217261
    """
    # TODO make compatible with pkl files
    stim_data = get_h5_stim_data(stim_path_or_data)
    frame_intervals = stim_data["frameIntervals"][:]
    if len(frame_intervals) == 0:
        return 0
    return len(frame_intervals) + 1


def get_stim_duration(stim_path_or_data: npc_io.PathLike | h5py.File) -> float:
    """
    >>> get_stim_duration('s3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5')
    3647.0994503999827
    """
    # TODO make compatible with pkl files
    stim_data = get_h5_stim_data(stim_path_or_data)
    return np.sum(stim_data["frameIntervals"][:])


def _normalize_opto_trigger_frames(
    stim_data: h5py.File,
    raw_frames: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Convert DynamicRouting opto onset frames to absolute frame indices.

    Depending on the task-control version, ``trialOptoOnsetFrame`` may contain
    either an offset from ``trialStimStartFrame`` or an absolute frame index.
    The encoding can even be mixed within one file. Trial boundaries let us
    distinguish the two cases without assuming opto starts after the visual
    stimulus: an absolute opto onset may legitimately precede
    ``trialStimStartFrame``.
    """
    raw = np.asarray(raw_frames, dtype=float).reshape(-1)
    n_trials = min(get_num_trials(stim_data), len(raw))
    raw = raw[:n_trials]

    stim_start_dataset = stim_data.get("trialStimStartFrame")
    trial_start_dataset = stim_data.get("trialStartFrame")
    trial_end_dataset = stim_data.get("trialEndFrame")
    if (
        stim_start_dataset is None
        or trial_start_dataset is None
        or trial_end_dataset is None
    ):
        return raw

    stim_start = np.asarray(stim_start_dataset[:n_trials], dtype=float).reshape(-1)
    trial_start = np.asarray(trial_start_dataset[:n_trials], dtype=float).reshape(-1)
    trial_end = np.asarray(trial_end_dataset[:n_trials], dtype=float).reshape(-1)
    absolute = raw
    relative = stim_start + raw
    finite = np.isfinite(raw)
    absolute_valid = finite & (absolute >= trial_start) & (absolute <= trial_end)
    relative_valid = finite & (relative >= trial_start) & (relative <= trial_end)
    absolute_only = absolute_valid & ~relative_valid
    relative_only = relative_valid & ~absolute_valid
    ambiguous = finite & ~(absolute_only | relative_only)

    if absolute_only.any() and relative_only.any():
        logger.warning(
            "Mixed absolute and relative opto onset frame encoding detected; "
            "normalizing each trial using its trial boundaries."
        )

    # If both candidates are valid, use the dominant unambiguous encoding. In
    # a file with no unambiguous values, retain the historical monotonicity
    # fallback for absolute OptoTagging frames versus relative task frames.
    if absolute_only.any() or relative_only.any():
        use_relative = relative_only | (
            ambiguous & (relative_only.sum() >= absolute_only.sum())
        )
        normalized = np.where(use_relative, relative, absolute)
    else:
        without_nans = raw[finite]
        if len(without_nans) > 1 and np.all(np.diff(without_nans) > 0):
            normalized = absolute
        else:
            normalized = relative

    return normalized


def get_stim_trigger_frames(
    stim_path_or_data: npc_io.PathLike | h5py.File,
    stim_type: str | Literal["opto"] = "stim",
) -> tuple[int | None, ...]:
    """Frame index of stim command being sent. len() == num trials.

    - for DynamicRouting1 files, use `stim_type='opto'` to get the trigger frames for opto

    >>> path = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5'
    >>> frames = get_stim_trigger_frames(path)
    >>> len(frames)
    524

    >>> frames = get_stim_trigger_frames(path, stim_type='opto')
    >>> len(frames)
    0
    """
    stim_data = get_h5_stim_data(stim_path_or_data)
    opto = stim_data.get("trialOptoOnsetFrame")
    normalize_opto_frames = stim_type == "opto"
    start_frames = (
        (stim_data.get("trialStimStartFrame") or stim_data.get("stimStartFrame"))
        if not normalize_opto_frames
        else opto
    )

    if start_frames is None and opto is not None:
        # optoTagging experiments use "trialOptoOnsetFrame" instead of
        # "trialStimStartFrame" - should be safe to switch.. the stim_type
        # parameter just wasn't set to 'opto' when the function was called
        if (
            "optoFeedbackBlocks" in stim_data
            or "feedback" in stim_data["taskVersion"].asstr()[()]
        ):
            logger.info("Feedback opto experiment detected")
        start_frames = opto  # may be adjusted below
        normalize_opto_frames = True
        if stim_data.get("optoTaggingLocs") is None:
            logger.warning(
                'Using "trialOptoOnsetFrame" instead of "trialStimStartFrame" - this is likely an old optoTagging experiment, and `stim_type` was specified as `stim` instead of `opto`.'
            )

    start_frames = start_frames[: get_num_trials(stim_data)].squeeze()
    if normalize_opto_frames:
        start_frames = _normalize_opto_trigger_frames(stim_data, start_frames)
    else:
        monotonic_increase = np.all(
            (without_nans := start_frames[~np.isnan(start_frames)])[1:]
            > without_nans[:-1]
        )
        if not monotonic_increase:
            # Behavior files with opto report the onset frame of opto relative
            # to stim onset for each trial. OptoTagging files specify absolute
            # frame indices.
            start_frames += stim_data.get("trialStimStartFrame")[
                : get_num_trials(stim_data)
            ].squeeze()

    return tuple(
        int(v) if ~np.isnan(v) else None
        for v in safe_index(start_frames, np.arange(len(start_frames)))
    )


def safe_index(
    array: npt.ArrayLike, indices: SupportsFloat | Iterable[SupportsFloat]
) -> npt.NDArray:
    """Checks `indices` can be safely used as array indices (i.e. all
    numerical float values are integers), then indexes into `array` using `np.where`.

    - returns nans where `indices` is nan
    - returns a scalar if `indices` is a scalar #TODO use @overload: current type annotation is insufficient

    >>> safe_index([1, 2], 0)
    1
    >>> safe_index([1., 2.], 0)
    1.0
    >>> safe_index([1., 2.], np.nan)
    nan
    >>> safe_index([1., 2., 3.1], [0, np.nan, 2.0])
    array([1. , nan, 3.1])

    Type of array is preserved, if possible:
    >>> safe_index([1, 2, 3], [0., 1., 2.])
    array([1, 2, 3])

    Type of array can't be preserved if any indices are nan:
    >>> safe_index([1, 2, 3], [0, np.nan, 2.0])
    array([ 1., nan,  3.])
    """
    idx: npt.NDArray = np.array(indices)  # copy
    if not all(idx[~np.isnan(idx)] == idx[~np.isnan(idx)].astype(np.int32)):
        raise TypeError(
            f"Non-integer numerical values cannot be used as indices: {idx[np.isnan(idx)][0]}"
        )
    array = np.array(array)  # copy/make sure array can be fancy-indexed
    int_idx = np.where(np.isnan(idx), -1, idx)
    result = np.where(np.isnan(idx), np.nan, array[int_idx.astype(np.int32)])
    # np.where casts indexed array to floats just because of the
    # possibility of nans being in result, even if they aren't:
    # cast back if appropriate
    if not np.isnan(result).any():
        result = result.astype(array.dtype)
    # if indices was a scalar, return a scalar instead of a 0d array
    if not isinstance(indices, Iterable):
        assert result.size == 1
        return result.item()
    return result


def validate_stim(
    *stim_paths: npc_io.PathLike,
    sync: npc_sync.SyncPathOrDataset,
) -> None:
    """
    Check that the stim files can be opened and have valid data that corresponds
    to the sync file. Raises an AssertionError if any of the stim files fail to
    open or can't be identified on sync.

    - validation of other properties (trial starts, display times, etc.) is
      specific to the type of stim and file format

    >>> good_stim_1 = 's3://aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/Spontaneous_670248_20230802_114611.hdf5'
    >>> good_stim_2 = 's3://aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/SpontaneousRewards_670248_20230802_130736.hdf5'
    >>> sync = 's3://aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/20230802T113053.h5'
    >>> validate_stim(good_stim_1, good_stim_2, sync=sync)

    # stim file that doesn't open or has bad data:
    >>> bad_stim = 's3://aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/DynamicRouting1_670248_20230802_120703.hdf5'
    >>> validate_stim(bad_stim, sync=sync)
    Traceback (most recent call last):
    ...
    AssertionError: Failed to validate stim_path = 's3://aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/DynamicRouting1_670248_20230802_120703.hdf5'
    """

    def validate_single_stim(stim_path, sync=sync) -> None:
        for v in get_stim_frame_times(
            stim_path, sync=sync
        ).values():  # should be only one entry
            assert_stim_times(v)

    logger.info(f"Validating {len(stim_paths)} stim files with sync data")
    for stim_path in stim_paths:
        logger.debug(f"Validating {stim_path = }")
        try:
            validate_single_stim(stim_path)
        except Exception as exc:
            raise AssertionError(f"Failed to validate {stim_path = }") from exc
        logger.info(f"Validated {stim_path = }")


if __name__ == "__main__":
    from npc_stim import testmod

    testmod()
