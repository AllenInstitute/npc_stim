"""
Frame display times: best estimate of when each stim frame appeared on the monitor,
from sync vsync + photodiode edges and the stim file that produced the frames.

Moved from npc_sync (<= 0.2.9: `SyncDataset.frame_display_time_blocks` and the module-level
estimator functions). npc_sync is now only responsible for line access, fault detection and
block splitting; everything that interprets the stimulus (diode box, monitor latency, stim
files) lives here.

The estimator (`get_frame_display_times` and helpers) is unchanged from npc_sync 0.2.9.
Orchestration changes:

- stim files are an explicit input (no globbing next to the sync file)
- diode-box parameters are read from the stim file being processed; a missing
  `diodeBoxFrameInterval` key is only accepted for files that pre-date the key
- the constant-lag fallback and re-raising errors are explicit parameters rather than env
  vars, and the source of each block's times is returned with the times
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Union

import h5py
import npc_io
import npc_sync
import numpy as np
import numpy.typing as npt
import scipy.signal
import upath
from typing_extensions import TypeAlias

import npc_stim.stim

if TYPE_CHECKING:
    import matplotlib.figure

logger = logging.getLogger(__name__)

MONITOR_CENTER_REFRESH_TIME = 0.008
"""Best estimate of time from photodiode flip (at top of screen) to updated
content at the center of the screen. Screen refreshes in stages from top to bottom,
in 16 ms: measured by Corbett."""

AVERAGE_VSYNC_TO_DIODE_FLIP_TIME = 0.01955  #! sometimes shifts to ~0.035

CONSTANT_MONITOR_LAG: float = (
    MONITOR_CENTER_REFRESH_TIME + AVERAGE_VSYNC_TO_DIODE_FLIP_TIME
)
"""Best estimate of the average time between vsync falling edge and the center of
the screen updating, in seconds. For use when a reliable photodiode signal
is unavailable."""

MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC = 0.015  # 0.022 is typical

FRAME_RATE = 59.95
FRAME_INTERVAL = 1 / FRAME_RATE

MAX_VSYNC_DIODE_FLIP_SEPARATION_SEC = 5 / FRAME_RATE
"""Diode flips later than this after the last vsync of a block are not assigned to it."""

DIODE_EDGE_END_PADDING_SEC = 10 * FRAME_INTERVAL
"""Diode edges up to this long after stim_running goes low are still assigned to the block."""

# we've seen apparently real flip intervals of 0.29 * 59.95, and the line between anomalous and real isn't clear-cut
SUSPICIOUS_INTERVAL_THRESHOLD = 0.3 / FRAME_RATE
DEFINITE_SHORT_INTERVAL_THRESHOLD = 0.05 / FRAME_RATE
# at 60 fps we should never have diode-flip intervals this small: consider them anomalous

DisplayTimeSource: TypeAlias = Literal["photodiode", "constant_lag", "unavailable"]
Fallback: TypeAlias = Union[Literal["constant_lag"], None]


def get_plot_dir(mkdir: bool = False) -> upath.UPath | None:
    if (p := os.environ.get("SYNC_PLOT_DIR", None)) is None:
        return None
    path = upath.UPath(p).resolve()
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


@dataclasses.dataclass(frozen=True, eq=False)
class BlockDisplayTimes:
    """Display times for one block of vsyncs on sync (one stimulus), with provenance."""

    block_index: int
    """Index into `SyncDataset.vsync_times_in_blocks`."""
    stim_path: upath.UPath | None
    """Stim file matched to the block, or None if no stim file was matched."""
    vsync_times: npt.NDArray[np.float64]
    """vsync falling edges for the block (seconds on sync)."""
    display_times: npt.NDArray[np.float64] | None
    """Estimated display time of each frame (seconds on sync). None if unavailable."""
    source: DisplayTimeSource
    """'photodiode': estimated from diode flips; 'constant_lag': vsync +
    `CONSTANT_MONITOR_LAG` (only when requested with `fallback='constant_lag'`);
    'unavailable': no display times."""
    diode_box_frame_interval: int | None = None
    """Frames per diode-box flip, from the stim file (None if no stim file)."""
    reason: str | None = None
    """Why the photodiode was not used (None if it was)."""

    @property
    def is_photodiode(self) -> bool:
        return self.source == "photodiode"


def assert_sync_frame_timing_is_valid(sync: npc_sync.SyncPathOrDataset) -> None:
    """Raise `AssertionError` if the sync file contains replayed data (see
    `SyncDataset.invalid_intervals`) or a timebase discontinuity (see
    `SyncDataset.timebase_discontinuities`), since vsync and diode edges can't be trusted.
    """
    sync_data = npc_sync.get_sync_data(sync)
    if sync_data.invalid_intervals:
        raise AssertionError(
            f"{sync_data} contains replayed sync data (all lines invalid) at "
            + ", ".join(f"{a:.3f}-{z:.3f} s" for a, z in sync_data.invalid_intervals)
            + ": frame display times can't be computed reliably"
        )
    if sync_data.timebase_discontinuities:
        raise AssertionError(
            f"{sync_data} has a timebase discontinuity (all lines affected, later events shifted) "
            "between barcode onsets at "
            + ", ".join(
                f"{a:.3f}-{z:.3f} s ({d:+.3f} s)"
                for a, z, d in sync_data.timebase_discontinuities
            )
            + ": frame display times can't be computed reliably"
        )


def get_diode_box_frame_interval(stim_path_or_data: npc_io.PathLike | h5py.File) -> int:
    """Number of frames between diode-box flips, from the stim file.

    Files written by TaskControl before `diodeBoxFrameInterval` existed
    (samgale/DynamicRoutingTask@11b32f5) have no key and flipped the box on every frame (1).
    """
    stim_data = npc_stim.stim.get_h5_stim_data(stim_path_or_data)
    if "diodeBoxFrameInterval" not in stim_data:
        return 1
    value = stim_data["diodeBoxFrameInterval"]
    if isinstance(value, h5py.Dataset):
        value = value[()]
    return int(value)


def get_expected_diode_flip_rate(sync: npc_sync.SyncPathOrDataset) -> int:
    """Best-guess at what the diode flip period should be, e.g. 1 s for
    MPE/pipeline recordings, 1/60 s for Sam's TaskControl scripts."""
    sync_data = npc_sync.get_sync_data(sync)
    med = np.median(
        np.diff(sync_data.get_edges("all", "stim_photodiode", units="seconds"))
    )
    diode_assymmetry = 0.05  # s
    for period in (1 / 60, 1):
        if 0.9 * period - diode_assymmetry < med < 1.1 * period + diode_assymmetry:
            return int(1 / period)
    raise ValueError(f"Unexpected diode flip period: {med} sec")


def check_stim_photodiode(sync: npc_sync.SyncPathOrDataset) -> None:
    """Raise `AssertionError` if the diode flip rate doesn't match a known convention."""
    try:
        _ = get_expected_diode_flip_rate(sync)
    except ValueError as exc:
        raise AssertionError("Frame rate estimated from diode flips is abnormal.") from exc


def get_constant_lag_display_time_blocks(
    sync: npc_sync.SyncPathOrDataset,
) -> tuple[npt.NDArray[np.floating], ...]:
    """Blocks of vsync times + `CONSTANT_MONITOR_LAG`: one block per stimulus. For use
    when a reliable photodiode signal is unavailable."""
    return tuple(
        block + CONSTANT_MONITOR_LAG
        for block in npc_sync.get_sync_data(sync).vsync_times_in_blocks
    )


def get_diode_edges_in_blocks(
    sync: npc_sync.SyncPathOrDataset,
) -> tuple[tuple[npt.NDArray[np.floating], ...], tuple[npt.NDArray[np.floating], ...]]:
    """Rising and falling photodiode edges, assigned to blocks so that they can be zipped
    with `SyncDataset.vsync_times_in_blocks` (there may still be more or fewer blocks than
    vsync blocks if the assignment fails).
    """
    sync_data = npc_sync.get_sync_data(sync)
    vsync_times_in_blocks = sync_data.vsync_times_in_blocks
    diode_rising_edges = sync_data.get_rising_edges("stim_photodiode", units="seconds")
    diode_falling_edges = sync_data.get_falling_edges(
        "stim_photodiode", units="seconds"
    )
    assert abs(len(diode_rising_edges) - len(diode_falling_edges)) < 2
    diode_rising_edges_in_blocks = sync_data.get_edges_in_blocks(
        "stim_photodiode", "rising", end_padding=DIODE_EDGE_END_PADDING_SEC
    )
    diode_falling_edges_in_blocks = sync_data.get_edges_in_blocks(
        "stim_photodiode", "falling", end_padding=DIODE_EDGE_END_PADDING_SEC
    )

    if any(
        len(diode_edge_blocks) < len(vsync_times_in_blocks)
        for diode_edge_blocks in (
            diode_rising_edges_in_blocks,
            diode_falling_edges_in_blocks,
        )
    ):
        # too few blocks of diode flips - affected some early DRPilot
        # sessions with near-constant diode flips between stimulus
        # blocks, and no stim-running signal to filter them

        # - vsyncs necessarily precede diode flips
        # - split existing diode flip blocks at the first vsync time for
        #   each vsync block
        # - we'll clean up any excess diode flips at the end of the block later
        # to see the problem compare these two lists:
        # [(v[0], v[-1]) for v in vsync_times_in_blocks]
        # [(v[0], v[-1]) for v in diode_falling_edges_in_blocks]
        for v_start in (v[0] for v in vsync_times_in_blocks):
            assert len(diode_rising_edges_in_blocks) == len(
                diode_falling_edges_in_blocks
            )
            new_rising_edges: list[npt.NDArray] = []
            new_falling_edges: list[npt.NDArray] = []
            for d_idx, (d_start, d_stop) in enumerate(
                (d[0], d[-1]) for d in diode_falling_edges_in_blocks
            ):
                if d_start < v_start < d_stop and (
                    split := np.searchsorted(
                        diode_falling_edges_in_blocks[d_idx], v_start
                    )
                ) not in (0, len(diode_falling_edges_in_blocks[d_idx])):
                    for new, old in zip(
                        (new_rising_edges, new_falling_edges),
                        (
                            diode_rising_edges_in_blocks[d_idx],
                            diode_falling_edges_in_blocks[d_idx],
                        ),
                    ):
                        new.extend((old[:split], old[split:]))
                else:
                    for new, old in zip(
                        (new_rising_edges, new_falling_edges),
                        (
                            diode_rising_edges_in_blocks[d_idx],
                            diode_falling_edges_in_blocks[d_idx],
                        ),
                    ):
                        new.append(old)
            diode_rising_edges_in_blocks, diode_falling_edges_in_blocks = (
                tuple(new_rising_edges),
                tuple(new_falling_edges),
            )

    if any(
        len(diode_edge_blocks) > len(vsync_times_in_blocks)
        for diode_edge_blocks in (
            diode_rising_edges_in_blocks,
            diode_falling_edges_in_blocks,
        )
    ):
        # more blocks of diode flips than blocks of vsyncs
        def is_mismatch(edges, block) -> bool:
            return edges[-1] < block[0]

        for idx, vsync_block in enumerate(vsync_times_in_blocks):
            # work through blocks in order,
            # discard blocks with diode flips that don't match any vsyncs
            while is_mismatch(diode_rising_edges_in_blocks[idx], vsync_block):
                diode_rising_edges_in_blocks = tuple(
                    block
                    for i, block in enumerate(diode_rising_edges_in_blocks)
                    if i != idx
                )
            while is_mismatch(diode_falling_edges_in_blocks[idx], vsync_block):
                diode_falling_edges_in_blocks = tuple(
                    block
                    for i, block in enumerate(diode_falling_edges_in_blocks)
                    if i != idx
                )
    return tuple(diode_rising_edges_in_blocks), tuple(diode_falling_edges_in_blocks)


def _get_block_to_stim_path(
    sync_data: npc_sync.SyncDataset,
    stim_paths: Iterable[npc_io.PathLike] | Mapping[int, npc_io.PathLike | None],
) -> dict[int, upath.UPath | None]:
    if isinstance(stim_paths, Mapping):
        return {
            int(k): (None if v is None else npc_io.from_pathlike(v))
            for k, v in stim_paths.items()
        }
    return npc_stim.stim.get_stim_block_to_path(tuple(stim_paths), sync_data)


def get_display_times(
    sync: npc_sync.SyncPathOrDataset,
    stim_paths: Iterable[npc_io.PathLike] | Mapping[int, npc_io.PathLike | None],
    *,
    fallback: Fallback = None,
    raise_on_error: bool = False,
) -> tuple[BlockDisplayTimes, ...]:
    """Frame display times for every block of vsyncs on sync, one entry per block of
    `SyncDataset.vsync_times_in_blocks`.

    - `stim_paths`: either stim files to match to blocks (with
      `npc_stim.get_stim_block_to_path`), or an existing mapping of block index to stim path.
      Blocks without a stim file get no display times (the vsync count and diode-box
      interval can't be checked).
    - `fallback='constant_lag'`: blocks where the photodiode can't be used get vsync +
      `CONSTANT_MONITOR_LAG`, with `source='constant_lag'`. Default: no display times
      (`source='unavailable'`).
    - `raise_on_error`: re-raise errors from the estimator instead of recording the block as
      not using the photodiode.

    Raises `AssertionError` if the sync file contains replayed data or a timebase
    discontinuity (see `assert_sync_frame_timing_is_valid`).
    """
    sync_data = npc_sync.get_sync_data(sync)
    assert_sync_frame_timing_is_valid(sync_data)
    vsync_times_in_blocks = sync_data.vsync_times_in_blocks
    block_to_stim_path = _get_block_to_stim_path(sync_data, stim_paths)
    diode_rising_edges_in_blocks, diode_falling_edges_in_blocks = (
        get_diode_edges_in_blocks(sync_data)
    )

    results: list[BlockDisplayTimes] = []
    for block_idx, (vsyncs, rising, falling) in enumerate(
        zip(
            vsync_times_in_blocks,
            diode_rising_edges_in_blocks,
            diode_falling_edges_in_blocks,
        )
    ):
        stim_path = block_to_stim_path.get(block_idx)

        def not_using_diode(
            reason: str, vsyncs_per_diode_flip: int | None
        ) -> BlockDisplayTimes:
            return BlockDisplayTimes(
                block_index=block_idx,
                stim_path=stim_path,
                vsync_times=vsyncs,
                display_times=(
                    vsyncs + CONSTANT_MONITOR_LAG
                    if fallback == "constant_lag"
                    else None
                ),
                source="constant_lag" if fallback == "constant_lag" else "unavailable",
                diode_box_frame_interval=vsyncs_per_diode_flip,
                reason=reason,
            )

        # keep flips after first vsync + an empirically determined min latency
        # between vsync and screen update
        falling = falling[falling > (vsyncs[0] + MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC)]
        rising = rising[rising > (vsyncs[0] + MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC)]

        # keep flips only up to a certain time after the last vsync
        falling = falling[falling < (vsyncs[-1] + MAX_VSYNC_DIODE_FLIP_SEPARATION_SEC)]
        rising = rising[rising < (vsyncs[-1] + MAX_VSYNC_DIODE_FLIP_SEPARATION_SEC)]

        if stim_path is None:
            msg = "no stim file matched to block: number of vsyncs and diode-box interval can't be checked"
            logger.warning(f"{sync_data} block {block_idx}: {msg}")
            results.append(not_using_diode(msg, None))
            continue

        is_first_frame_on = rising[0] < falling[0]  # may be updated below

        stim_data = npc_stim.stim.get_h5_stim_data(stim_path)
        try:
            vsyncs_per_diode_flip = get_diode_box_frame_interval(stim_data)
            expected_vsyncs = npc_stim.stim.get_total_stim_frames(stim_data)
        finally:
            stim_data.close()

        if vsyncs_per_diode_flip == 1 and expected_vsyncs != len(vsyncs):
            msg = f"{len(vsyncs)} vsyncs, expected {expected_vsyncs} from stim file"
            logger.warning(f"Block {block_idx} has {msg}: Skipping")
            results.append(not_using_diode(msg, vsyncs_per_diode_flip))
            continue

        def concat_flips(rising, falling):
            """Concatenate the rising and falling edges into a single array."""
            return np.sort(np.concatenate((rising, falling)))

        def split_flips(concat_flips, is_first_frame_on):
            """Split the concatenated flips into rising and falling edges."""
            on = concat_flips[::2]
            off = concat_flips[1::2]
            if is_first_frame_on:
                return (on, off)
            else:
                return (off, on)

        flips = concat_flips(rising, falling)
        flips, is_first_frame_on, n_removed = remove_anomalous_diode_flips(
            flips, vsyncs_per_diode_flip, is_first_frame_on
        )
        if n_removed:
            logger.warning(
                f"Removed {n_removed} anomalous short diode intervals in block {block_idx}"
            )

        # one flip per `vsyncs_per_diode_flip` frames. For N > 1 this check didn't exist: a block
        # whose flips were mostly removed above would get almost all of its display times invented
        # at the end of the block (`n_missing` in `get_frame_display_times`). Across 787 box-3
        # blocks the excess after removal is within +/-1 of ceil(n_vsyncs / 3).
        expected_flips = int(np.ceil(len(vsyncs) / vsyncs_per_diode_flip))
        if np.abs(len(flips) - expected_flips) > 5:
            msg = (
                f"{len(flips)} diode flips (after basic filtering), but {len(vsyncs)} vsyncs"
                + (
                    f" (expected {expected_flips} flips with diodeBoxFrameInterval={vsyncs_per_diode_flip})"
                    if vsyncs_per_diode_flip > 1
                    else ""
                )
            )
            logger.warning(f"Block {block_idx} has {msg}: Skipping")
            results.append(not_using_diode(msg, vsyncs_per_diode_flip))
            continue

        rising, falling = split_flips(flips, is_first_frame_on)

        stim_id = f"{stim_path.parent.name}_{block_idx}_{stim_path.stem}"

        try:
            times = get_frame_display_times(
                vsync_times=vsyncs,
                on_flip_times=rising,
                off_flip_times=falling,
                vsyncs_per_flip=vsyncs_per_diode_flip,
                adjust_dropped_frames=True,
                check_lengths=False,
                check_vsyncs=True,
                stim_id=stim_id,
            )
        except (AssertionError, IndexError) as exc:
            if raise_on_error:
                raise
            logger.exception(
                f"Failed to get correct frame display times for {stim_id} block {block_idx}."
                + (
                    "Using vsync times + constant lag"
                    if fallback == "constant_lag"
                    else "No frame display times will be available"
                )
            )
            results.append(
                not_using_diode(
                    f"estimator failed: {exc!r}", vsyncs_per_diode_flip
                )
            )
            continue

        if vsyncs_per_diode_flip == 1:
            while len(times) < len(vsyncs):
                if len(vsyncs) - len(times) > 3:
                    raise AssertionError(
                        f"{stim_id}: Block {block_idx} has {len(times)} frame display times, "
                        f"but {len(vsyncs)} vsyncs: too many missing frames"
                    )
                times = add_missing_diode_flips_for_truncated_sync(times, vsyncs)

            if (len(times) == len(vsyncs) + 1) and times[-1] > vsyncs[-1] + (
                times[:-1] - vsyncs
            ).mean():
                # special case where we have an extra flip at the end, which doesn't correspond to a vsync
                # - can happen if the last frame's sync square is high and the screen then returns to black
                times = times[:-1]
            if len(times) - len(vsyncs) < 5:
                # if we have a few more frames than vsyncs, we can assume these are just extra frames
                # at the end of the block caused by spontaneous flicker, so we remove them
                times = times[: len(vsyncs)]
            if len(times) != len(vsyncs):
                msg = f"{len(times)} frame display times, but {len(vsyncs)} vsyncs"
                logger.error(f"{stim_id}: Block {block_idx} has {msg}")
                results.append(not_using_diode(msg, vsyncs_per_diode_flip))
                continue

        if len(times) == len(vsyncs):
            times, msg = correct_whole_block_frame_offset(times, vsyncs)
            if msg is not None:
                logger.error(f"{stim_id}: Block {block_idx}: {msg}")
                results.append(not_using_diode(msg, vsyncs_per_diode_flip))
                continue

        results.append(
            BlockDisplayTimes(
                block_index=block_idx,
                stim_path=stim_path,
                vsync_times=vsyncs,
                display_times=times + MONITOR_CENTER_REFRESH_TIME,
                source="photodiode",
                diode_box_frame_interval=vsyncs_per_diode_flip,
            )
        )
        logger.info(f"Added frame display times for {stim_path}")

    assert len(results) == len(
        vsync_times_in_blocks
    ), f"{sync_data}: {len(results)} blocks of diode flips matched to {len(vsync_times_in_blocks)} blocks of vsyncs"
    return tuple(results)


# plots -------------------------------------------------------------------- #


def _block_onsets_offsets(
    sync_data: npc_sync.SyncDataset, blocks: Sequence[BlockDisplayTimes]
) -> tuple[list[float], list[float]]:
    """stim_running edges for each block if present, else first/last vsync."""
    on, off = sync_data.stim_running_edges
    onsets, offsets = [], []
    for b in blocks:
        v = b.vsync_times
        if on.size:
            i = int(np.searchsorted(on, v[0], side="right") - 1)
            onsets.append(float(on[max(i, 0)]))
            offsets.append(float(off[max(i, 0)]))
        else:
            onsets.append(float(v[0]))
            offsets.append(float(v[-1]))
    return onsets, offsets


def _plot_block_ends(
    sync: npc_sync.SyncPathOrDataset,
    blocks: Sequence[BlockDisplayTimes],
    which: Literal["onsets", "offsets"],
) -> matplotlib.figure.Figure:
    import matplotlib.pyplot as plt

    sync_data = npc_sync.get_sync_data(sync)
    onsets, offsets = _block_onsets_offsets(sync_data, blocks)
    fig, _ = plt.subplots(len(blocks))
    axes = fig.axes
    fig.suptitle(f"Stim {which}")
    for ind, block in enumerate(blocks):
        vs = block.vsync_times
        dvs = block.display_times if block.display_times is not None else np.array([])
        labels = []
        axes[ind].plot(vs, 0.5 * np.ones(len(vs)), "|")
        labels.append("stim vsyncs")
        axes[ind].plot(dvs, 0.5 * np.ones(len(dvs)), "|", ms=20)
        labels.append(f"frame display (estimated, {block.source})")
        if which == "onsets":
            # focus on the start of vsyncs if they occur well after the stim-TTL onset
            son = onsets[ind]
            x0, x1 = max(son - 1, vs[0] - 1), max(son + 2, vs[0] + 2)
        else:
            # focus on the end of vsyncs if they occur well before the stim-TTL offset
            soff = offsets[ind]
            x0, x1 = min(soff - 2, vs[-1] - 2), min(soff + 1, vs[-1] + 1)
        sync_data.plot_bit(4, x0, x1, axes=axes[ind], auto_show=False)
        labels.append("diode-measured sync square")
        sync_data.plot_bit(5, x0, x1, axes=axes[ind], auto_show=False)
        labels.append("stim running or ends of block")
        axes[ind].set_xlim((x0, x1))
        axes[ind].legend(
            labels,
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(labels),
            fancybox=True,
        )
        if len(fig.axes) > 1:
            name = block.stim_path.stem if block.stim_path else "no stim file"
            axes[ind].set_title(f"block {block.block_index}: {name}", fontsize=8)
            legend = axes[ind].get_legend()
            if ind > 0 and legend is not None:
                legend.remove()
    fig.set_size_inches(10, 5 * len(fig.axes))
    fig.subplots_adjust(hspace=0.3)
    return fig


def plot_stim_onsets(
    sync: npc_sync.SyncPathOrDataset, blocks: Sequence[BlockDisplayTimes]
) -> matplotlib.figure.Figure:
    """vsyncs, estimated display times, diode and stim_running at the start of each block.
    `blocks` from `get_display_times`."""
    return _plot_block_ends(sync, blocks, "onsets")


def plot_stim_offsets(
    sync: npc_sync.SyncPathOrDataset, blocks: Sequence[BlockDisplayTimes]
) -> matplotlib.figure.Figure:
    """vsyncs, estimated display times, diode and stim_running at the end of each block.
    `blocks` from `get_display_times`."""
    return _plot_block_ends(sync, blocks, "offsets")


def plot_diode_measured_sync_square_flips(
    sync: npc_sync.SyncPathOrDataset,
) -> matplotlib.figure.Figure:
    """Plot the diode-measured sync-square flip intervals in each stim_running block."""
    import matplotlib.pyplot as plt

    sync_data = npc_sync.get_sync_data(sync)
    all_diode_flips = np.concatenate(
        [
            sync_data.get_rising_edges("stim_photodiode", units="seconds"),
            sync_data.get_falling_edges("stim_photodiode", units="seconds"),
        ]
    )
    all_diode_flips.sort()

    frequency = get_expected_diode_flip_rate(sync_data)
    expected_period = 1 / frequency

    # get the intervals in parts (one for each stimulus)
    diode_flips_per_stim = sync_data.divide_into_stim_running_blocks(all_diode_flips)

    num_diode_flips_per_stim = np.array([len(_) for _ in diode_flips_per_stim])
    fig, _ = plt.subplots(
        1,
        len(diode_flips_per_stim),
        sharey=True,
        gridspec_kw={
            "width_ratios": num_diode_flips_per_stim / min(num_diode_flips_per_stim)
        },
    )
    fig.suptitle(f"diode-measured sync-square flip intervals, {expected_period = } s")
    y_deviations_from_expected_period: list[float] = []
    for idx, (ax, d) in enumerate(zip(fig.axes, diode_flips_per_stim)):
        # add horizontal line at expected period
        ax.axhline(expected_period, linewidth=0.5, c="k", linestyle="--", alpha=0.3)
        plt.sca(ax)
        intervals = np.diff(d)
        times = np.diff(d) / 2 + d[:-1]  # plot at mid-point of interval
        markerline, stemline, baseline = plt.stem(
            times, intervals, bottom=expected_period
        )
        plt.setp(stemline, linewidth=0.5, alpha=0.3)
        plt.setp(markerline, markersize=0.5, alpha=0.8)
        plt.setp(baseline, visible=False)

        y_deviations_from_expected_period.append(max(intervals - expected_period))
        y_deviations_from_expected_period.append(max(expected_period - intervals))
        if len(fig.axes) > 1:
            ax.set_title(f"stim {idx}", fontsize=8)
        ax.set_xlabel("time (s)")
        if idx == 0:
            ax.set_ylabel("flip interval (s)")
        ax.set_xlim(min(d) - 20, max(d) + 20)

    for ax in fig.axes:
        # after all ylims are established
        ax.set_ylim(
            bottom=max(
                0,
                expected_period - np.max(np.abs(y_deviations_from_expected_period)),
            ),
        )
        ticks_with_period = sorted(set(ax.get_yticks()) | {expected_period})
        ax.set_yticks(ticks_with_period)
        ax.set_yticklabels([f"{_:.3f}" for _ in ticks_with_period])
    fig.set_layout_engine("tight")

    return fig


# estimator ----------------------------------------------------------------- #


def correct_whole_block_frame_offset(
    display_times: npt.NDArray[np.float64],
    vsyncs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], str | None]:
    """Check the block's median vsync-to-diode-flip latency against
    `AVERAGE_VSYNC_TO_DIODE_FLIP_TIME` and correct a whole-block offset of +1 frame.

    If the first diode flip of a block is missed (e.g. white-to-white transition from the
    initializing screen, or flicker before the first frame), every frame is assigned the next
    frame's flip, and the end of the block is padded. `get_frame_display_times` corrects this
    with a vote over long (dropped-frame) intervals, but that vote is skipped for Spontaneous*
    stims and can't run on blocks with few long intervals; enabling it for Spontaneous* stims
    rejects blocks whose vsync falling edges are delayed (stall after `flip()`), so this check
    uses the latency instead. Across 1,899 photodiode blocks without the offset the block
    median latency is 27.2-32.5 ms (incl. `MONITOR_CENTER_REFRESH_TIME`); with it, ~44 ms.

    Returns (display_times, None) if the median latency is within half a frame of the average
    (unchanged) or one frame more (shifted: the first frame is inserted one median frame
    interval before the first flip and the last, padded, frame is dropped). Otherwise returns
    (display_times, reason) and the block should not use the photodiode.

    >>> v = np.arange(100) / 59.95
    >>> t, msg = correct_whole_block_frame_offset(v + AVERAGE_VSYNC_TO_DIODE_FLIP_TIME, v)
    >>> msg is None and bool(np.array_equal(t, v + AVERAGE_VSYNC_TO_DIODE_FLIP_TIME))
    True
    >>> t, msg = correct_whole_block_frame_offset(v + AVERAGE_VSYNC_TO_DIODE_FLIP_TIME + 1 / 59.95, v)
    >>> msg is None and bool(np.allclose(t - v, AVERAGE_VSYNC_TO_DIODE_FLIP_TIME))
    True
    >>> correct_whole_block_frame_offset(v + 0.1, v)[1]
    'median vsync-to-diode-flip latency is 100.0 ms: 5 frames more than expected (19.6 ms)'
    """
    latency = float(np.median(display_times - vsyncs))
    n_frames = round((latency - AVERAGE_VSYNC_TO_DIODE_FLIP_TIME) / FRAME_INTERVAL)
    if n_frames == 0:
        return display_times, None
    first_flip_latency = display_times[0] - vsyncs[0]
    if n_frames == 1 and first_flip_latency > AVERAGE_VSYNC_TO_DIODE_FLIP_TIME + 0.5 * FRAME_INTERVAL:
        logger.warning(
            f"median vsync-to-diode-flip latency is {latency * 1e3:.1f} ms, one frame more than "
            f"expected: assuming the first diode flip was missed. Inserting the first frame and "
            "dropping the last"
        )
        interval = float(np.median(np.diff(display_times)))
        return np.concatenate(([display_times[0] - interval], display_times[:-1])), None
    return display_times, (
        f"median vsync-to-diode-flip latency is {latency * 1e3:.1f} ms: {n_frames} frames "
        f"{'more' if n_frames > 0 else 'less'} than expected ({AVERAGE_VSYNC_TO_DIODE_FLIP_TIME * 1e3:.1f} ms)"
    )


def adjust_diode_flip_intervals(
    diode_flips: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    diode flip intervals have a bimodal distribution due to asymmetry of
    photodiode thresholding: adjust every other interval to get a closer
    estimate of actual transition time for each diode flip
    """
    diode_flips = np.array(diode_flips)  # make a copy
    original_intervals = np.diff(diode_flips)
    outliers = np.logical_or(
        np.percentile(original_intervals, 5) > original_intervals,
        original_intervals > np.percentile(original_intervals, 95),
    )
    intervals = original_intervals[~outliers]

    # the two distributions are symmetric about the mean
    deviation = np.mean(np.abs(intervals - np.mean(intervals)))
    if np.mean(original_intervals[0::2]) > np.mean(original_intervals[1::2]):
        sign = 1
    else:
        sign = -1

    for idx in range(1, len(diode_flips) - 1, 2):
        # alternate on every short/long interval and expand/contract
        # interval
        diode_flips[idx] -= sign * 0.5 * deviation
        diode_flips[idx + 1] += sign * 0.5 * deviation
    return diode_flips


def add_missing_diode_flips_for_truncated_sync(
    diode_flips: npt.NDArray,
    vsyncs: npt.NDArray,
) -> npt.NDArray[np.float64]:
    approx_interval = np.mean(diode_flips - vsyncs[: len(diode_flips)])
    idx = np.searchsorted(vsyncs, diode_flips[-1], side="left")
    while diode_flips[-1] < vsyncs[-1] + approx_interval:
        logger.info(
            "Creating extra diode flip at end of stim block to account for truncated sync recording"
        )
        diode_flips = np.array([*diode_flips, vsyncs[idx - 1] + approx_interval])
        idx += 1
    return diode_flips


def add_missing_diode_flip_at_stim_onset(
    diode_flips: npt.NDArray,
    vsyncs: npt.NDArray,
) -> npt.NDArray[np.float64]:
    """
    Transition from white screen to white sync-square on first frame results
    in a missing diode flip (same for black-to-black).
    Create a flip after the first vsync with an interval
    based on the statistics of other intervals in the
    recording.
    Shouldn't have any side-effects, and the first frame likely contains nothing important.
    """
    logger.info(
        "Creating extra diode flip at start of stim block to account for white-to-white or black-to-black transition"
    )
    common_len = min([len(vsyncs) - 1, len(diode_flips)])
    avg_vsync_to_flip_interval = np.median(
        adjust_diode_flip_intervals(diode_flips[:common_len])
        - vsyncs[1 : common_len + 1]
    )
    # TODO could be more accurate by adding short/long interval as appropriate
    # - diode_flips are pre-adjustment, but we're currently adding the expected
    #   post-adjustment interval
    t = vsyncs[0] + avg_vsync_to_flip_interval
    assert t < diode_flips[0]
    return np.array([t, *diode_flips])


def discard_erroneous_diode_flips_at_stim_offset(
    diode_flips: npt.NDArray, vsyncs: npt.NDArray
) -> npt.NDArray[np.float64]:
    """
    - after a stimulus the screen usually turns grey, causing
    the diode to flip at least one more time after all stim
    vsyncs are finished.
    - the grey screen itself may be on the threshold of diode activation, causing
    multiple additional flips

    remove the last flip if doesn't look time-locked with a
    vsync (ie. vsync-to-flip interval is an outlier)
    """
    while len(diode_flips[diode_flips > vsyncs[-1]]) > 1 and len(diode_flips) > len(
        vsyncs
    ):
        diode_flips = diode_flips[:-1]
    return diode_flips


def get_frame_display_times(
    vsync_times: npt.NDArray,
    on_flip_times: npt.NDArray,
    off_flip_times: npt.NDArray,
    vsyncs_per_flip: int,
    is_first_frame_off: bool | None = None,
    adjust_dropped_frames: bool = True,
    check_lengths: bool = True,
    check_vsyncs: bool = True,
    stim_id: str | None = None,
) -> npt.NDArray[np.float64]:
    # first, shift vsyncs forwards in time by the monitor latency, so we can more easily match up
    # diode flips and vsyncs
    vsync_times = vsync_times + AVERAGE_VSYNC_TO_DIODE_FLIP_TIME
    if vsyncs_per_flip == 1:
        # default to using ON flip times (half length of vsyncs) upsampled 2x to put OFF frame halfway between ON
        # frames
        # - if ON flip interval is longer than two screen refreshes (ie a "dropped" frame), check
        #   whether the two corresponding vsync intervals show one longer and one shorter, then use this
        #   to determine the position of the OFF frame
        # - if the first frame is OFF, we can't adjust the first vsync time: we have to create it

        # some checks to make sure arrays are the expected lengths:
        if is_first_frame_off is None and off_flip_times is None:
            raise ValueError(
                "Either is_first_frame_off or off_flip_times must be provided to get_frame_display_times"
            )
        if is_first_frame_off is None:
            is_first_frame_off = off_flip_times[0] < on_flip_times[0]
        if off_flip_times is not None:
            is_last_frame_on = on_flip_times[-1] > off_flip_times[-1]
        elif len(vsync_times) % 2:
            is_last_frame_on = not is_first_frame_off
        else:
            is_last_frame_on = is_first_frame_off
        if check_lengths:
            # lengths aren't actually critical for the function to work
            assert (
                len(vsync_times)
                == len(on_flip_times) * 2 + is_first_frame_off - is_last_frame_on
            )
        if off_flip_times is not None:
            assert (
                len(off_flip_times)
                == len(on_flip_times) + is_first_frame_off - is_last_frame_on
            )
            assert (off_flip_times[0] < on_flip_times[0]) == is_first_frame_off
            assert (on_flip_times[-1] > off_flip_times[-1]) == is_last_frame_on
        if check_lengths:
            display_times = np.full(len(vsync_times), np.nan)
        else:
            display_times = np.full(
                len(on_flip_times) * 2 + is_first_frame_off - is_last_frame_on, np.nan
            )

        on_intervals = np.diff(on_flip_times)

        # Create OFF frame times as midpoints between ON frames
        fake_off_flip_times = on_flip_times[:-1] + 0.5 * on_intervals
        if not is_last_frame_on:
            # the last OFF flip happens after the last ON flip - currently missing a time value
            fake_off_flip_times = np.append(
                fake_off_flip_times, fake_off_flip_times[-1] + on_intervals[-1]
            )

        if is_first_frame_off:
            fake_off_flip_times = np.insert(
                fake_off_flip_times, 0, fake_off_flip_times[0] - on_intervals[0]
            )

        if adjust_dropped_frames:
            if off_flip_times is None:
                raise ValueError(
                    "off_flip_times must be provided if adjust_dropped_frames is True"
                )

            median_refresh_interval = np.median(on_intervals) / 2

            for idx in np.where(on_intervals > 2.5 * median_refresh_interval)[0]:

                # get the off frame idx we're going to adjust:
                off_frame_idx = idx + is_first_frame_off

                # check for correctness so we can raise in debugger and examine values from previous loop:
                if is_first_frame_off:
                    assert all(
                        (
                            on_flip_times[:off_frame_idx]
                            - fake_off_flip_times[:off_frame_idx]
                        )
                        > 0
                    ), "Adjusting OFF flips in previous loop has disturbed order of ON-OFF sequences"
                else:
                    assert all(
                        (
                            fake_off_flip_times[:off_frame_idx]
                            - on_flip_times[:off_frame_idx]
                        )
                        > 0
                    ), "Adjusting OFF flips in previous loop has disturbed order of ON-OFF sequences"

                n_refresh_in_interval: int = round(
                    on_intervals[idx] / median_refresh_interval
                )
                assert (
                    n_refresh_in_interval >= 3
                ), f"should only be considering long intervals here (regular intervals are 2): {n_refresh_in_interval=}"

                #!
                # TODO check that we're considering frames that are dropped for more than one refresh
                # if there's more than one and we don't do it correctly we'll create a spurious long
                # interval following the real dropped interval
                #!

                ## try just using position of OFF flip - ignore vsyncs
                frames_to_off = (
                    off_flip_times[off_frame_idx] - on_flip_times[idx]
                ) / median_refresh_interval
                n_frames_to_off = asym_round(frames_to_off, 0.9, allow_zero=False)
                assert n_frames_to_off >= 1, "Trying to place OFF at ON frame"
                assert (
                    n_frames_to_off < n_refresh_in_interval
                ), "Trying to place OFF beyond ON-ON interval"
                fake_off_flip_times[off_frame_idx] = on_flip_times[
                    idx
                ] + n_frames_to_off * (on_intervals[idx] / n_refresh_in_interval)

        # Fill frame times
        display_times[0::2] = (
            fake_off_flip_times if is_first_frame_off else on_flip_times
        )
        display_times[1::2] = (
            on_flip_times if is_first_frame_off else fake_off_flip_times
        )

        common_len = min(len(vsync_times), len(display_times))
        if get_plot_dir():
            import matplotlib.pyplot as plt

            stim_dir = get_plot_dir() / f'{stim_id or "stim"}'
            stim_dir.mkdir(parents=True, exist_ok=True)

            plt.figure()
            plt.plot(display_times[:common_len] - vsync_times[:common_len])
            plt.title(f"display times minus vsync times\n{stim_id=}")
            plt.xlabel("frame idx")
            plt.ylabel("display time - vsync time (s)")
            plt.savefig(stim_dir / "display_times_minus_vsyncs.png")
            plt.close()

            plt.figure()
            plt.hist(np.diff(vsync_times), label="vsyncs", bins=100, log=True)
            plt.hist(np.diff(display_times), label="display_times", bins=100, log=True)
            plt.legend()
            plt.xlim(0.0, median_refresh_interval * 3.5)
            plt.xlabel("interval (s)")
            plt.ylabel("count")
            plt.title(f"vsyncs and display times intervals\n{stim_id=}")
            plt.savefig(stim_dir / "vsync_display_time_hist.png")
            plt.close()

        long_len = 2  # frames
        if (
            adjust_dropped_frames
            and check_vsyncs
            and "spontaneous" not in (stim_id or "").lower()
        ):  # quiescent frame timings don't matter
            long_thr = long_len - 0.9
            long_display_time_idx = np.where(
                np.diff(display_times) > long_thr * median_refresh_interval
            )[0]
            # vsyncs should be long at these same idx
            # - vsyncs intervals aren't discretized, so it's difficult to find a threshold for a dropped frame
            # - just check the vsyncs that correspond to lon display times (which are discretized)
            all_vsync_diffs = np.diff(vsync_times)
            # all_vsync_diffs =  script_frame_intervals #! <---- if unsure about vsyncs can check against script intervals - so far this always results in the same offsets (eg long intervals are synced between script and vsyncs)
            long_vsync_misalignment = []
            corresponding_long_vsyncs = []
            max_shift = 6
            for idx in long_display_time_idx:
                shifts = list(range(0, -max_shift, -1)) + list(range(1, max_shift))
                shifts = [
                    shift
                    for shift in shifts
                    if (idx + shift > 0) and (idx + shift < len(all_vsync_diffs))
                ]
                shifted_vsync_diffs = [all_vsync_diffs[idx + shift] for shift in shifts]
                shift_idx = np.argmax(shifted_vsync_diffs)
                if (
                    v := shifted_vsync_diffs[shift_idx]
                ) < long_thr * median_refresh_interval:
                    raise IndexError(
                        f"{stim_id}: Could not find long vsync for {idx=} with {max_shift=} either side of long frame display time: signals have become too misaligned"
                    )
                long_vsync_misalignment.append(
                    shifts[shift_idx] * -1
                )  # inv sign to express vsync idx relative to flip idx (ie -1 means long vsync idx is 1 less than long diode flip idx)
                corresponding_long_vsyncs.append(v)
            long_vsync_misalignment = np.array(long_vsync_misalignment)
            if get_plot_dir():
                plt.figure()
                plt.step(long_display_time_idx, long_vsync_misalignment)
                plt.xlabel("frame idx")
                plt.ylabel(
                    "misalignment with vsyncs\n(+ve means index of long flips is shifted to right relative to vsyncs)"
                )
                plt.title(
                    f"long frame display times misalignment with vsyncs\n{stim_id=}"
                )
                plt.savefig(stim_dir / "long_frame_misalignment_with_vsyncs.png")
                plt.close()

            smoothed_long_vsync_misalignment = scipy.signal.medfilt(
                long_vsync_misalignment, kernel_size=3
            )
            # tolerate some misalignment:
            # - we often get a few frames that are misaligned individually, but not sequentially
            # - can't currently explain them, but if they don't cause a systematic shift then we can't correct them
            min_long_interval_num = 3
            if len(long_vsync_misalignment) < min_long_interval_num:
                # check vsyncs or stim file also have few long intervals
                assert (
                    x := len(
                        all_vsync_diffs[
                            all_vsync_diffs > long_thr * median_refresh_interval
                        ]
                    )
                ) < min_long_interval_num * 2, f"{stim_id}: found fewer than {min_long_interval_num} long intervals in photodiode signal, but {x} long intervals in vsync signal"
                pass
            elif len(
                smoothed_long_vsync_misalignment[smoothed_long_vsync_misalignment == 1]
            ) > 0.90 * len(smoothed_long_vsync_misalignment):
                logger.warning(
                    f"All {len(smoothed_long_vsync_misalignment)} long frame display times are misaligned with vsyncs by +1 frame: we have an extra diode flip at the start of the stim block to remove"
                )
                #! remember we shifted vsyncs at the top of the function
                assert (
                    display_times[0] < vsync_times[0]
                ), f"Expected first display time to have lower than usual latency after first vsync: got {x} s"
                display_times = display_times[1:]  # remove the first frame
            elif len(
                smoothed_long_vsync_misalignment[smoothed_long_vsync_misalignment == -1]
            ) > 0.90 * len(smoothed_long_vsync_misalignment):
                # - first sync square used to be white, which would usually be detected when transitioning from the "initializing" grey screen, so we'd be missing the diode flip
                # corresponding to the first vsync.
                # - this was changed to always be black first on Aug 25 2023: https://github.com/samgale/DynamicRoutingTask/commit/cb6af876f4e42ddef820a8aea0f955faa8dfcb68
                # - it's still possible to miss the first sync square frame when black if the phantom flickering during the initializing grey screen caused a black frame to precede the first frame
                logger.warning(
                    f"All {len(smoothed_long_vsync_misalignment)} long frame display times are misaligned with vsyncs by -1 frame: likely missed a diode flip at the start of the stim block"
                )
                #! remember we shifted vsyncs at the top of the function
                assert (
                    x := (display_times[0] - vsync_times[0])
                ) > 0.5 * FRAME_INTERVAL, f"Expected first display time to have higher than usual latency after first vsync: got {MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC + x} s"
                display_times = np.insert(
                    display_times, 0, display_times[0] - median_refresh_interval
                )  # add the first frame

            elif len(
                smoothed_long_vsync_misalignment[smoothed_long_vsync_misalignment == 0]
            ) < 0.75 * len(long_vsync_misalignment):
                raise AssertionError(
                    f"{stim_id}: Too many frames are misaligned with vsyncs - check plots."
                )
                # get longest run of non-zero elements (misaligned idx):
                misaligned_idx = np.where(long_vsync_misalignment != 0)[0]
                first_mismatch_idx = [
                    i for i, v in enumerate(long_vsync_misalignment) if v != 0
                ][0]
                first_long_mismatch = long_display_time_idx[first_mismatch_idx]
                last_idx_before_mismatch = (
                    None
                    if first_mismatch_idx == 0
                    else long_display_time_idx[first_mismatch_idx - 1]
                )

        # common_len = min(len(vsync_times), len(display_times))
        # assert np.all(display_times[:common_len] - vsync_times[:common_len] > 0)
        ## ^ at the moment this may be violated (very infrequently, and only on a couple of frames in a block) so it's disabled
    else:
        # photodiode signal changes every N screen refreshes (here N = 3)
        # - when frames are "dropped", N is extended by some number of frames (typically 1 but can be
        #   more). We have reliable ON times and must work out when the dropped frames ocurred
        #    _____________           _____________              _____________
        #    |   :   :   |   :   :   |   :   :   |   :      :   |   :   :   |
        # ___|   :   :   |___:___:___|   :   :   |___:______:___|   :   :   |__
        #                                                ^ 1 "dropped" frame

        # the OFF edge times are unreliable, so first use ON edges only, and discretize OFF edges to
        # be some multiple of screen refreshes from preceding ON edge:
        if on_flip_times[0] > off_flip_times[0]:
            raise NotImplementedError(
                f"{stim_id} | first diode edge in block is falling - this should not be possible with default TaskControl settings."
                "Either the min vsync-diode flip interval needs shortening or we have spontaneous flicker. Needs handling."
            )

        n_on_to_off_refreshes = np.round(
            (off_flip_times - on_flip_times[: len(off_flip_times)]) / FRAME_INTERVAL
        )
        assert len(n_on_to_off_refreshes) == len(off_flip_times) <= len(on_flip_times)
        assert min(n_on_to_off_refreshes[:-1]) >= vsyncs_per_flip
        adjusted_off_flip_times = on_flip_times[: len(off_flip_times)] + (
            n_on_to_off_refreshes * FRAME_INTERVAL
        )
        assert np.all(adjusted_off_flip_times > on_flip_times)
        flips = np.sort(np.concatenate([on_flip_times, adjusted_off_flip_times]))

        # - flips wont be exactly N-frames long, but must be longer than (N-1 + some variability)
        #   frames
        # - only check up to last flip, incase it's a rising edge caused by transition to desktop
        assert np.all(
            np.diff(flips[:-1]) > ((vsyncs_per_flip - 0.75) * FRAME_INTERVAL)
        ), f"{stim_id} has diode flip intervals that are too short: {min(np.diff(flips)) * FRAME_RATE = :.3f}, {np.mean(np.diff(flips)) * FRAME_RATE = :.3f} (expected {vsyncs_per_flip})"

        # now work out actual frame display times between each rising-falling edge pair
        display_times = []
        regular_intervals = np.arange(
            0, vsyncs_per_flip * FRAME_INTERVAL, FRAME_INTERVAL
        )
        n_refreshes_per_interval = np.round(np.diff(flips) / FRAME_INTERVAL)
        assert len(n_refreshes_per_interval) == len(flips[:-1])
        for idx, (flip, n_refreshes) in enumerate(
            zip(flips[:-1], n_refreshes_per_interval)
        ):
            # last flip is skipped, which is what we would want in the ideal case that
            # n_frames % (2 * vsyncs_per_flip) < vsyncs_per_flip, AND the inter-stim screen is black
            # e.g.
            #    _____________ <-- last OFF is not a stim frame but transition to inter-stim black
            #    |   :   :   |
            # ___|   :   :   |___________________
            # doesn't hold if the last sync square is black: deal with this after the loop
            if display_times:
                assert (
                    flip > display_times[-1]
                ), f"{stim_id} {idx=} | diode flips are not strictly increasing"

            n_dropped = n_refreshes - vsyncs_per_flip
            if n_dropped == 0:
                display_times.extend(flip + regular_intervals)
                continue
            if n_dropped < 0 and idx == (len(n_refreshes_per_interval) - 1):
                # short last interval as stim stops before N presentations of sync square
                display_times.extend(flip + regular_intervals[: int(n_dropped)])
                continue

            # use corresponding vsync intervals to determine which frame(s) were dropped
            vsync_intervals = np.diff(
                vsync_times[idx * vsyncs_per_flip : (idx + 1) * vsyncs_per_flip + 1]
            )
            if n_dropped == 1:
                long_idx = [np.argmax(vsync_intervals)]
                long_refreshes = [1]
            else:
                assert n_dropped > 0
                # figure out how many frames were dropped,
                # and how many screen refreshes dropped for each
                long_thr = 1.1 * FRAME_INTERVAL
                long_idx = np.where(vsync_intervals > long_thr)[0].tolist()
                long_refreshes = np.round(vsync_intervals[long_idx] / FRAME_INTERVAL)
                # long_refreshes is number of screen refreshes for each dropped frame
                if sum(long_refreshes) != n_dropped:
                    if len(long_refreshes) == 1:
                        assert long_refreshes[0] != n_dropped
                        # we found a long vsync, but it rounds up to more frames than expected
                        long_refreshes[0] = n_dropped
                    elif len(long_refreshes) > 1:  #
                        # we found multiple long vsyncs, but they don't sum to the expected number
                        # of dropped frames - use the relative lengths to assign dropped frames
                        prop_refreshes = long_refreshes / sum(long_refreshes)
                        long_refreshes = np.ceil(prop_refreshes * n_dropped).astype(int)
                        while sum(long_refreshes) > n_dropped:
                            # remove one from the largest
                            long_refreshes[np.argmax(long_refreshes)] -= 1
                        assert np.all(long_refreshes > 0)
                    else:
                        msg = f"{stim_id} | Faulty locating of dropped frames - something doesn't add up: {vsync_intervals=}, {long_refreshes=} ({long_thr=})"
                        if (
                            vsyncs_per_flip > 30
                        ):  # we're potentially mis-aligning many frames here
                            raise AssertionError(msg)
                        logger.warning(msg)
                        # put one long frame in the middle of the long flip interval, which is the best we can
                        # do on average
                        long_idx = [vsyncs_per_flip // 2]
                        long_refreshes = [n_dropped]

            evenly_spaced_times = np.linspace(
                flip, flips[idx + 1], int(n_refreshes) + 1, endpoint=True
            )[:-1]
            assert np.all(
                np.isclose(np.diff(evenly_spaced_times) / FRAME_INTERVAL, 1, rtol=0.05)
            )
            new_times = []
            i = -1
            for n in range(vsyncs_per_flip):
                i += 1
                if i - 1 in long_idx:
                    i += int(long_refreshes[long_idx.index(i - 1)])
                t = evenly_spaced_times[i]
                new_times.append(t)
                # if the last interval is a long one nothing happens: inserting the next flip will
                # produce the long interval
            assert len(new_times) == vsyncs_per_flip
            assert flips[idx + 1] not in new_times, "prematurely added next flip"
            display_times.extend(new_times)

        # at this point, we could have up to vsync_intervals -1 frames not enclosed by a diode flip at
        # the end of the experiment, for example:
        #    _____________           _____________________ <-- display stays bright after last frame
        #    |   :   :   |   :   :   |   :   :
        # ___|   :   :   |___:___:___|   :   :
        #    _____________
        #    |   :   :   |   :   :
        # ___|   :   :   |___:___:______________
        n_missing = len(vsync_times) - len(display_times)

        if n_missing > 0:
            assert (
                flips[-1] not in display_times
            ), "already used last flip in display times"
            if n_missing > vsyncs_per_flip:
                logger.warning(
                    f"More missing frames ({n_missing}) than vsyncs per flip ({vsyncs_per_flip}) - indicates that this is something different to the last flip being missed"
                )
            display_times.extend(
                flips[-1] + np.arange(0, n_missing * FRAME_INTERVAL, FRAME_INTERVAL)
            )

    assert not np.any(np.isnan(display_times))
    assert np.all(np.diff(display_times) > 0)

    return np.array(display_times)


def is_part_of_normal_pair(
    prev_interval_dur, suspicious_interval_dur, next_interval_dur
) -> bool:
    """
    Normal pairs sum to 2 frames (without any dropped frames), and typically alternate long-short intervals:
    >>> is_part_of_normal_pair(1.3 / 59.95, 0.7 / 59.95, 1.3 / 59.95)
    True

    Should be robust to short intervals on either side of the suspicious interval:
    >>> is_part_of_normal_pair(0.05 / 59.95, 0.7 / 59.95, 1.3 / 59.95)
    True
    >>> is_part_of_normal_pair(1.3 / 59.95, 0.7 / 59.95, 0.05 / 59.95)
    True

    Should allow for an integer number of frames to accommodate dropped frames:
    >>> is_part_of_normal_pair(2.3 / 59.95, 0.7 / 59.95, 2.3 / 59.95)
    True
    >>> is_part_of_normal_pair(1.3 / 59.95, 1.7 / 59.95, 0.1 / 59.95)
    True

    Must detect anomalous intervals that don't sum to an integer number of frames:
    >>> is_part_of_normal_pair(1.3 / 59.95, 0.01 / 59.95, 0.7 / 59.95)
    False

    Must detect anomalous intervals that don't sum to an integer number of frames:
    >>> is_part_of_normal_pair(2.3 / 59.95, 0.01 / 59.95, 0.7 / 59.95)
    False

    Must be robust to short intervals on either side of the suspicious interval:
    >>> is_part_of_normal_pair(1.3 / 59.95, 0.01 / 59.95, 0.01 / 59.95)
    False
    >>> is_part_of_normal_pair(0.01 / 59.95, 0.01 / 59.95, 1.3 / 59.95)
    False
    """
    if (suspicious_interval_dur * FRAME_RATE) < DEFINITE_SHORT_INTERVAL_THRESHOLD:
        # if the suspicious interval is extremely short, it can't be real
        return False

    result = False
    for other in (prev_interval_dur, next_interval_dur):
        if other is None:
            # if there's no next or previous interval, we can't check the pair
            continue
        interval_pair_sum = suspicious_interval_dur + other

        # if the sum of the two intervals is less than 2 frames it's anomalous
        if interval_pair_sum < (2 * FRAME_INTERVAL) - SUSPICIOUS_INTERVAL_THRESHOLD:
            continue

        # if the sum of the two intervals is not close to an integer number of frames, it's  anomalous
        if round(interval_pair_sum * FRAME_RATE, 1) % 1 != 0:
            continue

        # after checking previous, we can assume the pair is normal
        result = True
        break

    return result


def _is_anomalous_interval(
    flips: list[float], idx: int, num_vsyncs_per_diode_flip: float
) -> bool:
    """Same test as `anomalous_interval_indices` for the single interval `flips[idx + 1] - flips[idx]`."""
    n_intervals = len(flips) - 1
    interval = flips[idx + 1] - flips[idx]
    if not (
        interval
        < ((num_vsyncs_per_diode_flip - 1) * FRAME_INTERVAL)
        + SUSPICIOUS_INTERVAL_THRESHOLD
    ):
        return False
    if num_vsyncs_per_diode_flip > 1 and idx == n_intervals - 1:
        return False
    if num_vsyncs_per_diode_flip == 1 and is_part_of_normal_pair(
        prev_interval_dur=flips[idx] - flips[idx - 1] if idx - 1 >= 0 else None,
        suspicious_interval_dur=interval,
        next_interval_dur=(
            flips[idx + 2] - flips[idx + 1] if idx + 1 < n_intervals else None
        ),
    ):
        return False
    return True


def remove_anomalous_diode_flips(
    flips: npt.ArrayLike,
    num_vsyncs_per_diode_flip: int,
    is_first_frame_on: bool,
) -> tuple[npt.NDArray[np.float64], bool, int]:
    """Repeatedly delete the pair of flips enclosing the first anomalous interval (see
    `anomalous_interval_indices`) until none are left, toggling `is_first_frame_on` when the
    first interval is deleted. Returns (flips, is_first_frame_on, n_pairs_removed).

    Gives the same result as the original loop (npc_sync <= 0.2.9)::

        while any(anomalous_interval_indices(flips, n)):
            indices = anomalous_interval_indices(flips, n)
            flips = np.delete(flips, slice(indices[0], indices[0] + 2))
            if 0 in indices:
                is_first_frame_on = not is_first_frame_on

        including its quirk that an anomaly at index 0 alone doesn't trigger a deletion
        (`any([0])` is False), but is deleted first when any other anomaly exists.

    The original rescanned the whole block after every deletion (O(n^2): >30 min for a block
    with 54k anomalous intervals). Deleting flips i, i+1 only changes the status of intervals
    i-2 and i-1 (each interval's status depends on itself and its neighbours), so after a
    deletion the scan resumes from i-2.

    >>> f = np.array([0.0, 1, 2, 2.001, 3, 4]) / 59.95
    >>> out, on, n = remove_anomalous_diode_flips(f, 1, True)
    >>> (np.round(out * 59.95, 3).tolist(), on, n)
    ([0.0, 1.0, 3.0, 4.0], True, 1)
    """
    f = [float(x) for x in np.asarray(flips)]
    n_removed = 0
    start = 1
    while True:
        zero_is_anomalous = len(f) > 1 and _is_anomalous_interval(
            f, 0, num_vsyncs_per_diode_flip
        )
        first_nonzero = next(
            (
                i
                for i in range(start, len(f) - 1)
                if _is_anomalous_interval(f, i, num_vsyncs_per_diode_flip)
            ),
            None,
        )
        if first_nonzero is None:
            break
        idx = 0 if zero_is_anomalous else first_nonzero
        del f[idx : idx + 2]
        n_removed += 1
        if idx == 0:
            is_first_frame_on = not is_first_frame_on
            start = 1
        else:
            start = max(1, idx - 2)
    return np.array(f, dtype=np.float64), is_first_frame_on, n_removed


def anomalous_interval_indices(flips, num_vsyncs_per_diode_flip: float):
    intervals = np.diff(flips)
    short_intervals = []
    for idx in sorted(
        np.where(
            intervals
            < ((num_vsyncs_per_diode_flip - 1) * FRAME_INTERVAL)
            + SUSPICIOUS_INTERVAL_THRESHOLD
        )[0]
    ):
        if idx in short_intervals:
            continue
        if num_vsyncs_per_diode_flip > 1 and idx == len(intervals) - 1:
            continue  # the last interval can be short as it returns to dark screen before N stim frames

        # a short blip on the line must occur as an ON-OFF pair, due to the way sync detects rising/falling edges (e.g. cannot have to rising)
        # ie both indices must be present, so we can delete the first one

        if num_vsyncs_per_diode_flip == 1 and is_part_of_normal_pair(
            prev_interval_dur=intervals[idx - 1] if idx - 1 >= 0 else None,
            suspicious_interval_dur=intervals[idx],
            next_interval_dur=intervals[idx + 1] if idx + 1 < len(intervals) else None,
        ):
            # if the short interval actually sums to an integer number of intervals with one of its neighbours, it's not anomalous
            continue
        short_intervals.append(idx)
    short_intervals = np.sort(short_intervals)
    return short_intervals


def asym_round(
    x: npt.ArrayLike, threshold: float = 0.5, allow_zero: bool = False
) -> npt.NDArray[np.int_]:
    """
    For figuring out how many frames *should* have been displayed given a noisy interval.

    Like a regular round, but rounding up from >=`threshold` instead of the usual 0.5.
    Anything that would become zero is pushed to 1 if `allow_zero` is False (default, since 0 frames
    isn't possible).

    >>> asym_round([0.1, 1.19, 1.5, 2.21], threshold=0.2, allow_zero=False)
    array([1, 1, 2, 3])

    """
    x = np.asarray(x)
    if allow_zero:
        lower_values = np.floor(x)
    else:
        # take greater of floor(x) and 1, to avoid rounding to zero
        lower_values = np.maximum(np.floor(x), np.ones_like(x))
    # avoid using x - np.floor(x) which will incur floating point errors:
    return np.where(x < np.floor(x) + threshold, lower_values, np.ceil(x)).astype(int)

