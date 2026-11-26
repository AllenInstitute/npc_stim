"""
Frame display times: best estimate of when each stim frame appeared on the monitor,
from sync vsync + photodiode edges and the stim file that produced the frames.

Moved from npc_sync (<= 0.2.9: `SyncDataset.frame_display_time_blocks` and the module-level
estimator functions). npc_sync is now only responsible for line access, fault detection and
block splitting; everything that interprets the stimulus (diode box, monitor latency, stim
files) lives here.

The estimator (`get_frame_display_times` and helpers) was moved unchanged from npc_sync 0.2.9;
branch display-times-p2-fixes changes it (see the commit messages and
npc_sessions/outputs/display-times-issues.md).
Orchestration changes:

- stim files are an explicit input (no globbing next to the sync file)
- diode-box parameters are read from the stim file being processed; a missing
  `diodeBoxFrameInterval` key is only accepted for files that pre-date the key
- the constant-lag fallback and re-raising errors are explicit parameters rather than env
  vars, and the source of each block's times is returned with the times
- a block that gets the constant-lag fallback also gets a check of its vsyncs against the stim
  file's `frameIntervals` (`check_frame_indices`, returned as
  `BlockDisplayTimes.frame_index_check`): constant lag is only as good as the vsyncs

Photodiode phantom flicker (extra edges while a frame is held, diodeBoxFrameInterval 1) and
comparator glitches are removed by the sync file's `npc_sync.SyncPatch`, which also records per
block whether every diode edge was explained (`SyncPatch.diode_blocks`). For a block the patch
'cleaned', the estimator consumes the line as patched and does not remove edges itself (no
anomalous-interval removal); a block the patch 'rejected' gets no photodiode display times
(only the constant-lag fallback, if requested). Other blocks
(no patch, box interval > 1, not checked) keep the estimator's own handling.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias

import h5py
import npc_io
import npc_sync
import numpy as np
import numpy.typing as npt
import scipy.signal
import upath

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

LONG_FRAME_THRESHOLD = 1.5
"""A frame is 'long' (at least one dropped refresh) if the interval to the next frame is more
than this many refresh periods. See `check_frame_indices`."""

FRAME_INDEX_CHECK_WINDOW = 3
"""Frames either side of a long frame over which the vsync and stim-file timelines are compared
(medians), in `check_frame_indices`. 3 is the smallest window whose median ignores one frame:
it gives the +/-1 index tolerance."""

FRAME_INDEX_CHECK_MAX_STEP = 0.25
"""Max difference, in refresh periods, between the time that passes across a long frame
according to the vsyncs and according to the stim file, in `check_frame_indices`."""

FRAME_INDEX_CHECK_MAX_CLOCK_DRIFT = 20e-6
"""Max difference between a block's duration on sync and in the stim file, as a fraction of
the duration (in addition to `FRAME_INDEX_CHECK_MAX_STEP`). See `check_frame_indices`."""

HELD_FRAME_MAX_PULSE = 1.1
"""Longest frame-signal pulse (vsync rising edge to falling edge), in refresh periods, for which
a frame's vsync falling edge is used to place it after a held frame, in
`get_frame_display_times` (diodeBoxFrameInterval 1): see `get_held_off_frame_refreshes`.

The frame signal goes high before the frame is drawn and flipped, and low when the flip
returns, at the vertical blank where the frame was swapped in. If the flip returns a refresh
late (the frame is then on screen a refresh before its falling edge + latency), the pulse is a
refresh longer than the wait for the vertical blank.

From the ON frames that follow a held frame (their display time is measured: the diode's
rising edge) in 1,308 box-1 blocks of 247 sessions, by pulse length: under 1.1 refreshes, 4 of
181,894 are on screen 0.75 - 1.5 refreshes before falling edge + latency (0.002%); 1.1 - 1.3:
20 of 14,350 (0.14%); 1.3 - 1.5: 22 of 267 (8%); 1.5 or more: 403 of 671 (60%). Pulses of
1.05 - 1.1 refreshes are the commonest among held frames (the frame signal went high about
1.4 ms before a vertical blank, too late for the frame to be swapped in at it). Of the long
ON-ON intervals where the vsync falling edge and the diode's OFF edge disagree, those with a
pulse under 1.1 are frames with an early OFF edge; with longer pulses they are mostly late
returns (hub/experiments/npc-stim-held-frames)."""

HELD_FRAME_GRID_TOLERANCE = 0.2
"""Max distance, in refresh periods, between a frame's vsync falling edge + latency and a whole
number of refreshes after the preceding diode ON flip, for the falling edge to be used to place
the frame after a held frame (`get_held_off_frame_refreshes`). A falling edge further off the
refresh grid is late by part of a refresh (flip returned late). On-time falling edges are
within 0.1 refresh (0.1 - 99.9th percentile)."""

DisplayTimeSource: TypeAlias = Literal["photodiode", "constant_lag", "unavailable"]
Fallback: TypeAlias = Literal["constant_lag"] | None


@dataclasses.dataclass(frozen=True)
class FrameIndexCheck:
    """Result of comparing a block's long (dropped) frames on sync (vsyncs) with those in the
    stim file (`frameIntervals`): see `check_frame_indices` for the rule."""

    passed: bool | None
    """True: vsyncs and stim file agree. False: they don't. None: the check couldn't be run
    (see `reason`)."""
    reason: str | None = None
    """Why the check failed or couldn't be run (None if it passed)."""
    n_long_sync: int | None = None
    """Long frames in the vsyncs."""
    n_long_stim: int | None = None
    """Long frames in the stim file."""
    n_same_index: int | None = None
    """Long frames at the same index in both."""
    n_adjacent_index: int | None = None
    """Long frames in the vsyncs paired with a long frame at index +/-1 in the stim file."""
    n_unpaired: int | None = None
    """Long frames in one with no long frame within +/-1 index in the other."""
    n_unmatched: int | None = None
    """Long frames (in either) across which the vsyncs and the stim file disagree on the time
    that passed: any fails the check. Paired long frames are counted once."""
    unmatched_frames: tuple[int, ...] = ()
    """Indices of up to the first 20 unmatched long frames."""
    max_step: float | None = None
    """Largest disagreement across a long frame (seconds; signed, sync - stim file)."""
    duration_diff: float | None = None
    """Block duration (first to last frame) on sync minus in the stim file (seconds)."""
    duration_tolerance: float | None = None
    """Max `abs(duration_diff)` allowed (seconds)."""
    refresh_period: float | None = None
    """Median stim-file frame interval (seconds)."""

    @property
    def summary(self) -> str:
        """One line, for logs and notes."""
        if self.n_long_sync is None:
            return f"frame index check not run: {self.reason}"
        assert self.duration_diff is not None and self.max_step is not None
        return (
            f"frame index check {'passed' if self.passed else 'failed'}: "
            f"{self.n_long_sync} long frames on sync, {self.n_long_stim} in stim file "
            f"({self.n_same_index} at the same index, {self.n_adjacent_index} at +/-1, "
            f"{self.n_unpaired} unpaired, {self.n_unmatched} unmatched), max step "
            f"{self.max_step * 1e3:.2f} ms, duration difference {self.duration_diff * 1e3:.2f} ms"
            + ("" if self.reason is None else f": {self.reason}")
        )


def check_frame_indices(
    vsync_times: npt.ArrayLike, frame_intervals: npt.ArrayLike
) -> FrameIndexCheck:
    """Do the long (dropped) frames in a block's vsyncs match those in its stim file?

    The acceptance check for a block that uses vsync + `CONSTANT_MONITOR_LAG`: without the
    photodiode, display times are only as good as the vsyncs, and the stim file's
    `frameIntervals` (PsychoPy's own flip-to-flip times) are the one independent record of them.
    It can't see frames that were displayed late after the flip returned: neither line records
    them.

    - `vsync_times`: vsync falling edges for the block (seconds on sync), one per frame.
    - `frame_intervals`: `frameIntervals` from the stim file (seconds), one fewer.

    With T the refresh period (median of `frame_intervals`), interval k the time from frame k
    to frame k + 1, and c[j] = (vsync j - vsync 0) - sum(frame_intervals[:j]), the time of
    frame j on sync minus its time in the stim file:

    1. Frame k is long if interval k > `LONG_FRAME_THRESHOLD` (1.5) * T, in the vsyncs or in
       the stim file. Intervals aren't whole numbers of refreshes in either (a swap can come
       mid-refresh, followed by a short interval), so an interval near 1.5 T can be long in
       one and not the other.
    2. A long frame k matches if both agree on the time that passes across it:
       abs(median(c[k+1 : k+1+W]) - median(c[k+1-W : k+1])) <= `FRAME_INDEX_CHECK_MAX_STEP`
       (0.25) * T, with W = `FRAME_INDEX_CHECK_WINDOW` (3) frames. A median of 3 ignores one
       frame, so this allows:
         - the same long frame at index k in one and k +/- 1 in the other: jitter in the vsync
           edge between them splits the interval differently (e.g. 31.7 + 18.3 ms on sync,
           16.7 + 33.4 ms in the stim file);
         - one late vsync falling edge (a long then a short interval on sync only, e.g.
           26.7 + 6.4 ms vs 17.8 + 15.5 ms), next to the long frame or itself over the
           threshold: no refresh was dropped.
       It doesn't allow a long frame 2 or more indices away in the other, a long frame
       missing from the other, or one that is a different number of refreshes long: each
       shifts c by about a refresh (T) or more for 2 or more frames.
    3. The block's duration is the same: abs(c[-1]) <= `FRAME_INDEX_CHECK_MAX_STEP` * T +
       `FRAME_INDEX_CHECK_MAX_CLOCK_DRIFT` (20 ppm) * duration. The sync and stim clocks run
       at slightly different rates, so this can't resolve single frames in a long block (one
       refresh in an hour is 4.6 ppm): it's a check for gross differences; rule 2 is the
       frame-level one.

    The check passes if no long frame is unmatched and the duration is the same.

    Thresholds, from the 175 photodiode blocks (all stims) of 28 sessions from 2023-08 to
    2024-06, with 34,567 long frames (hub/experiments/constant-lag-acceptance):

    - step across a long frame (rule 2): at most 0.016 T (0.26 ms), against about 1 T for a
      real mismatch. Any limit from 0.03 to 0.9 T passes every block; 0.25 T is 16x the
      largest seen. A refresh added to, removed from or moved by 2 frames at a long frame in
      the stim file was detected every time (474 long frames).
    - with W = 2 (no index tolerance) 34 long frames have a step over 0.25 T (up to 6 in a
      block, in 21 blocks): W = 3 is needed. W = 5 gives the same results as 3.
    - clock rate difference (rule 3): -5.5 to +1.3 ppm; 20 ppm is over 3x the largest seen.
    - 1.5 T is halfway between one and two refreshes. 4% of intervals over 1.25 T are
      between 1.25 and 1.75 T (the rest are close to a whole number of refreshes), and rule 2
      covers the ones that land either side of the threshold: any threshold from 1.25 to
      1.75 T gives the same results.

    Counts of long frames at the same index, at +/-1 (paired one-to-one in order) and
    unpaired are returned for information; only rule 2 and 3 decide `passed`.

    >>> v = np.arange(100) / 60
    >>> v[50:] += 1 / 60                        # one dropped refresh after frame 49
    >>> check_frame_indices(v, np.diff(v)).passed
    True
    >>> f = np.diff(v)
    >>> f[[49, 50]] = f[[50, 49]]               # long frame at 50 in the stim file
    >>> r = check_frame_indices(v, f)
    >>> r.passed, r.n_same_index, r.n_adjacent_index
    (True, 0, 1)
    >>> f = np.diff(v)
    >>> f[[49, 51]] = f[[51, 49]]               # long frame at 51 in the stim file
    >>> r = check_frame_indices(v, f)
    >>> r.passed, r.n_unmatched, r.unmatched_frames
    (False, 2, (49, 51))
    >>> r = check_frame_indices(v, np.full(99, 1 / 60))  # not in the stim file at all
    >>> r.passed, r.n_unmatched
    (False, 1)
    >>> check_frame_indices(v, np.full(98, 1 / 60)).reason
    '100 vsyncs, but 98 frame intervals in the stim file (expected 99)'
    """
    vsyncs = np.asarray(vsync_times, dtype=float)
    stim_intervals = np.asarray(frame_intervals, dtype=float)
    if len(vsyncs) < 2 or len(stim_intervals) != len(vsyncs) - 1:
        return FrameIndexCheck(
            passed=None if len(vsyncs) < 2 else False,
            reason=f"{len(vsyncs)} vsyncs, but {len(stim_intervals)} frame intervals in the stim "
            f"file (expected {max(len(vsyncs) - 1, 0)})",
        )
    period = float(np.median(stim_intervals))
    sync_intervals = np.diff(vsyncs)
    # time of each frame on sync minus its time in the stim file
    c = (vsyncs - vsyncs[0]) - np.concatenate(([0.0], np.cumsum(stim_intervals)))

    long_sync = np.flatnonzero(sync_intervals > LONG_FRAME_THRESHOLD * period)
    long_stim = np.flatnonzero(stim_intervals > LONG_FRAME_THRESHOLD * period)

    # pair long frames for reporting: same index, then +/-1 (one-to-one, in order)
    same = np.intersect1d(long_sync, long_stim)
    sync_only = np.setdiff1d(long_sync, same)
    stim_only = np.setdiff1d(long_stim, same)
    adjacent: list[tuple[int, int]] = []
    i = j = 0
    while i < len(sync_only) and j < len(stim_only):
        d = int(sync_only[i]) - int(stim_only[j])
        if abs(d) <= 1:
            adjacent.append((int(sync_only[i]), int(stim_only[j])))
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    n_unpaired = len(sync_only) + len(stim_only) - 2 * len(adjacent)

    # rule 2: step in c across each long frame
    w = FRAME_INDEX_CHECK_WINDOW
    long_frames = np.union1d(long_sync, long_stim)
    steps = np.array(
        [
            np.median(c[k + 1 : k + 1 + w]) - np.median(c[max(k + 1 - w, 0) : k + 1])
            for k in long_frames
        ]
    )
    bad = set(long_frames[np.abs(steps) > FRAME_INDEX_CHECK_MAX_STEP * period].tolist())
    # the two halves of a pair are one long frame
    n_unmatched = len(bad) - sum(a in bad and b in bad for a, b in adjacent)
    max_step = float(steps[np.argmax(np.abs(steps))]) if len(steps) else 0.0

    # rule 3: duration
    duration_diff = float(c[-1])
    duration_tolerance = float(
        FRAME_INDEX_CHECK_MAX_STEP * period
        + FRAME_INDEX_CHECK_MAX_CLOCK_DRIFT * (vsyncs[-1] - vsyncs[0])
    )

    reasons = []
    if n_unmatched:
        frames = sorted(bad)
        reasons.append(
            f"{n_unmatched} long frame(s) don't match between vsyncs and stim file "
            f"(max step {max_step * 1e3:.1f} ms): frame(s) "
            + ", ".join(str(k) for k in frames[:20])
            + (", ..." if len(frames) > 20 else "")
        )
    if abs(duration_diff) > duration_tolerance:
        reasons.append(
            f"block duration differs by {duration_diff * 1e3:.1f} ms between vsyncs and stim "
            f"file (tolerance {duration_tolerance * 1e3:.1f} ms)"
        )
    return FrameIndexCheck(
        passed=not reasons,
        reason="; ".join(reasons) or None,
        n_long_sync=len(long_sync),
        n_long_stim=len(long_stim),
        n_same_index=len(same),
        n_adjacent_index=len(adjacent),
        n_unpaired=n_unpaired,
        n_unmatched=n_unmatched,
        unmatched_frames=tuple(sorted(bad)[:20]),
        max_step=max_step,
        duration_diff=duration_diff,
        duration_tolerance=duration_tolerance,
        refresh_period=period,
    )


def get_frame_index_check(
    vsync_times: npt.ArrayLike,
    stim_path_or_data: npc_io.PathLike | h5py.File | None,
) -> FrameIndexCheck:
    """`check_frame_indices` for a block of vsyncs and its stim file (any block, whatever the
    source of its display times): e.g. `get_frame_index_check(block.vsync_times,
    block.stim_path)` for a `BlockDisplayTimes`. `passed` is None if the stim file is missing
    or its `frameIntervals` can't be read."""
    if stim_path_or_data is None:
        return FrameIndexCheck(passed=None, reason="no stim file matched to block")
    try:
        stim_data = npc_stim.stim.get_h5_stim_data(stim_path_or_data)
        try:
            frame_intervals = stim_data["frameIntervals"][:]
        finally:
            if stim_data is not stim_path_or_data:
                stim_data.close()
    except (OSError, KeyError) as exc:
        return FrameIndexCheck(
            passed=None, reason=f"frameIntervals can't be read from stim file: {exc!r}"
        )
    return check_frame_indices(vsync_times, frame_intervals)


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
    frame_source: npt.NDArray[np.int8] | None = None
    """Per frame: 0 (measured), 1 (interpolated: the frame's vsync, or the diode edge its
    display time comes from, was reconstructed by the sync file's `SyncPatch` inside a
    replayed interval) or 2 (constant lag in a photodiode block: the frame's diode flip was
    misassigned, see `MAX_EARLY_FRAMES`, so its display time is vsync +
    `CONSTANT_MONITOR_LAG`). None if every frame is 0 (no patch and no constant-lag frames)."""
    notes: str | None = None
    """Diagnostics for a photodiode block: frames given constant lag (`frame_source` 2), or the
    diode-flip window starting at the first vsync's rising edge."""
    frame_index_check: FrameIndexCheck | None = None
    """For `source='constant_lag'`: do the long (dropped) frames in the vsyncs match the stim
    file's (`check_frame_indices`)? `passed` is None if it couldn't be run (no stim file).
    None for other sources (not run: use `get_frame_index_check`)."""

    @property
    def is_photodiode(self) -> bool:
        return self.source == "photodiode"

    @property
    def n_interpolated(self) -> int:
        return 0 if self.frame_source is None else int(np.sum(self.frame_source == 1))

    @property
    def n_constant_lag_frames(self) -> int:
        """Frames whose display time is vsync + `CONSTANT_MONITOR_LAG`: all of them for
        `source='constant_lag'`, else those with `frame_source` 2."""
        if self.source == "constant_lag":
            return len(self.vsync_times)
        return 0 if self.frame_source is None else int(np.sum(self.frame_source == 2))


def _is_patched_for_display_times(sync_data: npc_sync.SyncDataset) -> bool:
    """The sync file's replayed intervals are covered by a `SyncPatch` for both the vsync and
    photodiode lines (spans may still be 'unknown': those blocks are rejected individually).
    """
    patch = getattr(sync_data, "patch", None)
    return patch is not None and all(
        patch.covers(line) for line in ("vsync_stim", "stim_photodiode")
    )


def assert_sync_frame_timing_is_valid(sync: npc_sync.SyncPathOrDataset) -> None:
    """Raise `AssertionError` if the sync file contains replayed data (see
    `SyncDataset.invalid_intervals`) that isn't covered by a `SyncPatch` for the vsync and
    photodiode lines, or a timebase discontinuity (see `SyncDataset.timebase_discontinuities`),
    since vsync and diode edges can't be trusted.
    """
    sync_data = npc_sync.get_sync_data(sync)
    if sync_data.invalid_intervals and not _is_patched_for_display_times(sync_data):
        raise AssertionError(
            f"{sync_data} contains replayed sync data (all lines invalid) at "
            + ", ".join(f"{a:.3f}-{z:.3f} s" for a, z in sync_data.invalid_intervals)
            + ": frame display times can't be computed reliably without a SyncPatch"
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
        raise AssertionError(
            "Frame rate estimated from diode flips is abnormal."
        ) from exc


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
    # edges alternate, so each span of a SyncPatch with the line 'unknown' (edges removed, none
    # replacing them) can unbalance the counts by one (DRpilot_644867_20230221: 7 spans); the
    # blocks overlapping them are rejected in `get_display_times`
    n_unknown = (
        len(sync_data.get_unknown_intervals("stim_photodiode"))
        if getattr(sync_data, "patch", None) is not None
        else 0
    )
    assert abs(len(diode_rising_edges) - len(diode_falling_edges)) < 2 + n_unknown
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


def _diode_block_record(
    sync_data: npc_sync.SyncDataset, vsyncs: npt.NDArray
) -> object | None:
    """The sync patch's photodiode record for the block of `vsyncs`
    (`npc_sync.SyncPatch.diode_blocks`), or None (no patch, or an npc_sync without records).
    """
    patch = getattr(sync_data, "patch", None)
    if patch is None or not hasattr(patch, "diode_block"):
        return None
    return patch.diode_block(float(vsyncs[0]), float(vsyncs[-1]))


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


STIMS_WITHOUT_DISPLAY_TIMED_VISUAL_STIMS: tuple[str, ...] = (
    "Spontaneous",
    "SpontaneousRewards",
    "OptoTagging",
)
"""Stim scripts whose blocks aren't rejected for photodiode / display-time problems alone:
they get vsync + `CONSTANT_MONITOR_LAG` (`source='constant_lag'`) whatever `fallback` is.
Spontaneous and SpontaneousRewards show no visual stim (their rewards and licks are timed by
sync lines, or by frame indices on the vsyncs), and OptoTagging's stim onsets and offsets are
on the laser lines of sync. Their frame times only need the vsyncs to be right: a block
overlapping a span where the SyncPatch couldn't reconstruct the vsync line still gets no
display times (its frames can't be indexed), but one where only the photodiode is unknown
does."""


def has_display_timed_visual_stims(stim_path: npc_io.PathLike) -> bool:
    """False for stims in `STIMS_WITHOUT_DISPLAY_TIMED_VISUAL_STIMS` (by script name: the
    file name up to the first underscore).

    >>> has_display_timed_visual_stims("DynamicRouting1_733891_20240919_130255.hdf5")
    True
    >>> has_display_timed_visual_stims("SpontaneousRewards_733891_20240919_125243.hdf5")
    False
    """
    name = npc_io.from_pathlike(stim_path).stem.split("_")[0]
    return name not in STIMS_WITHOUT_DISPLAY_TIMED_VISUAL_STIMS


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
    - `fallback='constant_lag'`: blocks where the photodiode can't be used (including blocks
      whose photodiode edges the `SyncPatch` rejected) get vsync + `CONSTANT_MONITOR_LAG`, with
      `source='constant_lag'`. Default: no display times (`source='unavailable'`). Blocks
      overlapping a span the patch couldn't reconstruct never get a fallback. Blocks that get
      the fallback also get `BlockDisplayTimes.frame_index_check`: whether the long (dropped)
      frames in the vsyncs match those in the stim file (`check_frame_indices`). The times
      are returned either way: it's up to the caller not to use a block that fails.
    - a photodiode block with up to `MAX_EARLY_FRAMES` frames whose diode flip comes too soon
      after their vsync rising edge keeps its photodiode times, with constant lag for those
      frames (`frame_source` 2, listed in `BlockDisplayTimes.notes`)
    - diode flips are kept from the first vsync's falling edge; if the estimate fails, it's
      retried from the first vsync's rising edge (recorded in `notes`)
    - `raise_on_error`: re-raise errors from the estimator instead of recording the block as
      not using the photodiode.
    - stims without visual stims timed by display times (`STIMS_WITHOUT_DISPLAY_TIMED_VISUAL_STIMS`:
      Spontaneous, SpontaneousRewards, OptoTagging) always get the constant-lag fallback,
      also with `fallback=None`, and also where only the photodiode line is unknown in a
      replayed span: they aren't rejected for display-time problems alone.

    Raises `AssertionError` if the sync file contains replayed data not covered by a
    `SyncPatch`, or a timebase discontinuity (see `assert_sync_frame_timing_is_valid`). With
    a patch: blocks overlapping a span where the vsync or photodiode line is 'unknown' get no
    display times; elsewhere frames whose vsync or diode edge was reconstructed are marked in
    `BlockDisplayTimes.frame_source`.
    """
    sync_data = npc_sync.get_sync_data(sync)
    assert_sync_frame_timing_is_valid(sync_data)
    patched = bool(sync_data.invalid_intervals) and _is_patched_for_display_times(
        sync_data
    )
    unknown_vsync: list = []
    if patched:
        unknown_vsync = list(sync_data.get_unknown_intervals("vsync_stim"))
        unknown = unknown_vsync + list(
            sync_data.get_unknown_intervals("stim_photodiode")
        )
        all_vsyncs = sync_data.get_falling_edges("vsync_stim", units="seconds")
        vsync_src = sync_data.get_edge_sources("vsync_stim", "falling")
        diode_src = np.concatenate(
            [
                sync_data.get_edge_sources("stim_photodiode", "rising"),
                sync_data.get_edge_sources("stim_photodiode", "falling"),
            ]
        )
        # spans where diode edges were reconstructed; display times up to one diode interval
        # after a span are placed relative to a reconstructed edge
        reconstructed_diode_spans = [
            (sp.span[0], sp.span[1] + 0.1)
            for sp in sync_data.patch.lines.get("stim_photodiode", [])
            if sp.status == "reconstructed" and (diode_src == 1).any()
        ]

    def frame_source(
        vsyncs: npt.NDArray[np.float64], display: npt.NDArray[np.float64] | None
    ) -> npt.NDArray[np.int8] | None:
        if not patched:
            return None
        idx = np.searchsorted(all_vsyncs, vsyncs)
        src = vsync_src[np.clip(idx, 0, len(vsync_src) - 1)].astype(np.int8)
        if display is not None and reconstructed_diode_spans:
            src[npc_sync.in_intervals(display, reconstructed_diode_spans)] = 1
        return src

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
        needs_display_times = stim_path is None or has_display_timed_visual_stims(
            stim_path
        )
        block_fallback: Fallback = fallback if needs_display_times else "constant_lag"

        def overlaps(spans: Sequence[tuple[float, float]]) -> bool:
            return patched and any(
                a <= vsyncs[-1] + MAX_VSYNC_DIODE_FLIP_SEPARATION_SEC and z >= vsyncs[0]
                for a, z in spans
            )

        # a block overlapping a span where the SyncPatch couldn't reconstruct vsync/photodiode
        # gets no display times, even with a fallback: its vsyncs can't be trusted either.
        # Stims without display-timed visual stims only need the vsyncs.
        overlaps_unknown = overlaps(unknown if patched else []) and (
            needs_display_times or overlaps(unknown_vsync)
        )

        def unavailable(reason: str) -> BlockDisplayTimes:
            return BlockDisplayTimes(
                block_index=block_idx,
                stim_path=stim_path,
                vsync_times=vsyncs,
                display_times=None,
                source="unavailable",
                reason=reason,
            )

        def not_using_diode(
            reason: str, vsyncs_per_diode_flip: int | None
        ) -> BlockDisplayTimes:
            check = None
            if block_fallback == "constant_lag":
                check = get_frame_index_check(vsyncs, stim_path)
                logger.log(
                    logging.INFO if check.passed else logging.WARNING,
                    f"{sync_data} block {block_idx}: constant lag: {check.summary}",
                )
            return BlockDisplayTimes(
                block_index=block_idx,
                stim_path=stim_path,
                vsync_times=vsyncs,
                display_times=(
                    vsyncs + CONSTANT_MONITOR_LAG
                    if block_fallback == "constant_lag"
                    else None
                ),
                source=(
                    "constant_lag"
                    if block_fallback == "constant_lag"
                    else "unavailable"
                ),
                diode_box_frame_interval=vsyncs_per_diode_flip,
                reason=reason,
                frame_source=(
                    frame_source(vsyncs, None)
                    if block_fallback == "constant_lag"
                    else None
                ),
                frame_index_check=check,
            )

        diode_record = _diode_block_record(sync_data, vsyncs)
        if diode_record is not None and diode_record.status == "rejected":
            msg = f"SyncPatch rejected the block's photodiode edges: {diode_record.reason}"
            logger.warning(f"{sync_data} block {block_idx}: {msg}")
            # the vsyncs are fine (unless the block also overlaps an unknown span): the
            # fallback applies, as for any other block whose photodiode can't be used
            results.append(
                unavailable(msg) if overlaps_unknown else not_using_diode(msg, None)
            )
            continue
        diode_cleaned = diode_record is not None and diode_record.status == "cleaned"

        if overlaps_unknown:
            msg = "block overlaps a replayed sync interval that the SyncPatch couldn't reconstruct for vsync/photodiode"
            logger.warning(f"{sync_data} block {block_idx}: {msg}")
            results.append(unavailable(msg))
            continue

        if stim_path is None:
            msg = "no stim file matched to block: number of vsyncs and diode-box interval can't be checked"
            logger.warning(f"{sync_data} block {block_idx}: {msg}")
            results.append(not_using_diode(msg, None))
            continue

        stim_data = npc_stim.stim.get_h5_stim_data(stim_path)
        try:
            vsyncs_per_diode_flip = get_diode_box_frame_interval(stim_data)
            expected_vsyncs = npc_stim.stim.get_total_stim_frames(stim_data)
        finally:
            stim_data.close()

        stim_id = f"{stim_path.parent.name}_{block_idx}_{stim_path.stem}"
        vsync_rising_edges = get_vsync_rising_edges(sync_data, vsyncs)

        # Diode flips are kept from the start of the first frame + an empirically determined
        # min latency between vsync and screen update. The first frame starts at its vsync
        # falling edge (as in npc_stim <= 0.1.26); if the estimate fails, it's retried from the
        # first frame's vsync rising edge: a late first falling edge can be followed by the
        # first flip < MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC later (666986_2023-08-16 RFMapping:
        # falling edge 12 ms late, flip 13 ms after it). The falling edge comes first because
        # the rising-edge window admits diode flicker before the first frame
        # (670180_2023-07-27 task: edges 16 and 18 ms after the rising edge, first flip at 41 ms)
        first_frame_rise = get_first_frame_start(sync_data, vsyncs)
        can_retry = first_frame_rise < vsyncs[0]
        first = _estimate_block_display_times(
            sync_data_name=str(sync_data),
            block_idx=block_idx,
            stim_id=stim_id,
            vsyncs=vsyncs,
            vsync_rising_edges=vsync_rising_edges,
            rising=rising,
            falling=falling,
            first_frame_start=float(vsyncs[0]),
            vsyncs_per_diode_flip=vsyncs_per_diode_flip,
            expected_vsyncs=expected_vsyncs,
            diode_cleaned=diode_cleaned,
            raise_on_error=raise_on_error and not can_retry,
        )
        estimate, notes = first, None
        if first.reason is not None and first.retryable and can_retry:
            second = _estimate_block_display_times(
                sync_data_name=str(sync_data),
                block_idx=block_idx,
                stim_id=stim_id,
                vsyncs=vsyncs,
                vsync_rising_edges=vsync_rising_edges,
                rising=rising,
                falling=falling,
                first_frame_start=first_frame_rise,
                vsyncs_per_diode_flip=vsyncs_per_diode_flip,
                expected_vsyncs=expected_vsyncs,
                diode_cleaned=diode_cleaned,
                raise_on_error=raise_on_error,
            )
            if second.reason is None:
                estimate = second
                notes = (
                    "diode flips kept from the first vsync's rising edge (not its falling "
                    f"edge, {(vsyncs[0] - first_frame_rise) * 1e3:.1f} ms later): "
                    f"from the falling edge, {first.reason}"
                )
                logger.warning(f"{stim_id}: Block {block_idx}: {notes}")
            else:
                estimate = dataclasses.replace(
                    first,
                    reason=f"{first.reason} (also with diode flips kept from the first vsync's "
                    f"rising edge: {second.reason})",
                )

        if estimate.reason is not None or estimate.times is None:
            logger.error(f"{stim_id}: Block {block_idx}: {estimate.reason}")
            results.append(
                not_using_diode(str(estimate.reason), estimate.diode_box_frame_interval)
            )
            continue

        times = estimate.times
        src = frame_source(vsyncs, times)
        if len(estimate.early_frames):
            early_note = (
                f"{len(estimate.early_frames)} frame(s) with a diode flip less than "
                f"{MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC * 1e3:.0f} ms after their vsync rising edge "
                f"(flip misassigned) given vsync + constant lag (frame_source 2): frame(s) "
                + ", ".join(str(int(i)) for i in estimate.early_frames)
            )
            logger.warning(f"{stim_id}: Block {block_idx}: {early_note}")
            notes = early_note if notes is None else f"{notes}; {early_note}"
            if src is None:
                src = np.zeros(len(vsyncs), dtype=np.int8)
            src[estimate.early_frames] = 2

        if len(estimate.invented_frames):
            invented_note = (
                f"{len(estimate.invented_frames)} frame(s) without a measured or supported diode "
                "flip (at the end of the block, or an unsupported first OFF frame) are given "
                "vsync + the block's median latency (frame_source 2): "
                "frame(s) " + ", ".join(str(int(i)) for i in estimate.invented_frames)
            )
            logger.warning(f"{stim_id}: Block {block_idx}: {invented_note}")
            notes = invented_note if notes is None else f"{notes}; {invented_note}"
            if src is None:
                src = np.zeros(len(vsyncs), dtype=np.int8)
            src[estimate.invented_frames] = 2

        results.append(
            BlockDisplayTimes(
                block_index=block_idx,
                stim_path=stim_path,
                vsync_times=vsyncs,
                display_times=times + MONITOR_CENTER_REFRESH_TIME,
                source="photodiode",
                diode_box_frame_interval=vsyncs_per_diode_flip,
                frame_source=src,
                notes=notes,
            )
        )
        logger.info(f"Added frame display times for {stim_path}")

    assert len(results) == len(
        vsync_times_in_blocks
    ), f"{sync_data}: {len(results)} blocks of diode flips matched to {len(vsync_times_in_blocks)} blocks of vsyncs"
    return tuple(results)


MAX_EARLY_FRAMES = 3
"""A photodiode block with up to this many frames whose diode flip is earlier than their vsync
rising edge + `MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC` keeps its photodiode times: those frames get
vsync + `CONSTANT_MONITOR_LAG` (`frame_source` 2). Seen after a dropped frame with a short vsync
pulse (714748_2024-06-24, 733891_2024-09-17) and on the frame invented at the end of a block
(742903_2024-10-24, npc_stim#9). With more, the flips are taken to be misassigned."""


@dataclasses.dataclass(frozen=True)
class _BlockEstimate:
    """One attempt at a block's photodiode display times (without
    `MONITOR_CENTER_REFRESH_TIME`)."""

    times: npt.NDArray[np.float64] | None
    reason: str | None = None
    """Why the photodiode can't be used (None if `times` are usable)."""
    diode_box_frame_interval: int | None = None
    retryable: bool = True
    """The failure may depend on where the diode-flip window starts."""
    early_frames: npt.NDArray[np.intp] = dataclasses.field(
        default_factory=lambda: np.array([], dtype=np.intp)
    )
    invented_frames: npt.NDArray[np.intp] = dataclasses.field(
        default_factory=lambda: np.array([], dtype=np.intp)
    )
    """Frames at the end of a box-1 block without a diode flip, given vsync + the block's
    median latency (see `reconcile_box1_frame_count`)."""
    """Frames given vsync + constant lag (see `MAX_EARLY_FRAMES`)."""


def _estimate_block_display_times(
    *,
    sync_data_name: str,
    block_idx: int,
    stim_id: str,
    vsyncs: npt.NDArray[np.float64],
    vsync_rising_edges: npt.NDArray[np.float64],
    rising: npt.NDArray[np.float64],
    falling: npt.NDArray[np.float64],
    first_frame_start: float,
    vsyncs_per_diode_flip: int,
    expected_vsyncs: int,
    diode_cleaned: bool,
    raise_on_error: bool,
) -> _BlockEstimate:
    """Photodiode display times for one block (see `get_display_times`), with diode flips kept
    from `first_frame_start` + `MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC`."""

    def fail(
        reason: str,
        interval: int | None = vsyncs_per_diode_flip,
        retryable: bool = True,
    ) -> _BlockEstimate:
        return _BlockEstimate(
            None, reason, diode_box_frame_interval=interval, retryable=retryable
        )

    # keep flips after the start of the first frame + an empirically determined min latency
    # between vsync and screen update
    falling = falling[
        falling > (first_frame_start + MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC)
    ]
    rising = rising[rising > (first_frame_start + MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC)]

    # keep flips only up to a certain time after the last vsync
    falling = falling[falling < (vsyncs[-1] + MAX_VSYNC_DIODE_FLIP_SEPARATION_SEC)]
    rising = rising[rising < (vsyncs[-1] + MAX_VSYNC_DIODE_FLIP_SEPARATION_SEC)]

    if not len(rising) or not len(falling):
        # previously an IndexError that failed the whole session (DRpilot_796848_20250715)
        msg = f"no diode {'rising' if not len(rising) else 'falling'} edges between the block's first vsync and {MAX_VSYNC_DIODE_FLIP_SEPARATION_SEC * 1e3:.0f} ms after its last"
        logger.warning(f"{sync_data_name} block {block_idx}: {msg}")
        return fail(msg, None)

    is_first_frame_on = rising[0] < falling[0]  # may be updated below

    if vsyncs_per_diode_flip == 1 and expected_vsyncs != len(vsyncs):
        msg = f"{len(vsyncs)} vsyncs, expected {expected_vsyncs} from stim file"
        logger.warning(f"Block {block_idx} has {msg}: Skipping")
        return fail(msg, retryable=False)

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
    # a block whose diode edges the SyncPatch cleaned is used as patched: the patch is the
    # one place where edges are removed (phantom flicker, glitches), so don't remove more
    while not diode_cleaned and any(
        anomalous_interval_indices(flips, vsyncs_per_diode_flip)
    ):

        indices = anomalous_interval_indices(flips, vsyncs_per_diode_flip)
        # just delete one interval then check again:
        logger.warning(f"Removing anomalous short diode interval in block {block_idx}")
        flips = np.delete(flips, slice(indices[0], indices[0] + 2))
        if 0 in indices:
            is_first_frame_on = not is_first_frame_on

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
        return fail(msg)

    rising, falling = split_flips(flips, is_first_frame_on)

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
            vsync_rising_times=vsync_rising_edges,
        )
    except (AssertionError, IndexError) as exc:
        if raise_on_error:
            raise
        logger.exception(
            f"Failed to get correct frame display times for {stim_id} block {block_idx}"
        )
        return fail(f"estimator failed: {exc!r}")

    invented = np.array([], dtype=np.intp)
    if vsyncs_per_diode_flip == 1 and len(times) != len(vsyncs):
        reconciled, invented, problem = reconcile_box1_frame_count(times, vsyncs)
        if problem is not None:
            if raise_on_error:
                raise AssertionError(f"{stim_id}: Block {block_idx}: {problem}")
            return fail(problem)
        times = reconciled

    early = np.array([], dtype=np.intp)
    if len(times) == len(vsyncs):
        problem: str | None
        times, problem = correct_whole_block_frame_offset(times, vsyncs)
        if problem is None:
            # replaces the disabled `display >= vsync` assert: vsync falling edges can be
            # late (a delay between the buffer swap and the falling-edge write), but a frame
            # can't be displayed before its frame signal went high (+ the minimum
            # vsync-to-flip latency)
            early = np.flatnonzero(
                times < vsync_rising_edges + MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC
            )
            if len(early):
                problem = (
                    f"{len(early)} frame(s) have a diode flip less than "
                    f"{MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC * 1e3:.0f} ms after their vsync rising edge "
                    "(frame signal raised before draw + flip): flips are misassigned"
                )
                if len(early) <= MAX_EARLY_FRAMES:
                    # isolated frames (see `MAX_EARLY_FRAMES`): use constant lag for them only
                    times = times.copy()
                    times[early] = vsyncs[early] + AVERAGE_VSYNC_TO_DIODE_FLIP_TIME
                    problem = (
                        None
                        if np.all(np.diff(times) > 0)
                        else f"{problem} (vsync + constant lag for them is out of order)"
                    )
        if problem is None and vsyncs_per_diode_flip == 1 and not is_first_frame_on:
            times, first_replaced = replace_unsupported_first_frame(
                times, vsyncs, vsync_rising_edges
            )
            if first_replaced:
                invented = np.union1d(invented, [0]).astype(np.intp)
        if problem is None:
            problem = get_residual_step_problem(times, vsyncs)
        if problem is not None:
            return fail(problem)

    return _BlockEstimate(
        times,
        diode_box_frame_interval=vsyncs_per_diode_flip,
        early_frames=early,
        invented_frames=invented,
    )


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
    rejects blocks whose vsync falling edges are late (see `get_frame_display_times`), so this check
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
    if (
        n_frames == 1
        and first_flip_latency > AVERAGE_VSYNC_TO_DIODE_FLIP_TIME + 0.5 * FRAME_INTERVAL
    ):
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


def get_vsync_rising_edges(
    sync_data: npc_sync.SyncDataset, vsync_falling_edges: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The rising edge of the frame signal preceding each falling edge (-inf if none)."""
    rising = sync_data.get_rising_edges("vsync_stim", units="seconds")
    idx = np.searchsorted(rising, vsync_falling_edges) - 1
    return np.where(idx >= 0, rising[np.clip(idx, 0, None)], -np.inf)


def get_first_frame_start(
    sync_data: npc_sync.SyncDataset, vsync_falling_edges: npt.NDArray[np.float64]
) -> float:
    """The vsync rising edge of the first frame in a block, or its falling edge if there's no
    rising edge within `MAX_VSYNC_DIODE_FLIP_SEPARATION_SEC` before it (no rising edge on the
    line, or the nearest one belongs to an earlier pulse/block)."""
    first_fall = float(vsync_falling_edges[0])
    first_rise = float(get_vsync_rising_edges(sync_data, vsync_falling_edges[:1])[0])
    if not (
        first_fall - MAX_VSYNC_DIODE_FLIP_SEPARATION_SEC < first_rise <= first_fall
    ):
        return first_fall
    all_falling = sync_data.get_falling_edges("vsync_stim", units="seconds")
    previous_falling = all_falling[all_falling < first_fall]
    if len(previous_falling) and previous_falling[-1] >= first_rise:
        # the rising edge belongs to an earlier pulse
        return first_fall
    return first_rise


def place_frames_in_diode_interval(
    slot_times: npt.ArrayLike,
    lower_bounds: npt.ArrayLike,
    targets: npt.ArrayLike,
) -> list[float]:
    """Assign N frames to screen refreshes ("slots") within one diode-box interval.

    The first frame is the diode flip (slot 0). Each later frame gets the slot nearest its
    target (vsync falling edge + average latency) among slots that are after the previous
    frame's slot, at or after its lower bound (rising edge + min latency) where possible, and
    leave enough slots for the remaining frames.

    >>> place_frames_in_diode_interval([0, 1, 2, 3, 4, 5, 6], [-1, 4.5, 5], [0, 1, 2])
    [0.0, 5.0, 6.0]
    >>> place_frames_in_diode_interval([0, 1, 2, 3], [-1, 0, 0], [0, 1, 3])
    [0.0, 1.0, 3.0]
    """
    slots = np.asarray(slot_times, dtype=float)
    lower = np.asarray(lower_bounds, dtype=float)
    target = np.asarray(targets, dtype=float)
    n_frames, n_slots = len(target), len(slots)
    chosen = [0]
    for j in range(1, n_frames):
        allowed = np.arange(chosen[-1] + 1, n_slots - (n_frames - 1 - j))
        feasible = allowed[slots[allowed] >= lower[j]]
        candidates = feasible if len(feasible) else allowed[-1:]
        chosen.append(int(candidates[np.argmin(np.abs(slots[candidates] - target[j]))]))
    return [float(slots[i]) for i in chosen]


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


MAX_MISSING_END_FLIPS = 3
"""Most frames at the end of a box-1 block that may lack a diode flip (e.g. the sync recording
stopped, or the last sync squares weren't detected): they get vsync + the block's median
latency, marked in `frame_source`. See `reconcile_box1_frame_count`."""

MAX_EXTRA_END_FLIPS = 4
"""Most diode flips after the display of a box-1 block's last frame that are discarded (e.g.
the transition from the last sync square to the grey screen). See
`reconcile_box1_frame_count`."""

END_CHECK_FRAMES = 50
"""Frames at the end of a box-1 block whose latency (display - vsync) is compared with the
block's median latency before frames are added or removed at the end (see
`reconcile_box1_frame_count`)."""

RESIDUAL_STEP_WINDOW = 301
"""Frames in the running median of display - vsync latency used to find stretches of a block
whose flips are assigned to the wrong frames (`get_residual_step_problem`)."""


def reconcile_box1_frame_count(
    display_times: npt.NDArray[np.float64],
    vsyncs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], str | None]:
    """Make a box-1 block's display times (one per diode flip, from
    `get_frame_display_times`) one per vsync, if the extra or missing flips are verifiably at
    the end of the block.

    Flips are assigned to frames in order, so a flip missing (or extra) in the middle of the
    block puts every later frame a refresh late (early). Before npc_stim 0.1.32 up to 3
    missing display times were invented and up to 4 extra ones dropped at the end without
    checking where the discrepancy was (npc_stim#9: in 733891_2024-09-18 the task block's
    frames were 2 refreshes early from frame 213,257 to the end). Now:

    - the latency (display - vsync) of the last `END_CHECK_FRAMES` frames that have a flip
      must be within half a refresh of the block's median latency (otherwise the
      discrepancy isn't at the end)
    - extra flips (at most `MAX_EXTRA_END_FLIPS`) must come later than the last frame's
      display time + half a refresh: they're dropped
    - missing flips (at most `MAX_MISSING_END_FLIPS`) are frames at the end without a flip:
      they get vsync + the block's median latency, and their indices are returned so they
      can be marked

    A discrepancy in the middle of a block that the end checks miss (e.g. one missing and
    one extra flip) is caught by `get_residual_step_problem`.

    Returns (display_times, invented frame indices, problem); problem is None if reconciled.

    >>> v = np.arange(1000) / 59.95
    >>> lag = AVERAGE_VSYNC_TO_DIODE_FLIP_TIME
    >>> t, inv, msg = reconcile_box1_frame_count(np.append(v + lag, v[-1] + lag + 1 / 59.95), v)
    >>> msg is None and len(t) == 1000 and len(inv) == 0
    True
    >>> t, inv, msg = reconcile_box1_frame_count((v + lag)[:-2], v)
    >>> msg is None, inv.tolist(), bool(np.allclose(t - v, lag))
    (True, [998, 999], True)
    >>> # a flip missing at frame 900: frames 900-998 are given the next frame's flip
    >>> late = np.concatenate([(v + lag)[:900], (v + lag)[901:]])
    >>> reconcile_box1_frame_count(late, v)[2]
    '999 display times for 1000 vsyncs, but the latency of the last 50 frames with a flip is 16.7 ms from the block median: the missing flip(s) are not at the end of the block'
    """
    n_vsyncs, n_times = len(vsyncs), len(display_times)
    common = min(n_vsyncs, n_times)
    if common == 0:
        return display_times, np.array([], dtype=np.intp), "no display times"
    latency = display_times[:common] - vsyncs[:common]
    median_latency = float(np.median(latency))
    end_dev = float(np.median(latency[-END_CHECK_FRAMES:])) - median_latency
    kind = "missing" if n_times < n_vsyncs else "extra"
    if abs(end_dev) >= 0.5 * FRAME_INTERVAL:
        return (
            display_times,
            np.array([], dtype=np.intp),
            f"{n_times} display times for {n_vsyncs} vsyncs, but the latency of the last "
            f"{min(END_CHECK_FRAMES, common)} frames with a flip is {end_dev * 1e3:.1f} ms from the "
            f"block median: the {kind} flip(s) are not at the end of the block",
        )
    if n_times > n_vsyncs:
        n_extra = n_times - n_vsyncs
        last_display = vsyncs[-1] + median_latency
        if n_extra > MAX_EXTRA_END_FLIPS or np.any(
            display_times[n_vsyncs:] < last_display + 0.5 * FRAME_INTERVAL
        ):
            return (
                display_times,
                np.array([], dtype=np.intp),
                f"{n_times} display times for {n_vsyncs} vsyncs: the {n_extra} extra flip(s) "
                "aren't all after the display of the block's last frame",
            )
        return display_times[:n_vsyncs], np.array([], dtype=np.intp), None
    n_missing = n_vsyncs - n_times
    if n_missing > MAX_MISSING_END_FLIPS:
        return (
            display_times,
            np.array([], dtype=np.intp),
            f"{n_times} frame display times, but {n_vsyncs} vsyncs: too many missing frames",
        )
    invented = np.arange(n_times, n_vsyncs, dtype=np.intp)
    times = np.concatenate([display_times, vsyncs[invented] + median_latency])
    if not np.all(np.diff(times[n_times - 1 :]) > 0):
        return (
            display_times,
            np.array([], dtype=np.intp),
            f"{n_times} frame display times, but {n_vsyncs} vsyncs, and vsync + median latency "
            "for the missing frames at the end is out of order",
        )
    return times, invented, None


def replace_unsupported_first_frame(
    display_times: npt.NDArray[np.float64],
    vsyncs: npt.NDArray[np.float64],
    vsync_rising_edges: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], bool]:
    """Box-1 blocks whose first frame is an OFF frame: its display time isn't measured (there's
    no ON-ON interval around it, so it is extrapolated from the first ON interval), and is
    wrong if the frame was held. If it is half a refresh or more from the block's median
    latency relative to *both* its vsync falling edge and its rising edge, nothing supports it:
    it gets vsync + the block's median latency (and is marked like the invented end frames,
    `frame_source` 2). (666986_2023-08-17 and 5 other 2023 RFMapping blocks: frame 0, an RF stim
    onset, held for 3 refreshes, was placed 30-34 ms late.)

    >>> v = np.arange(100) / 59.95
    >>> t = v + 0.0275
    >>> t[0] += 0.030
    >>> out, replaced = replace_unsupported_first_frame(t, v, v - 0.0155)
    >>> replaced, round(float(out[0] - v[0]), 4)
    (True, 0.0275)
    """
    latency = display_times - vsyncs
    rise_latency = display_times - vsync_rising_edges
    med, rise_med = float(np.median(latency)), float(np.median(rise_latency))
    half = 0.5 * FRAME_INTERVAL
    if abs(latency[0] - med) < half or abs(rise_latency[0] - rise_med) < half:
        return display_times, False
    replacement = vsyncs[0] + med
    if len(display_times) > 1 and replacement >= display_times[1]:
        return display_times, False
    out = display_times.copy()
    out[0] = replacement
    logger.warning(
        f"first (OFF) frame's extrapolated display time is {(display_times[0] - replacement) * 1e3:+.1f} ms "
        "from its vsync + the median latency, and disagrees with its rising edge too: "
        "replaced with vsync + median latency"
    )
    return out, True


def get_residual_step_problem(
    display_times: npt.NDArray[np.float64],
    vsyncs: npt.NDArray[np.float64],
    window: int = RESIDUAL_STEP_WINDOW,
) -> str | None:
    """None if the running median (over `window` frames) of the display - vsync latency stays
    within half a refresh of the block's median latency; otherwise a description of the
    stretch where it doesn't.

    A stretch of frames whose diode flips are assigned one frame off (a flip missing in
    one place and an extra one in another, or one missing in the middle of a block with
    the count made up at the end) has a latency a whole refresh from the rest of the block.
    Isolated frames with late vsync falling edges (see `get_frame_display_times`) don't move
    the running median. Across the 2,072 photodiode blocks of 345 sessions in
    hub experiments/timing-quartile (audit_display.py), 1 block had such a stretch
    (706401_2024-04-24 SpontaneousRewards: the first 551 frames a refresh late, with the
    missing flip invented at the end).

    >>> v = np.arange(5000) / 59.95
    >>> lag = AVERAGE_VSYNC_TO_DIODE_FLIP_TIME
    >>> get_residual_step_problem(v + lag, v) is None
    True
    >>> t = v + lag
    >>> t[:1000] += 1 / 59.95
    >>> get_residual_step_problem(t, v)
    'display - vsync latency is 16.7 ms from the block median (19.6 ms) over frames 0-999 (running median of 301 frames): diode flips are assigned to the wrong frames there'
    """
    import scipy.ndimage

    latency = display_times - vsyncs
    if len(latency) < window:
        return None
    median_latency = float(np.median(latency))
    running = scipy.ndimage.median_filter(latency, size=window, mode="reflect")
    dev = running - median_latency
    bad = np.flatnonzero(np.abs(dev) >= 0.5 * FRAME_INTERVAL)
    if not len(bad):
        return None
    worst = float(dev[np.argmax(np.abs(dev))])
    # the running median crosses half a refresh where the majority of the window does, so
    # for a stretch longer than half a window this is its extent
    first, last = int(bad[0]), int(bad[-1])
    return (
        f"display - vsync latency is {worst * 1e3:.1f} ms from the block median "
        f"({median_latency * 1e3:.1f} ms) over frames {first}-{last} (running median of "
        f"{window} frames): diode flips are assigned to the wrong frames there"
    )


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


def get_held_off_frame_refreshes(
    vsync_times: npt.NDArray,
    vsync_rising_times: npt.NDArray | None,
    on_flip_times: npt.NDArray,
    is_first_frame_off: bool,
    refresh_interval: float,
) -> npt.NDArray[np.float64]:
    """For each diode ON flip (diodeBoxFrameInterval 1): the number of refreshes from the ON
    flip to the display of the next frame (the OFF frame), from the frame signal alone, or NaN
    if the frame signal can't tell.

    Used by `get_frame_display_times` to place the OFF frame inside a long ON-ON interval (a
    held frame), where the time of the diode's OFF (falling) edge isn't reliable: while a white
    sync square is held, its measured luminance sags or flickers (phantom flicker), so after a
    hold of 2 or more refreshes the OFF edge can come more than half a refresh early, and from
    the edge alone the OFF frame is then placed a refresh early.

    - `vsync_times`: vsync falling edges, one per frame, + `AVERAGE_VSYNC_TO_DIODE_FLIP_TIME`
    - `vsync_rising_times`: frame-signal rising edge of each frame (None: all NaN returned)
    - `on_flip_times`: diode ON (rising) flips: flip i is display index 2 * i +
      `is_first_frame_off`

    The frame signal goes high, the frame is drawn and flipped, and the signal goes low when
    `flip()` returns, at the vertical blank where the frame was swapped in. The frame is on
    screen one latency later, like every other frame: this is what vsync + constant lag
    assumes. The exception is a late return: the swap happens on time but `flip()` returns
    late, by up to a refresh or more, and the frame is on screen before its falling edge +
    latency. So the falling edge is used only if:

    1. the frame's pulse (rising to falling edge) is shorter than `HELD_FRAME_MAX_PULSE`
       refreshes. A late return of a whole refresh makes the pulse at least a refresh longer
       than the wait for the vertical blank;
    2. falling edge + latency is within `HELD_FRAME_GRID_TOLERANCE` refreshes of a whole
       number of refreshes after the ON flip (a return late by part of a refresh isn't).

    Otherwise the result is NaN, and `get_frame_display_times` uses the diode's OFF edge, as
    npc_stim <= 0.1.29 did for every held frame.

    The latency is the block's own: the median of (ON flip - vsync falling edge) over the
    block, which also gives the index offset between diode flips and vsyncs when the first
    flip of the block is missing or extra (-1, 0 or +1 frame, corrected at the end of
    `get_frame_display_times`). All NaN if the offset is anything else.

    Evidence (hub/experiments/npc-stim-held-frames): ON frames have the same frame-signal
    timing as OFF frames and their display time is measured (the diode's rising edge is
    reliable), so they show when a frame's falling edge can be trusted; the NI-DAQ analog
    photodiode (2025 sessions) shows the OFF frames directly.

    >>> R = 1 / 60
    >>> falling = np.arange(10) * R
    >>> falling[5:] += 2 * R                     # frame 4 (ON) held for 3 refreshes
    >>> v = falling + AVERAGE_VSYNC_TO_DIODE_FLIP_TIME
    >>> on = v[::2] + 0.0005                     # diode ON flips 0.5 ms after vsync + latency
    >>> rising = falling - 0.9 * R
    >>> get_held_off_frame_refreshes(v, rising, on, False, R).tolist()
    [1.0, 1.0, 3.0, 1.0, 1.0]
    >>> rising[5] = falling[5] - 2.9 * R         # frame 5: pulse of 2.9 refreshes
    >>> get_held_off_frame_refreshes(v, rising, on, False, R).tolist()
    [1.0, 1.0, nan, 1.0, 1.0]
    """
    n_refreshes = np.full(len(on_flip_times), np.nan)
    if vsync_rising_times is None or not len(on_flip_times):
        return n_refreshes
    vsync_times = np.asarray(vsync_times, dtype=float)
    vsync_rising_times = np.asarray(vsync_rising_times, dtype=float)
    on_frame_idx = 2 * np.arange(len(on_flip_times)) + int(is_first_frame_off)
    is_in_block = on_frame_idx < len(vsync_times)
    if not is_in_block.any():
        return n_refreshes
    # vsync_times include the average latency: the median is the offset in frames, plus
    # this block's difference from the average latency
    median_latency = float(
        np.median(on_flip_times[is_in_block] - vsync_times[on_frame_idx[is_in_block]])
    )
    frame_offset = round(median_latency / refresh_interval)
    if abs(frame_offset) > 1:
        return n_refreshes
    latency = median_latency - frame_offset * refresh_interval
    off_vsync_idx = on_frame_idx + 1 + frame_offset
    has_vsync = (off_vsync_idx >= 0) & (
        off_vsync_idx < min(len(vsync_times), len(vsync_rising_times))
    )
    k = off_vsync_idx[has_vsync]
    # the rising edges are as measured: remove the average latency from the falling edges
    pulse = (
        vsync_times[k] - AVERAGE_VSYNC_TO_DIODE_FLIP_TIME - vsync_rising_times[k]
    ) / refresh_interval
    refreshes = (vsync_times[k] + latency - on_flip_times[has_vsync]) / refresh_interval
    is_usable = (
        (pulse > 0)
        & (pulse < HELD_FRAME_MAX_PULSE)
        & (np.abs(refreshes - np.round(refreshes)) < HELD_FRAME_GRID_TOLERANCE)
    )
    n_refreshes[has_vsync] = np.where(is_usable, np.round(refreshes), np.nan)
    return n_refreshes


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
    vsync_rising_times: npt.NDArray | None = None,
) -> npt.NDArray[np.float64]:
    """Estimate the display time of each frame from vsync falling edges and diode flips.

    - `vsync_rising_times`: optional, the rising edge of the frame signal for each vsync (falling
      edge). TaskControl raises the frame signal before drawing and calling `flip()`, and lowers it
      after `flip()` returns, so a frame can't be displayed before its rising edge +
      `MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC`. The falling edge can come late, from a delay
      somewhere between the buffer swap and the falling-edge write; the frame is displayed on
      time (99.4% of 23,281 isolated late falling edges in box-1 blocks, TTL and NI-DAQ AI0:
      npc_sessions experiments/phantom-flicker/late_return_check.log; most of these delays don't
      reach PsychoPy's frameIntervals). Used when diodeBoxFrameInterval > 1 to place frames within
      a diode interval that contains dropped refreshes, and when it is 1 to place the OFF frame
      after a held ON frame (below).

    diodeBoxFrameInterval 1: ON frames get their diode (rising) flip. An OFF frame is put
    halfway between its neighbours, except in an ON-ON interval of 3 or more refreshes (a frame
    was held), where the refreshes are split between the ON and the OFF frame:

    - from the OFF frame's vsync falling edge, if its frame-signal pulse shows that the
      falling edge isn't late (`get_held_off_frame_refreshes`; needs `vsync_rising_times`)
    - otherwise from the time of the diode's OFF edge, `asym_round(..., 0.9)`, as npc_stim
      <= 0.1.29 did for every held frame. The OFF edge after a held white square can come
      more than half a refresh early (phantom flicker below the comparator threshold), which
      put the OFF frame a refresh early, and failed blocks with many such frames ("Too many
      frames are misaligned with vsyncs").

    The two agree for all but those frames, so blocks without them are unchanged.
    """
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

            # refreshes from each ON flip to the next (OFF) frame according to the frame
            # signal: NaN where it can't tell
            frame_signal_refreshes_to_off = get_held_off_frame_refreshes(
                vsync_times=vsync_times,
                vsync_rising_times=vsync_rising_times,
                on_flip_times=on_flip_times,
                is_first_frame_off=bool(is_first_frame_off),
                refresh_interval=median_refresh_interval,
            )

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
                # the OFF edge can be early after a held ON frame: use the OFF frame's vsync
                # falling edge where the frame signal shows it isn't late
                if 1 <= frame_signal_refreshes_to_off[idx] < n_refresh_in_interval:
                    n_frames_to_off = np.asarray(
                        int(frame_signal_refreshes_to_off[idx])
                    )
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
                # - vsync intervals aren't discretized: count those over 1.5 frames (a dropped
                #   refresh), not over `long_thr` (1.1 frames), which counts vsync jitter
                assert (
                    x := len(
                        all_vsync_diffs[all_vsync_diffs > 1.5 * median_refresh_interval]
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
                # figure out which frames were held, and how many refreshes were dropped
                # after each: a vsync interval of k refreshes is k - 1 dropped refreshes.
                # (npc_stim <= 0.1.32 used k, so with 2+ dropped refreshes the counts never
                # added up and were re-apportioned: a frame held for 3 refreshes in a box-3
                # interval was split as 2 + 2, putting the next frame - often an RFMapping
                # stim onset - a refresh early: 807082_2025-09-30 RFMapping frame 22010)
                dropped_per_interval = np.round(vsync_intervals / FRAME_INTERVAL) - 1
                long_idx = np.flatnonzero(dropped_per_interval > 0).tolist()
                long_refreshes = dropped_per_interval[long_idx]
                # long_refreshes is the number of dropped screen refreshes after each held frame
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
                        if not np.all(long_refreshes > 0):
                            # more long vsync intervals than dropped refreshes (previously an
                            # AssertionError): e.g. a late falling edge gives a long-short pair of
                            # vsync intervals with no dropped refresh.
                            # Place frames with `place_frames_in_diode_interval` below.
                            long_idx = None
                    else:
                        msg = f"{stim_id} | Faulty locating of dropped frames - something doesn't add up: {vsync_intervals=}, {long_refreshes=}"
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
                if long_idx is not None and i - 1 in long_idx:
                    i += int(long_refreshes[long_idx.index(i - 1)])
                t = evenly_spaced_times[i]
                new_times.append(t)
                # if the last interval is a long one nothing happens: inserting the next flip will
                # produce the long interval
            assert len(new_times) == vsyncs_per_flip
            frames = slice(idx * vsyncs_per_flip, (idx + 1) * vsyncs_per_flip)
            if (idx + 1) * vsyncs_per_flip <= len(vsync_times):
                if vsync_rising_times is not None and (
                    idx + 1
                ) * vsyncs_per_flip <= len(vsync_rising_times):
                    lower_bound = (
                        vsync_rising_times[frames] + MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC
                    )
                    # where the frame-signal pulse is short, the falling edge isn't late: it
                    # marks the vertical blank where the frame was swapped in, so the frame
                    # can't be on screen sooner than the min latency after it (as for held
                    # frames in box-1 blocks: see `HELD_FRAME_MAX_PULSE`). Without this, a
                    # late-falling-edge long/short pair of vsync intervals next to a held frame
                    # could still put the next frame a refresh early
                    falling = vsync_times[frames] - AVERAGE_VSYNC_TO_DIODE_FLIP_TIME
                    short_pulse = (
                        falling - vsync_rising_times[frames]
                        <= HELD_FRAME_MAX_PULSE * FRAME_INTERVAL
                    )
                    lower_bound = np.where(
                        short_pulse,
                        np.maximum(
                            lower_bound, falling + MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC
                        ),
                        lower_bound,
                    )
                else:
                    lower_bound = np.full(vsyncs_per_flip, -np.inf)
                if long_idx is None or np.any(
                    np.array(new_times)[1:] < lower_bound[1:]
                ):
                    # placement from vsync falling-edge intervals put a frame before its frame
                    # signal went high: the falling edges are late (frame displayed on time), so
                    # place frames using the rising edges as a lower bound instead
                    new_times = place_frames_in_diode_interval(
                        slot_times=evenly_spaced_times,
                        lower_bounds=lower_bound,
                        targets=vsync_times[
                            frames
                        ],  # already shifted by the average latency
                    )
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
