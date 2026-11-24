"""Placement of the OFF frame after a held frame, diodeBoxFrameInterval 1
(`get_frame_display_times`, `get_held_off_frame_refreshes`): synthetic blocks.

Model (hub/experiments/npc-stim-held-frames): frame k is swapped in at a vertical blank and is
on the monitor one latency later, for `refreshes[k]` refreshes. Its frame signal goes high
`pulse` refreshes before the swap and low at the swap (the vsync falling edge), or a refresh
after it for a late return. The sync square toggles every frame: the diode's rising edge is at
the ON frame's display time, its falling edge `fall_delay` refreshes after the OFF frame's, or
earlier after a held ON frame (phantom flicker).

Without `vsync_rising_times` the estimator behaves as npc_stim <= 0.1.29 (OFF frame placed
from the diode's OFF edge alone), which the tests use as the reference for the old behaviour.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

import npc_stim.display_times as dt

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"
R = 1 / 60
LATENCY = dt.AVERAGE_VSYNC_TO_DIODE_FLIP_TIME + 0.0005


@dataclasses.dataclass
class Block:
    vsyncs: np.ndarray
    """vsync falling edges, one per frame"""
    rising: np.ndarray
    """frame-signal rising edge of each frame"""
    on: np.ndarray
    """diode rising edges"""
    off: np.ndarray
    """diode falling edges"""
    display: np.ndarray
    """true display time of each frame (rising-edge convention)"""

    def estimate(self, frame_signal: bool = True, **kwargs) -> np.ndarray:
        return dt.get_frame_display_times(
            vsync_times=self.vsyncs,
            on_flip_times=self.on,
            off_flip_times=self.off,
            vsyncs_per_flip=1,
            check_lengths=False,
            stim_id="synthetic",
            vsync_rising_times=self.rising if frame_signal else None,
            **kwargs,
        )


def make_block(
    refreshes: np.ndarray,
    *,
    first_frame_on: bool = False,
    early_off_edges: tuple[int, ...] = (),
    late_returns: tuple[int, ...] = (),
    late_by: float = 1.0,
    pulse: float = 0.9,
    fall_delay: float = 0.3,
    early_by: float = 0.7,
    missing_first_flip: bool = False,
    extra_first_flip: bool = False,
) -> Block:
    """- `early_off_edges`: OFF frames whose diode falling edge is `early_by` refreshes early
    - `late_returns`: frames whose vsync falling edge is `late_by` refreshes after the swap
    - `missing_first_flip`: no diode edge for the first frame
    - `extra_first_flip`: a diode edge of the first frame's polarity one refresh before it
    """
    refreshes = np.asarray(refreshes, dtype=float)
    swap = 100 + np.concatenate(([0], np.cumsum(refreshes[:-1]))) * R
    display = swap + LATENCY
    vsyncs = swap.copy()
    vsyncs[list(late_returns)] += late_by * R
    rising = swap - pulse * R
    frames = np.arange(len(refreshes))
    is_on = (frames % 2 == 0) == first_frame_on
    on = display[is_on]
    off = display[~is_on] + fall_delay * R
    off_frames = frames[~is_on]
    off[np.isin(off_frames, early_off_edges)] -= early_by * R
    if missing_first_flip:
        if is_on[0]:
            on = on[1:]
        else:
            off = off[1:]
    if extra_first_flip:
        if is_on[0]:
            on, off = np.r_[display[0] - R, on], np.r_[display[0] - R + 0.3 * R, off]
        else:
            # ON then OFF again before the first (OFF) frame
            on = np.r_[display[0] - R, on]
    return Block(vsyncs=vsyncs, rising=rising, on=on, off=off, display=display)


def refreshes_with_holds(n: int, holds: dict[int, int]) -> np.ndarray:
    r = np.ones(n, dtype=int)
    for frame, n_refreshes in holds.items():
        r[frame] = n_refreshes
    return r


def assert_display(times: np.ndarray, block: Block) -> None:
    assert len(times) == len(block.display)
    np.testing.assert_allclose(times, block.display, atol=1e-9)


def test_no_held_frames_is_unchanged_by_frame_signal() -> None:
    block = make_block(np.ones(400, dtype=int), first_frame_on=True)
    with_signal, without = block.estimate(), block.estimate(frame_signal=False)
    assert np.array_equal(with_signal, without)
    assert_display(with_signal, block)


@pytest.mark.parametrize("first_frame_on", (True, False))
@pytest.mark.parametrize("held_refreshes", (2, 3, 4))
def test_held_frames_with_normal_off_edges_are_unchanged(
    first_frame_on: bool, held_refreshes: int
) -> None:
    """Held ON and held OFF frames, OFF edges on time: the frame signal and the OFF edge
    agree, so the result is bit-identical to the OFF-edge-only estimate."""
    n = 600
    on_parity = 0 if first_frame_on else 1
    holds = {
        f: held_refreshes
        for f in (
            50 + on_parity,
            151 - on_parity,
            250 + on_parity,
            351 - on_parity,
            450 + on_parity,
        )
    }
    block = make_block(refreshes_with_holds(n, holds), first_frame_on=first_frame_on)
    with_signal, without = block.estimate(), block.estimate(frame_signal=False)
    assert np.array_equal(with_signal, without)
    assert_display(with_signal, block)


@pytest.mark.parametrize(
    "held_refreshes, early_by", ((3, 0.5), (3, 0.7), (2, 0.7), (5, 0.6))
)
def test_early_off_edge_after_held_on_frame(
    held_refreshes: int, early_by: float
) -> None:
    """ON frame held, OFF edge of the next frame early (phantom flicker): the OFF edge alone
    puts the OFF frame a refresh early; with the frame signal it's placed at its vsync.
    """
    n = 600
    held = (50, 150, 250, 350, 450)  # ON frames (first frame ON)
    block = make_block(
        refreshes_with_holds(n, dict.fromkeys(held, held_refreshes)),
        first_frame_on=True,
        early_off_edges=(151, 351),
        early_by=early_by,
    )
    assert_display(block.estimate(), block)
    old = block.estimate(frame_signal=False)
    moved = np.flatnonzero(~np.isclose(old, block.display, atol=1e-9))
    assert moved.tolist() == [151, 351]
    np.testing.assert_allclose(old[moved] - block.display[moved], -R, atol=1e-9)


def test_late_return_keeps_off_edge_placement() -> None:
    """OFF frame swapped in on time but its flip returns a refresh late (vsync intervals of 2
    and 1 refreshes, frames displayed for 1 and 2): its pulse is long, so the OFF edge decides.
    The vsync intervals alone would put the OFF frame a refresh late."""
    n = 600
    held_off = (51, 151, 251, 351, 451)  # OFF frames held for 2 refreshes
    block = make_block(
        refreshes_with_holds(n, dict.fromkeys(held_off, 2)),
        first_frame_on=True,
        late_returns=(151, 351),
    )
    intervals = np.round(np.diff(block.vsyncs) / R).astype(int)
    assert intervals[150:152].tolist() == [2, 1] and intervals[50:52].tolist() == [1, 2]
    times = block.estimate()
    assert_display(times, block)
    assert np.array_equal(times, block.estimate(frame_signal=False))
    refreshes = dt.get_held_off_frame_refreshes(
        block.vsyncs + dt.AVERAGE_VSYNC_TO_DIODE_FLIP_TIME,
        block.rising,
        block.on,
        False,
        R,
    )
    assert (
        np.isnan(refreshes[[75, 175]]).all()
        and not np.isnan(refreshes[[25, 125]]).any()
    )


def test_partly_late_falling_edge_keeps_off_edge_placement() -> None:
    """A vsync falling edge late by 0.7 refresh with a short pulse is off the refresh grid:
    not used."""
    n = 400
    block = make_block(
        refreshes_with_holds(n, {51: 2, 151: 2, 251: 2}),
        first_frame_on=True,
        late_returns=(151,),
        late_by=0.7,
        pulse=0.3,
    )
    times = block.estimate()
    assert_display(times, block)
    assert np.array_equal(times, block.estimate(frame_signal=False))


@pytest.mark.parametrize("flip", ("missing", "extra"))
def test_early_off_edges_with_first_flip_missing_or_extra(flip: str) -> None:
    """The first diode flip of the block missing (as in most task blocks: black square on a
    dark screen) or extra: diode flips are one frame off the vsyncs until the end of the
    estimate, and the frame signal is still read for the right frame."""
    n = 2000
    held = tuple(
        range(100, 1900, 100)
    )  # ON frames (first frame OFF: odd frames are ON)
    held = tuple(f + 1 for f in held)
    early = tuple(f + 1 for f in held[::4])
    block = make_block(
        refreshes_with_holds(n, dict.fromkeys(held, 3)),
        first_frame_on=False,
        early_off_edges=early,
        missing_first_flip=flip == "missing",
        extra_first_flip=flip == "extra",
    )
    times = block.estimate()
    assert_display(times, block)
    old = block.estimate(frame_signal=False)
    moved = np.flatnonzero(~np.isclose(old, block.display, atol=1e-9))
    assert moved.tolist() == list(early)


def test_many_early_off_edges_no_longer_fail_the_block() -> None:
    """As in the tasks of 702136_2024-03-05/06 and 708016_2024-05-01: the first flip is
    missing and 40% of the held ON frames have an early OFF edge (consecutive ones: the vote
    is median-filtered over 3). From the OFF edge alone, each gives two display intervals of 2
    refreshes instead of 3 + 1, the vote on the first flip gets under 90% and the estimate
    fails."""
    n = 6000
    held = tuple(f + 1 for f in range(100, 5900, 100))
    early = tuple(f + 1 for i, f in enumerate(held) if i % 5 in (0, 1))
    block = make_block(
        refreshes_with_holds(n, dict.fromkeys(held, 3)),
        first_frame_on=False,
        early_off_edges=early,
        missing_first_flip=True,
    )
    with pytest.raises(AssertionError, match="Too many frames are misaligned"):
        block.estimate(frame_signal=False)
    assert_display(block.estimate(), block)


def test_held_frames_at_block_ends() -> None:
    """Held frames next to the first and last frames: no vsync or rising edge is read out of
    range."""
    n = 300
    block = make_block(
        refreshes_with_holds(n, {0: 3, 100: 3, 200: 3, n - 3: 3}), first_frame_on=True
    )
    times = block.estimate(check_vsyncs=False)
    np.testing.assert_allclose(times[: n - 1], block.display[: n - 1], atol=1e-9)


def test_frame_signal_ignored_if_flips_are_not_aligned_to_vsyncs() -> None:
    """Diode flips 3 frames from the vsyncs (not a missing/extra first flip): no estimate
    from the frame signal."""
    block = make_block(refreshes_with_holds(400, {50: 3, 150: 3}), first_frame_on=True)
    refreshes = dt.get_held_off_frame_refreshes(
        block.vsyncs + dt.AVERAGE_VSYNC_TO_DIODE_FLIP_TIME - 3 * R,
        block.rising - 3 * R,
        block.on,
        False,
        R,
    )
    assert np.isnan(refreshes).all()
    assert np.isnan(
        dt.get_held_off_frame_refreshes(block.vsyncs, None, block.on, False, R)
    ).all()


def test_early_off_edges_in_recorded_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """90 s (5400 frames) of the 762526_2025-03-18 task from frame 148000, patched sync line,
    times in sync samples (10 us) from the first vsync. 9 frames are held for 3 refreshes;
    after 3 of them (chunk frames 1094, 2392, 4230: ON frames) the diode's OFF edge is 2.89,
    2.88 and 2.83 refreshes after the ON edge instead of 3.3, and npc_stim <= 0.1.29 put the
    next frame a refresh early, 4.3 ms after its vsync falling edge. The NI-DAQ analog
    photodiode shows the held white square sagging in its third refresh, then going dark
    (hub/experiments/npc-stim-held-frames)."""
    data = np.load(FIXTURES_DIR / "chunk_early_off_edges_after_held_frames.npz")
    vsyncs = data["vsyncs"] * 1e-5

    def estimate() -> np.ndarray:
        result = dt._estimate_block_display_times(
            sync_data_name="762526_2025-03-18",
            block_idx=0,
            stim_id="762526_2025-03-18_chunk",
            vsyncs=vsyncs,
            vsync_rising_edges=data["vsync_rising_edges"] * 1e-5,
            rising=data["rising"] * 1e-5,
            falling=data["falling"] * 1e-5,
            first_frame_start=float(vsyncs[0]),
            vsyncs_per_diode_flip=1,
            expected_vsyncs=len(vsyncs),
            diode_cleaned=True,
            raise_on_error=True,
        )
        assert result.reason is None and result.times is not None
        assert len(result.early_frames) == 0
        return result.times

    times = estimate()
    latency = times - vsyncs
    assert np.all(np.abs(latency - np.median(latency)) < 0.5 * R)
    held = np.flatnonzero(np.diff(vsyncs) > 1.5 * R)
    assert held.tolist() == [157, 1094, 1595, 2392, 2779, 3133, 3465, 4230, 5069]
    assert np.flatnonzero(np.diff(times) > 1.5 * R).tolist() == held.tolist()

    # without the frame signal (npc_stim <= 0.1.29): the three OFF frames are a refresh early
    monkeypatch.setattr(
        dt,
        "get_held_off_frame_refreshes",
        lambda *args, **kwargs: np.full(len(kwargs["on_flip_times"]), np.nan),
    )
    old = estimate()
    changed = np.flatnonzero(old != times)
    assert changed.tolist() == [1095, 2393, 4231]
    np.testing.assert_allclose(times[changed] - old[changed], R, atol=2e-4)
