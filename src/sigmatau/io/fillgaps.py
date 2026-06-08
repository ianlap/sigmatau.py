"""Howe & Schlossberger gap imputation — mirrors ``io/fillgaps.jl``.

Reflect-and-invert ``sizegap`` points from the larger neighbouring run (or half
from each side), low-pass filter via an FFT exp(-|k|/N) shaping, and add an
endpoint-matched linear ramp; iterate outward from the largest run. Deterministic
(no RNG), so it is parity-validated against the Julia oracle.

Gap positions are kept 1-based inclusive (as in the Julia source) so the boundary
arithmetic is identical; array access converts with ``-1``.
"""

from __future__ import annotations

import math

import numpy as np

from ..types import FrequencyData, PhaseData


def _howe_filter(x: np.ndarray) -> np.ndarray:
    """Symmetric reflect-and-FFT-shape filter; returns the central third."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    pad = np.concatenate((x[::-1], x, x[::-1]))
    big_n = pad.size
    spec = np.fft.fft(pad)
    half = big_n // 2
    filt = np.empty(big_n, dtype=np.float64)
    for k in range(half):
        filt[k] = math.exp(-k / big_n)
    if big_n % 2 == 0:
        for k in range(half):
            filt[big_n - 1 - k] = filt[k]
    else:
        filt[half] = filt[half - 1]
        for k in range(half):
            filt[big_n - 1 - k] = filt[k]
    yp = np.fft.ifft(spec * filt).real
    lo = round(big_n / 3) + 1  # 1-based
    hi = round(2 * big_n / 3)  # 1-based inclusive
    return yp[lo - 1 : hi]


def _filter_all(x: np.ndarray) -> np.ndarray:
    """Apply ``_howe_filter`` to each contiguous finite run, leaving NaNs alone."""
    y = np.full(x.size, np.nan)
    n = x.size
    i = 0
    while i < n:
        if np.isnan(x[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(x[j]):
            j += 1
        y[i:j] = _howe_filter(x[i:j])
        i = j
    return y


def _identify_gaps(x: np.ndarray) -> list[tuple[int, int]]:
    """1-based inclusive (start, stop) ranges of the contiguous NaN runs."""
    gaps: list[tuple[int, int]] = []
    n = x.size
    i = 0
    while i < n:
        if np.isnan(x[i]):
            j = i
            while j < n and np.isnan(x[j]):
                j += 1
            gaps.append((i + 1, j))  # 1-based inclusive
            i = j
        else:
            i += 1
    return gaps


def _fill_singletons(x: np.ndarray, gaps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Fill 1-sample gaps (neighbour mean / nearest at edges); drop them from ``gaps``."""
    n = x.size
    kept: list[tuple[int, int]] = []
    for s, e in gaps:
        if s != e:
            kept.append((s, e))
            continue
        if s == 1:
            x[0] = x[1]
        elif s == n:
            x[n - 1] = x[n - 2]
        else:
            x[s - 1] = 0.5 * (x[s - 2] + x[s])
    return kept


def _choose_initial_gap(gaps: list[tuple[int, int]]) -> int:
    """1-based index of the gap following the largest run between gaps."""
    if len(gaps) <= 1:
        return 1
    maxdiff = 0
    pick = 1
    for k in range(len(gaps) - 1):
        d = gaps[k + 1][0] - gaps[k][1]
        if d > maxdiff:
            maxdiff = d
            pick = k + 2
    return pick


# --- per-gap point selection (returns (side, pts)) -------------------------


def _check_right(x, gaps, curgap, sizegap, gap_num, gap_total):
    rightlen = x.size - curgap[1]
    if rightlen >= sizegap:
        if gap_num == gap_total:
            return "right", x[curgap[1] : curgap[1] + sizegap]
        g_next = gaps[gap_num]  # gaps[gap_num+1] (1-based) -> python index gap_num
        if g_next[0] - curgap[1] - 1 >= sizegap:
            return "right", x[curgap[1] : curgap[1] + sizegap]
    return _check_both_sides(x, gaps, curgap, sizegap, gap_num, gap_total)


def _check_left(x, gaps, curgap, sizegap, gap_num, gap_total):
    if gap_num > 1:
        g_prev = gaps[gap_num - 2]
        if curgap[0] - g_prev[1] - 1 >= sizegap:
            return "left", x[curgap[0] - 1 - sizegap : curgap[0] - 1]
    elif curgap[0] > sizegap:
        return "left", x[curgap[0] - 1 - sizegap : curgap[0] - 1]
    return _check_right(x, gaps, curgap, sizegap, gap_num, gap_total)


def _both(x, curgap, half):
    return (x[curgap[0] - 1 - half : curgap[0] - 1], x[curgap[1] : curgap[1] + half])


def _check_both_sides(x, gaps, curgap, sizegap, gap_num, gap_total):
    half = sizegap // 2
    if half == 0:
        return "none", None
    if gap_num == 1:
        if gap_total == 1:
            if curgap[0] > half and curgap[1] <= x.size - half:
                return "both", _both(x, curgap, half)
            return "done", None
        g_next = gaps[gap_num]
        if curgap[0] > half and g_next[0] - curgap[1] - 1 >= half:
            return "both", _both(x, curgap, half)
        return "none", None
    if gap_num == gap_total:
        g_prev = gaps[gap_num - 2]
        if curgap[0] - g_prev[1] - 1 >= half and curgap[1] <= x.size - half:
            return "both", _both(x, curgap, half)
        return "none", None
    g_prev = gaps[gap_num - 2]
    g_next = gaps[gap_num]
    if curgap[0] - g_prev[1] - 1 >= half and g_next[0] - curgap[1] - 1 >= half:
        return "both", _both(x, curgap, half)
    return "none", None


def _fill_one_gap(x, gaps, gap_num, reverse):
    """Fill ``gaps[gap_num-1]`` in place; return (status, new_gap_num)."""
    gap_total = len(gaps)
    curgap = gaps[gap_num - 1]
    sizegap = curgap[1] - curgap[0] + 1

    if reverse:
        if curgap[1] > x.size - sizegap:
            side, pts = _check_left(x, gaps, curgap, sizegap, gap_num, gap_total)
        else:
            side, pts = _check_right(x, gaps, curgap, sizegap, gap_num, gap_total)
    else:
        if curgap[0] <= sizegap:
            side, pts = _check_right(x, gaps, curgap, sizegap, gap_num, gap_total)
        else:
            side, pts = _check_left(x, gaps, curgap, sizegap, gap_num, gap_total)

    if side == "none":
        return ("advance", gap_num - 1 if reverse else gap_num + 1)
    if side == "done":
        return ("done", gap_num)

    if side == "both":
        left_pts, right_pts = pts
        left = -left_pts[::-1]
        right = -right_pts[::-1]
        filt_left = _howe_filter(left)
        filt_right = _howe_filter(right)
        shift = filt_left[-1] - filt_right[0]
        right = right + shift
        if sizegap % 2 == 1:
            midval = 0.5 * (left[-1] + right[0])
            fillvec = np.concatenate((left, [midval], right))
        else:
            fillvec = np.concatenate((left, right))
    else:
        fillvec = -pts[::-1]

    fillvec = _howe_filter(fillvec)

    filt_data = _filter_all(x)
    start_anchor = filt_data[curgap[0] - 2]
    end_anchor = filt_data[curgap[1]]
    ramp = np.linspace(start_anchor - fillvec[0], end_anchor - fillvec[-1], sizegap)

    for k, idx in enumerate(range(curgap[0], curgap[1] + 1)):
        x[idx - 1] = fillvec[k] + ramp[k]

    del gaps[gap_num - 1]
    if reverse:
        new_gap_num = gap_num - 1
    else:
        new_gap_num = len(gaps) if gap_num > len(gaps) else gap_num
    return ("ok", new_gap_num)


def _howe_fillgaps_core(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Howe fill of an equispaced vector with NaN gaps; returns (filled, mask)."""
    xv = np.array(x, dtype=np.float64)
    n = xv.size
    nan0 = np.isnan(xv)
    if not nan0.any():
        return xv, np.zeros(n, dtype=bool)

    gaps = _identify_gaps(xv)
    gaps = _fill_singletons(xv, gaps)
    gap_num = _choose_initial_gap(gaps)

    while gaps:
        while 1 <= gap_num <= len(gaps):
            status, gap_num = _fill_one_gap(xv, gaps, gap_num, False)
            if status == "done":
                raise RuntimeError(
                    "fillgaps: gap topology cannot be filled (record too short vs gap span)"
                )
            if not gaps:
                break
        if not gaps:
            break

        gap_num = min(gap_num, len(gaps))
        while gap_num >= 1 and gaps:
            status, gap_num = _fill_one_gap(xv, gaps, gap_num, True)
            if status == "done":
                raise RuntimeError(
                    "fillgaps: gap topology cannot be filled (record too short vs gap span)"
                )
            if gap_num < 1:
                break
        if not gaps:
            break
        gap_num = _choose_initial_gap(gaps)

    filled_mask = nan0 & ~np.isnan(xv)
    return xv, filled_mask


def _make_equispaced(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Snap an irregular (t, x) series to a uniform grid (spacing min Δt), NaN in gaps."""
    if t.size != x.size:
        raise ValueError("_make_equispaced: t and x must have equal length")
    if t.size < 2:
        raise ValueError("_make_equispaced: need at least 2 samples")
    dt = float(np.min(np.diff(t)))
    if not dt > 0:
        raise ValueError("_make_equispaced: time vector is not strictly increasing")
    n = round((t[-1] - t[0]) / dt) + 1
    tfilled = t[0] + dt * np.arange(n, dtype=np.float64)
    xfilled = np.full(n, np.nan)
    for i in range(t.size):
        idx = round((t[i] - t[0]) / dt) + 1  # 1-based
        xfilled[idx - 1] = x[i]
    return tfilled, xfilled


def fillgaps(data: PhaseData | FrequencyData) -> PhaseData | FrequencyData:
    """Impute NaN samples via Howe & Schlossberger. Returns a new record (same τ₀)."""
    if isinstance(data, FrequencyData):
        yfilled, _ = _howe_fillgaps_core(data.y)
        return FrequencyData(yfilled, data.tau0)
    xfilled, _ = _howe_fillgaps_core(data.x)
    return PhaseData(xfilled, data.tau0)


__all__ = ["fillgaps"]
