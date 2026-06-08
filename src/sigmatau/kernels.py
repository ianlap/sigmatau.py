"""Raw deviation kernels — mirrors ``kernels.jl`` (MVP subset).

Internal ``_*_core`` functions: plain ``float64`` array in, ``float64`` array
out. NumPy-vectorized ports of the Julia kernels; the public PhaseData/
FrequencyData API lives in ``deviations.py``. Each overlapping difference is
NaN when the averaging factor leaves fewer than 2 analysis windows, matching the
Julia ``L``/``Ne`` guards exactly.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
from scipy.signal import correlate as sig_correlate

# Optional Numba acceleration for the loop-bound modified-total family and pdev.
# When unavailable (or disabled via SIGMATAU_NO_NUMBA), the pure-NumPy batched
# kernels below are used instead — same results, just slower on long records.
try:  # pragma: no cover - exercised by whichever backend is installed
    if os.environ.get("SIGMATAU_NO_NUMBA"):
        raise ImportError
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


def _prefix_sum(x: np.ndarray) -> np.ndarray:
    """``X[0] = 0``, ``X[i+1] = Σⱼ₌₀ⁱ x[j]`` — length ``N + 1`` (as in the Julia cores)."""
    out = np.empty(x.size + 1, dtype=np.float64)
    out[0] = 0.0
    np.cumsum(x, out=out[1:])
    return out


def _adev_core(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
    """Overlapping Allan deviation for each averaging factor ``m``."""
    n = x.size
    devs = np.empty(len(m_values), dtype=np.float64)
    for k, m in enumerate(m_values):
        length = n - 2 * m
        if length < 2:  # need ≥2 analysis windows; one window is a single difference
            devs[k] = np.nan
            continue
        d2 = x[2 * m : n] - 2.0 * x[m : n - m] + x[0:length]
        devs[k] = np.sqrt(np.dot(d2, d2) / (2.0 * length * m**2 * tau0**2))
    return devs


def _mdev_core(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
    """Modified Allan deviation via prefix sums (O(N) per ``m``)."""
    n = x.size
    cs = _prefix_sum(x)
    devs = np.empty(len(m_values), dtype=np.float64)
    for k, m in enumerate(m_values):
        ne = n - 3 * m + 1
        if ne < 2:  # need ≥2 estimates
            devs[k] = np.nan
            continue
        d = cs[3 * m : 3 * m + ne] - 3.0 * cs[2 * m : 2 * m + ne] + 3.0 * cs[m : m + ne] - cs[0:ne]
        devs[k] = np.sqrt(np.dot(d, d) / (2.0 * ne * m**4 * tau0**2))
    return devs


def _hdev_core(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
    """Overlapping Hadamard deviation (third differences) for each ``m``."""
    n = x.size
    devs = np.empty(len(m_values), dtype=np.float64)
    for k, m in enumerate(m_values):
        length = n - 3 * m
        if length < 2:  # need ≥2 analysis windows
            devs[k] = np.nan
            continue
        d3 = x[3 * m : n] - 3.0 * x[2 * m : n - m] + 3.0 * x[m : n - 2 * m] - x[0:length]
        devs[k] = np.sqrt(np.dot(d3, d3) / (6.0 * length * m**2 * tau0**2))
    return devs


def _mhdev_core(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
    """Modified Hadamard deviation via prefix sums (fourth differences, O(N) per ``m``)."""
    n = x.size
    cs = _prefix_sum(x)
    devs = np.empty(len(m_values), dtype=np.float64)
    for k, m in enumerate(m_values):
        ne = n - 4 * m + 1
        if ne < 2:  # need ≥2 estimates; MHDEV is a raw fourth-difference estimator
            devs[k] = np.nan
            continue
        d = (
            cs[4 * m : 4 * m + ne]
            - 4.0 * cs[3 * m : 3 * m + ne]
            + 6.0 * cs[2 * m : 2 * m + ne]
            - 4.0 * cs[m : m + ne]
            + cs[0:ne]
        )
        devs[k] = np.sqrt(np.dot(d, d) / (6.0 * ne * m**4 * tau0**2))
    return devs


def _totvar_extension(x: np.ndarray) -> np.ndarray:
    """Howe / SP1065 eqn 2 doubly-reflected (mean-flip) phase sequence.

    Returns ``E`` of length ``3N − 2`` such that the reflected sample ``x*_k``
    (1-indexed, per SP1065) is ``E[k + N - 2]``: left tail ``2x₁ − x₂₋ₖ`` for
    ``k ≤ 0``, the record itself for ``1 ≤ k ≤ N``, right tail ``2x_N − x₂ₙ₋ₖ``
    for ``k ≥ N+1``. Lets TOTDEV's three-point stencil run as a single
    vectorized pass.
    """
    left = 2.0 * x[0] - x[1:][::-1]  # x*_{2-N..0}, length N-1
    right = 2.0 * x[-1] - x[:-1][::-1]  # x*_{N+1..2N-1}, length N-1
    return np.concatenate((left, x, right))


def _totdev_core(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
    """Total deviation (Howe 1995 / SP1065 eqn 25 mean-flip extension)."""
    n = x.size
    devs = np.empty(len(m_values), dtype=np.float64)
    if n <= 2:
        devs.fill(np.nan)
        return devs
    ext = _totvar_extension(x)
    # Interior centers n = 2..N-1 (1-indexed) -> extension index c = n + N - 2.
    c = np.arange(n, 2 * n - 2)  # c for n = 2..N-1, length N-2
    for k, m in enumerate(m_values):
        if m >= n:
            devs[k] = np.nan
            continue
        d2 = ext[c + m] - 2.0 * ext[c] + ext[c - m]
        devs[k] = np.sqrt(np.dot(d2, d2) / (2.0 * (n - 2) * m**2 * tau0**2))
    return devs


def _mtie_core(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
    """Maximum Time Interval Error (ITU-T G.810): max peak-to-peak over windows of m+1.

    Monotonic-deque sliding window — O(N) per ``m``, O(N) memory. Units of
    seconds (a σ_x quantity); no τ rescaling. ``tau0`` is unused (kept for the
    uniform kernel signature).
    """
    del tau0  # MTIE is a phase excursion; tau0 does not enter.
    n = x.size
    devs = np.empty(len(m_values), dtype=np.float64)
    max_dq = np.empty(n, dtype=np.int64)
    min_dq = np.empty(n, dtype=np.int64)
    for k, m in enumerate(m_values):
        if n - m <= 0:
            devs[k] = np.nan
            continue
        win = m + 1
        max_h, max_t = 0, -1  # empty when tail < head
        min_h, min_t = 0, -1
        max_excursion = 0.0
        for j in range(n):
            # Drop indices that have left the trailing edge of the window.
            while max_t >= max_h and max_dq[max_h] <= j - win:
                max_h += 1
            while min_t >= min_h and min_dq[min_h] <= j - win:
                min_h += 1
            xj = x[j]
            while max_t >= max_h and x[max_dq[max_t]] <= xj:
                max_t -= 1
            max_t += 1
            max_dq[max_t] = j
            while min_t >= min_h and x[min_dq[min_t]] >= xj:
                min_t -= 1
            min_t += 1
            min_dq[min_t] = j
            if j >= win - 1:
                d = x[max_dq[max_h]] - x[min_dq[min_h]]
                if d > max_excursion:
                    max_excursion = d
        devs[k] = max_excursion
    return devs


def _pdev_batched(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
    """Parabolic deviation (Vernotte 2016/2020).

    The weighted parabolic sum ``Σ_k (½(m−1) − k)·(x_{i+k} − x_{i+k+m})`` is a
    fixed-weight correlation of ``x`` with weights ``w_k = ½(m−1) − k``. It is
    evaluated with ``scipy.signal.correlate(method="auto")`` — direct for small
    ``m``, FFT (O(N log N)) for large ``m`` — avoiding the O(N·m) cost of a plain
    correlation. At ``m = 1`` the weights collapse to zero, so PDEV(τ₀) ≡
    ADEV(τ₀) (Vernotte 2015) — delegate to ADEV.
    """
    n = x.size
    devs = np.empty(len(m_values), dtype=np.float64)
    for k, m in enumerate(m_values):
        if m < 1:
            devs[k] = np.nan
            continue
        if m == 1:
            devs[k] = _adev_core(x, [1], tau0)[0]
            continue
        m_windows = n - 2 * m
        if m_windows < 2:  # need ≥2 windows
            devs[k] = np.nan
            continue
        w = (m - 1) / 2.0 - np.arange(m, dtype=np.float64)
        corr = sig_correlate(x, w, mode="valid", method="auto")  # corr[i] = Σ_k w_k·x[i+k]
        asum = corr[0:m_windows] - corr[m : m + m_windows]
        var = 72.0 * np.dot(asum, asum) / (m_windows * m**6 * tau0**2)
        devs[k] = np.sqrt(var)
    return devs


# The modified-total family (mtot/htot/mhtot) removes a per-window slope, builds
# a 3× time-reverse extension, and runs a sliding difference per subsequence. The
# subsequences are processed in memory-bounded BATCHES as 2-D arrays so the work
# runs at vectorized C speed (no Python inner loop, no Numba). The reflected
# segment of each detrended window equals the window reversed, so the extension
# is ``[w[:, ::-1], w, w[:, ::-1]]`` row-wise. Peak memory per batch ≈ B·9m.
_BATCH_ELEMS = 1_000_000


def _batch_starts(nsubs: int, row_len: int) -> Sequence[range]:
    """Yield contiguous start-index blocks sized so a batch holds ≲ _BATCH_ELEMS."""
    b = max(1, _BATCH_ELEMS // row_len)
    return [range(off, min(off + b, nsubs)) for off in range(0, nsubs, b)]


def _total_sumsq_2d(ext: np.ndarray, m: int) -> np.ndarray:
    """Per-row Σ over the 6m sliding windows of ((a3−2a2+a1)/m)² (MTOT/HTOT reduction).

    ``ext`` is ``(B, 9m)``; returns ``(B,)``.
    """
    cs = np.empty((ext.shape[0], ext.shape[1] + 1), dtype=np.float64)
    cs[:, 0] = 0.0
    np.cumsum(ext, axis=1, out=cs[:, 1:])
    s = np.arange(6 * m)
    a1 = cs[:, s + m] - cs[:, s]
    a2 = cs[:, s + 2 * m] - cs[:, s + m]
    a3 = cs[:, s + 3 * m] - cs[:, s + 2 * m]
    d = (a3 - 2.0 * a2 + a1) / m
    return np.einsum("ij,ij->i", d, d)


def _mtotdev_batched(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
    """Modified Total deviation (Greenhall per-window time-reverse + half-mean slope)."""
    n = x.size
    devs = np.empty(len(m_values), dtype=np.float64)
    for k, m in enumerate(m_values):
        nsubs = n - 3 * m + 1
        if nsubs < 1:
            devs[k] = np.nan
            continue
        seg_len = 3 * m
        half_n = seg_len / 2.0
        hi_half = seg_len // 2
        lo_half_len = seg_len - hi_half
        cols = np.arange(seg_len)
        jj = cols.astype(np.float64)
        total = 0.0
        for blk in _batch_starts(nsubs, 9 * m):
            s0 = np.arange(blk.start, blk.stop)
            win = x[s0[:, None] + cols[None, :]]
            if m == 1:
                slope = (x[s0 + 2] - x[s0]) / (2.0 * tau0)
            else:
                s1 = win[:, :hi_half].sum(axis=1)
                s2 = win[:, hi_half:].sum(axis=1)
                slope = (s2 / lo_half_len - s1 / hi_half) / (half_n * tau0)
            w = win - slope[:, None] * tau0 * jj[None, :]
            ext = np.concatenate((w[:, ::-1], w, w[:, ::-1]), axis=1)
            total += _total_sumsq_2d(ext, m).sum() / (6.0 * m)
        devs[k] = np.sqrt(total / (2.0 * m**2 * tau0**2 * nsubs))
    return devs


def _htotdev_batched(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
    """Hadamard Total deviation, computed on the frequency series y = diff(x)/tau0."""
    n = x.size
    y = np.diff(x) / tau0
    ny = y.size
    devs = np.empty(len(m_values), dtype=np.float64)
    for k, m in enumerate(m_values):
        if m == 1:
            length = n - 3
            if length <= 0:
                devs[k] = np.nan
                continue
            d3 = x[3:n] - 3.0 * x[2 : n - 1] + 3.0 * x[1 : n - 2] - x[0 : n - 3]
            devs[k] = np.sqrt(np.dot(d3, d3) / (6.0 * length * tau0**2))
            continue
        n_iter = ny - 3 * m + 1
        if n_iter < 1:
            devs[k] = np.nan
            continue
        seg_len = 3 * m
        hi = seg_len // 2
        lo_start = -(-seg_len // 2) + 1  # ceil(seg_len/2) + 1, 1-indexed
        lo_count = seg_len - lo_start + 1
        denom = (0.5 * (seg_len - 1) + 1.0) if seg_len % 2 == 1 else (0.5 * seg_len)
        mid = seg_len // 2
        cols = np.arange(seg_len)
        jj = cols.astype(np.float64)
        total = 0.0
        for blk in _batch_starts(n_iter, 9 * m):
            s0 = np.arange(blk.start, blk.stop)
            win = y[s0[:, None] + cols[None, :]]
            m1 = win[:, :hi].sum(axis=1) / hi
            m2 = win[:, lo_start - 1 :].sum(axis=1) / lo_count
            slope = (m2 - m1) / denom
            w = win - slope[:, None] * (jj - mid)[None, :]
            ext = np.concatenate((w[:, ::-1], w, w[:, ::-1]), axis=1)
            total += _total_sumsq_2d(ext, m).sum() / (6.0 * m)
        devs[k] = np.sqrt(total / (6.0 * n_iter))
    return devs


def _mhtotdev_batched(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
    """Modified Hadamard Total deviation (SigmaTau-original; Greenhall methodology)."""
    n = x.size
    devs = np.empty(len(m_values), dtype=np.float64)
    for k, m in enumerate(m_values):
        if m < 1:
            devs[k] = np.nan
            continue
        nsubs = n - 4 * m + 1
        if nsubs < 1:
            devs[k] = np.nan
            continue
        lp = 3 * m + 1
        l3 = 3 * lp - 3 * m  # = 6m + 3
        half = lp // 2
        cols = np.arange(lp)
        jj = cols.astype(np.float64)
        n_avg = l3 + 1 - m
        total = 0.0
        for blk in _batch_starts(nsubs, 3 * lp):
            s0 = np.arange(blk.start, blk.stop)
            win = x[s0[:, None] + cols[None, :]]
            s1 = win[:, :half].mean(axis=1)
            s2 = win[:, half:].mean(axis=1)
            slope = (s2 - s1) / ((lp / 2.0) * tau0)
            w = win - slope[:, None] * tau0 * jj[None, :]
            ext = np.concatenate((w[:, ::-1], w, w[:, ::-1]), axis=1)  # (B, 3·lp)
            d3 = (
                ext[:, 0:l3]
                - 3.0 * ext[:, m : m + l3]
                + 3.0 * ext[:, 2 * m : 2 * m + l3]
                - ext[:, 3 * m : 3 * m + l3]
            )
            if n_avg > 0:
                cs = np.empty((d3.shape[0], l3 + 1), dtype=np.float64)
                cs[:, 0] = 0.0
                np.cumsum(d3, axis=1, out=cs[:, 1:])
                s = np.arange(n_avg)
                a = cs[:, s + m] - cs[:, s]
                total += np.einsum("ij,ij->i", a, a).sum() / (n_avg * 6.0 * m**2)
        devs[k] = np.sqrt(total / (nsubs * m**2 * tau0**2))
    return devs


# ──────────────────────────────────────────────────────────────────────
# Numba-accelerated scalar loops for the loop-bound kernels. Ported directly
# from the Julia compiled loops (kernels.jl); parallel over subsequences via
# prange. Selected at call time by the dispatchers below when Numba is present.

if _HAS_NUMBA:

    @njit(cache=True, parallel=True)
    def _mtotdev_njit(x, m_arr, tau0):  # noqa: ANN001
        n = x.size
        devs = np.empty(m_arr.size)
        for k in range(m_arr.size):
            m = m_arr[k]
            nsubs = n - 3 * m + 1
            if nsubs < 1:
                devs[k] = np.nan
                continue
            seg_len = 3 * m
            half_n = seg_len / 2.0
            hi_half = seg_len // 2
            lo_half_len = seg_len - hi_half
            mf = float(m)
            total = 0.0
            for start in prange(nsubs):
                if m == 1:
                    slope = (x[start + 2] - x[start]) / (2.0 * tau0)
                else:
                    s1 = 0.0
                    for i in range(hi_half):
                        s1 += x[start + i]
                    s2 = 0.0
                    for i in range(hi_half, seg_len):
                        s2 += x[start + i]
                    slope = (s2 / lo_half_len - s1 / hi_half) / (half_n * tau0)
                ext = np.empty(3 * seg_len)
                for j in range(seg_len):
                    wj = x[start + j] - slope * tau0 * j
                    ext[seg_len - 1 - j] = wj
                    ext[seg_len + j] = wj
                    ext[2 * seg_len + seg_len - 1 - j] = wj
                a1 = 0.0
                a2 = 0.0
                a3 = 0.0
                for i in range(m):
                    a1 += ext[i]
                    a2 += ext[i + m]
                    a3 += ext[i + 2 * m]
                d2 = (a3 - 2.0 * a2 + a1) / mf
                block = d2 * d2
                for s in range(1, 6 * m):
                    a1 += ext[s + m - 1] - ext[s - 1]
                    a2 += ext[s + 2 * m - 1] - ext[s + m - 1]
                    a3 += ext[s + 3 * m - 1] - ext[s + 2 * m - 1]
                    d2 = (a3 - 2.0 * a2 + a1) / mf
                    block += d2 * d2
                total += block / (6.0 * mf)
            devs[k] = np.sqrt(total / (2.0 * mf**2 * tau0**2 * nsubs))
        return devs

    @njit(cache=True, parallel=True)
    def _htotdev_njit(x, m_arr, tau0):  # noqa: ANN001
        n = x.size
        y = np.empty(n - 1)
        for i in range(n - 1):
            y[i] = (x[i + 1] - x[i]) / tau0
        ny = y.size
        devs = np.empty(m_arr.size)
        for k in range(m_arr.size):
            m = m_arr[k]
            mf = float(m)
            if m == 1:
                length = n - 3
                if length <= 0:
                    devs[k] = np.nan
                    continue
                ss = 0.0
                for i in range(length):
                    d3 = x[i + 3] - 3.0 * x[i + 2] + 3.0 * x[i + 1] - x[i]
                    ss += d3 * d3
                devs[k] = np.sqrt(ss / (6.0 * length * tau0**2))
                continue
            n_iter = ny - 3 * m + 1
            if n_iter < 1:
                devs[k] = np.nan
                continue
            seg_len = 3 * m
            hi = seg_len // 2
            lo_start = (seg_len + 1) // 2 + 1  # ceil(seg_len/2) + 1, 1-indexed
            lo_count = seg_len - lo_start + 1
            denom = (0.5 * (seg_len - 1) + 1.0) if seg_len % 2 == 1 else (0.5 * seg_len)
            mid = seg_len // 2
            total = 0.0
            for i0 in prange(n_iter):
                s1 = 0.0
                for j in range(hi):
                    s1 += y[i0 + j]
                m1 = s1 / hi
                s2 = 0.0
                for j in range(lo_start - 1, seg_len):
                    s2 += y[i0 + j]
                m2 = s2 / lo_count
                slope = (m2 - m1) / denom
                ext = np.empty(3 * seg_len)
                for j in range(seg_len):
                    wj = y[i0 + j] - slope * (j - mid)
                    ext[seg_len - 1 - j] = wj
                    ext[seg_len + j] = wj
                    ext[2 * seg_len + seg_len - 1 - j] = wj
                a1 = 0.0
                a2 = 0.0
                a3 = 0.0
                for i in range(m):
                    a1 += ext[i]
                    a2 += ext[i + m]
                    a3 += ext[i + 2 * m]
                d3 = (a3 - 2.0 * a2 + a1) / mf
                block = d3 * d3
                for s in range(1, 6 * m):
                    a1 += ext[s + m - 1] - ext[s - 1]
                    a2 += ext[s + 2 * m - 1] - ext[s + m - 1]
                    a3 += ext[s + 3 * m - 1] - ext[s + 2 * m - 1]
                    d3 = (a3 - 2.0 * a2 + a1) / mf
                    block += d3 * d3
                total += block / (6.0 * mf)
            devs[k] = np.sqrt(total / (6.0 * n_iter))
        return devs

    @njit(cache=True, parallel=True)
    def _mhtotdev_njit(x, m_arr, tau0):  # noqa: ANN001
        n = x.size
        devs = np.empty(m_arr.size)
        for k in range(m_arr.size):
            m = m_arr[k]
            mf = float(m)
            if m < 1:
                devs[k] = np.nan
                continue
            nsubs = n - 4 * m + 1
            if nsubs < 1:
                devs[k] = np.nan
                continue
            lp = 3 * m + 1
            l3 = 3 * lp - 3 * m
            half = lp // 2
            n_avg = l3 + 1 - m
            total = 0.0
            for start in prange(nsubs):
                s1 = 0.0
                for j in range(half):
                    s1 += x[start + j]
                s1 /= half
                s2 = 0.0
                for j in range(half, lp):
                    s2 += x[start + j]
                s2 /= lp - half
                slope = (s2 - s1) / ((lp / 2.0) * tau0)
                ext = np.empty(3 * lp)
                for j in range(lp):
                    wj = x[start + j] - slope * tau0 * j
                    ext[lp - 1 - j] = wj
                    ext[lp + j] = wj
                    ext[2 * lp + lp - 1 - j] = wj
                d3v = np.empty(l3)
                for j in range(l3):
                    d3v[j] = ext[j] - 3.0 * ext[j + m] + 3.0 * ext[j + 2 * m] - ext[j + 3 * m]
                if n_avg > 0:
                    acc = 0.0
                    for j in range(m):
                        acc += d3v[j]
                    block = acc * acc
                    for w_ in range(1, n_avg):
                        acc += d3v[w_ + m - 1] - d3v[w_ - 1]
                        block += acc * acc
                    total += block / (n_avg * 6.0 * mf**2)
            devs[k] = np.sqrt(total / (nsubs * mf**2 * tau0**2))
        return devs

    @njit(cache=True)
    def _pdev_njit(x, m_arr, tau0):  # noqa: ANN001
        n = x.size
        devs = np.empty(m_arr.size)
        for k in range(m_arr.size):
            m = m_arr[k]
            mf = float(m)
            if m < 1:
                devs[k] = np.nan
                continue
            if m == 1:
                length = n - 2
                if length < 2:
                    devs[k] = np.nan
                    continue
                ss = 0.0
                for i in range(length):
                    d2 = x[i + 2] - 2.0 * x[i + 1] + x[i]
                    ss += d2 * d2
                devs[k] = np.sqrt(ss / (2.0 * length * tau0**2))
                continue
            big_m = n - 2 * m
            if big_m < 2:
                devs[k] = np.nan
                continue
            half = (m - 1) / 2.0
            a = 0.0
            b = 0.0
            for kk in range(m):
                yv = x[kk] - x[kk + m]
                a += yv
                b += kk * yv
            refresh_every = 4096 if 4096 > m else m
            next_refresh = refresh_every
            msum = 0.0
            for i in range(1, big_m + 1):
                asum = half * a - b
                msum += asum * asum
                if i < big_m:
                    if i == next_refresh:
                        a = 0.0
                        b = 0.0
                        for kk in range(m):
                            yv = x[i + kk] - x[i + kk + m]
                            a += yv
                            b += kk * yv
                        next_refresh += refresh_every
                    else:
                        yold = x[i - 1] - x[i - 1 + m]
                        ynew = x[i - 1 + m] - x[i - 1 + 2 * m]
                        old_a = a
                        a += ynew - yold
                        b += (m - 1) * ynew - old_a + yold
            devs[k] = np.sqrt(72.0 * msum / (big_m * mf**6 * tau0**2))
        return devs


def _mtotdev_core(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
    """Modified Total deviation — Numba scalar loop when available, else batched NumPy."""
    if _HAS_NUMBA:
        return _mtotdev_njit(x, np.asarray(m_values, dtype=np.int64), tau0)
    return _mtotdev_batched(x, m_values, tau0)


def _htotdev_core(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
    """Hadamard Total deviation — Numba scalar loop when available, else batched NumPy."""
    if _HAS_NUMBA:
        return _htotdev_njit(x, np.asarray(m_values, dtype=np.int64), tau0)
    return _htotdev_batched(x, m_values, tau0)


def _mhtotdev_core(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
    """Modified Hadamard Total deviation — Numba scalar loop when available, else NumPy."""
    if _HAS_NUMBA:
        return _mhtotdev_njit(x, np.asarray(m_values, dtype=np.int64), tau0)
    return _mhtotdev_batched(x, m_values, tau0)


def _pdev_core(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
    """Parabolic deviation — Numba rolling recurrence when available, else scipy FFT."""
    if _HAS_NUMBA:
        return _pdev_njit(x, np.asarray(m_values, dtype=np.int64), tau0)
    return _pdev_batched(x, m_values, tau0)
