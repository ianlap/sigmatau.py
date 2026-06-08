"""Raw deviation kernels — mirrors ``kernels.jl`` (MVP subset).

Internal ``_*_core`` functions: plain ``float64`` array in, ``float64`` array
out. NumPy-vectorized ports of the Julia kernels; the public PhaseData/
FrequencyData API lives in ``deviations.py``. Each overlapping difference is
NaN when the averaging factor leaves fewer than 2 analysis windows, matching the
Julia ``L``/``Ne`` guards exactly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import correlate as sig_correlate


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


def _pdev_core(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
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


def _time_reverse_ext(w: np.ndarray) -> np.ndarray:
    """Greenhall 3× time-reverse extension of a detrended window: [rev, w, rev].

    The reflected segment ``rev_val`` equals ``w`` reversed (the detrending slope
    makes the forward and reversed-index forms coincide), so the full extension is
    ``concatenate((w[::-1], w, w[::-1]))``.
    """
    wr = w[::-1]
    return np.concatenate((wr, w, wr))


def _total_sumsq(ext: np.ndarray, m: int) -> float:
    """Σ over the 6m sliding windows of ((a3 − 2a2 + a1)/m)², a1/a2/a3 being the
    length-m running sums of ``ext`` at offsets 0, m, 2m — the shared inner
    reduction of MTOTDEV and HTOTDEV."""
    cs = np.empty(ext.size + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(ext, out=cs[1:])
    s = np.arange(6 * m)
    a1 = cs[s + m] - cs[s]
    a2 = cs[s + 2 * m] - cs[s + m]
    a3 = cs[s + 3 * m] - cs[s + 2 * m]
    d = (a3 - 2.0 * a2 + a1) / m
    return float(np.dot(d, d))


def _mtotdev_core(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
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
        jj = np.arange(seg_len, dtype=np.float64)
        total = 0.0
        for start in range(nsubs):
            win = x[start : start + seg_len]
            if m == 1:
                slope = (x[start + 2] - x[start]) / (2.0 * tau0)
            else:
                s1 = win[:hi_half].sum()
                s2 = win[hi_half:].sum()
                slope = (s2 / lo_half_len - s1 / hi_half) / (half_n * tau0)
            w = win - slope * tau0 * jj
            total += _total_sumsq(_time_reverse_ext(w), m) / (6.0 * m)
        devs[k] = np.sqrt(total / (2.0 * m**2 * tau0**2 * nsubs))
    return devs


def _htotdev_core(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
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
        jj = np.arange(seg_len, dtype=np.float64)
        total = 0.0
        for i in range(n_iter):
            seg = y[i : i + seg_len]
            m1 = seg[:hi].sum() / hi
            m2 = seg[lo_start - 1 :].sum() / lo_count
            slope = (m2 - m1) / denom
            w = seg - slope * (jj - mid)
            total += _total_sumsq(_time_reverse_ext(w), m) / (6.0 * m)
        devs[k] = np.sqrt(total / (6.0 * n_iter))
    return devs


def _mhtotdev_core(x: np.ndarray, m_values: Sequence[int], tau0: float) -> np.ndarray:
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
        jj = np.arange(lp, dtype=np.float64)
        n_avg = l3 + 1 - m
        total = 0.0
        for start in range(nsubs):
            win = x[start : start + lp]
            s1 = win[:half].mean()
            s2 = win[half:].mean()
            slope = (s2 - s1) / ((lp / 2.0) * tau0)
            w = win - slope * tau0 * jj
            ext = _time_reverse_ext(w)
            d3 = (
                ext[0:l3]
                - 3.0 * ext[m : m + l3]
                + 3.0 * ext[2 * m : 2 * m + l3]
                - ext[3 * m : 3 * m + l3]
            )
            if n_avg > 0:
                cs = np.empty(l3 + 1, dtype=np.float64)
                cs[0] = 0.0
                np.cumsum(d3, out=cs[1:])
                s = np.arange(n_avg)
                a = cs[s + m] - cs[s]
                total += float(np.dot(a, a)) / (n_avg * 6.0 * m**2)
        devs[k] = np.sqrt(total / (nsubs * m**2 * tau0**2))
    return devs
