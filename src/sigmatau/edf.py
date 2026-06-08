"""Equivalent degrees of freedom, bias correction, and confidence intervals.

Port of ``edf.jl``. EDF uses the Greenhall–Riley spectral integrals for the
overlapped families and published coefficient tables for the total family;
``confidence_intervals`` uses ``scipy.stats`` for the χ²/normal quantiles where
Julia uses ``Distributions.jl``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import chi2, norm

_EPS = float(np.finfo(np.float64).eps)


def _alpha_from_noise(noise: str) -> int:
    return {"WHPM": 2, "FLPM": 1, "WHFM": 0, "FLFM": -1, "RWFM": -2}.get(noise, 0)


def _compute_sw(t: float, alpha: int) -> float:
    ta = abs(t)
    if alpha == 2:
        return -ta
    if alpha == 1:
        return t * t * math.log(max(ta, _EPS))
    if alpha == 0:
        return ta * ta * ta
    if alpha == -1:
        return -(t * t) * (t * t) * math.log(max(ta, _EPS))
    if alpha == -2:
        return -(ta**5)
    if alpha == -3:
        return (t * t) * (t * t) * (t * t) * math.log(max(ta, _EPS))
    if alpha == -4:
        return (ta**4) * (ta**3)
    return math.nan


def _compute_sx(t: float, f: int, alpha: int, inv_f: float) -> float:
    if f > 100 and alpha <= 0:
        return _compute_sw(t, alpha + 2)
    return float(f * f) * (
        2.0 * _compute_sw(t, alpha) - _compute_sw(t - inv_f, alpha) - _compute_sw(t + inv_f, alpha)
    )


def _compute_sz(t: float, f: int, alpha: int, d: int, inv_f: float) -> float:
    if d == 1:
        return (
            2.0 * _compute_sx(t, f, alpha, inv_f)
            - _compute_sx(t - 1.0, f, alpha, inv_f)
            - _compute_sx(t + 1.0, f, alpha, inv_f)
        )
    if d == 2:
        return (
            6.0 * _compute_sx(t, f, alpha, inv_f)
            - 4.0 * _compute_sx(t - 1.0, f, alpha, inv_f)
            - 4.0 * _compute_sx(t + 1.0, f, alpha, inv_f)
            + _compute_sx(t - 2.0, f, alpha, inv_f)
            + _compute_sx(t + 2.0, f, alpha, inv_f)
        )
    if d == 3:
        return (
            20.0 * _compute_sx(t, f, alpha, inv_f)
            - 15.0 * _compute_sx(t - 1.0, f, alpha, inv_f)
            - 15.0 * _compute_sx(t + 1.0, f, alpha, inv_f)
            + 6.0 * _compute_sx(t - 2.0, f, alpha, inv_f)
            + 6.0 * _compute_sx(t + 2.0, f, alpha, inv_f)
            - _compute_sx(t - 3.0, f, alpha, inv_f)
            - _compute_sx(t + 3.0, f, alpha, inv_f)
        )
    return math.nan


def _calc_edf_core(alpha: int, d: int, m: int, f: int, s: int, n: int) -> float:
    """Greenhall–Riley equivalent degrees of freedom."""
    if alpha + 2 * d <= 1:
        return math.nan
    length = m / f + m * d
    if n < length:
        return math.nan
    big_m = 1 + math.floor(s * (n - length) / m)
    big_j = min(big_m, (d + 1) * s)
    inv_f = 1.0 / f
    inv_s = 1.0 / s
    inv_m = 1.0 / big_m

    sz0 = _compute_sz(0.0, f, alpha, d, inv_f)
    bsum = sz0 * sz0
    for j in range(1, big_j):
        szj = _compute_sz(j * inv_s, f, alpha, d, inv_f)
        bsum += 2.0 * (1.0 - j * inv_m) * szj * szj
    if big_j <= big_m:
        szj = _compute_sz(big_j * inv_s, f, alpha, d, inv_f)
        bsum += (1.0 - big_j * inv_m) * szj * szj

    if not bsum > 0:
        return math.nan
    return big_m * sz0 * sz0 / bsum


def _coeff_totvar(alpha: int) -> tuple[float, float]:
    return {0: (1.50, 0.00), -1: (1.17, 0.22), -2: (0.93, 0.36)}.get(alpha, (math.nan, math.nan))


def _coeff_mtot(alpha: int) -> tuple[float, float]:
    return {
        2: (1.90, 2.10),
        1: (1.20, 1.40),
        0: (1.10, 1.20),
        -1: (0.85, 0.50),
        -2: (0.75, 0.31),
    }.get(alpha, (math.nan, math.nan))


def _coeff_mhtot(alpha: int) -> tuple[float, float]:
    return {
        2: (1.8534, 5.4817),
        1: (1.2185, 3.6691),
        0: (1.0998, 3.5040),
        -1: (1.0300, 3.3868),
        -2: (0.8132, 2.5406),
    }.get(alpha, (math.nan, math.nan))


def _coeff_htot(alpha: int) -> tuple[float, float]:
    return {
        0: (0.559, 1.004),
        -1: (0.868, 1.140),
        -2: (0.938, 1.696),
        -3: (0.974, 2.554),
        -4: (1.276, 3.149),
    }.get(alpha, (math.nan, math.nan))


def _pvar_a(alpha: int) -> float:
    return {2: 23.0, 1: 27.0, 0: 27.0, -1: 28.0, -2: 34.0}.get(alpha, math.nan)


def _pvar_edf(alpha: int, m: int, n: int) -> float:
    """PVAR equivalent degrees of freedom (Vernotte–Chen–Rubiola 2020)."""
    if m <= 0:
        return math.nan
    if m == 1:
        return _calc_edf_core(alpha, 2, 1, 1, 1, n)

    a = _pvar_a(alpha)
    if math.isnan(a):
        return math.nan

    def edf16(mm: int) -> float:
        big_m = n - 2 * mm
        if big_m < 1:
            return math.nan
        r = mm / big_m
        denom = a * r - 12.0 * r * r
        return 35.0 / denom if denom > 0 else math.nan

    m1 = round(1.11 * n / 4)
    if m <= m1:
        return edf16(m)

    m2 = round(0.901 * n / 2)
    if m1 < 2 or m2 <= m1:
        return edf16(m)
    nu_m1 = edf16(m1)
    if not math.isfinite(nu_m1):
        return math.nan
    slope = (nu_m1 - 1.0) / (math.log(m1) - math.log(m2))
    intercept = 1.0 - slope * math.log(m2)
    return max(slope * math.log(m) + intercept, 1.0)


def _kn_from_alpha(alpha: int) -> float:
    return {-2: 0.75, -1: 0.77, 0: 0.87, 1: 0.99, 2: 0.99}.get(alpha, 1.10)


def calculate_edf(
    method: str,
    devs: np.ndarray,
    noises: np.ndarray,
    m_values: Sequence[int],
    taus: np.ndarray,
    n: int,
    t: float,
) -> np.ndarray:
    """Equivalent degrees of freedom per τ for the given deviation ``method``."""
    edfs = np.empty(len(devs), dtype=np.float64)
    for k in range(len(devs)):
        m = m_values[k]
        alpha = _alpha_from_noise(noises[k])
        tau = taus[k]

        if method == "adev":
            edfs[k] = _calc_edf_core(alpha, 2, m, m, m, n)
        elif method == "mdev":
            edfs[k] = _calc_edf_core(alpha, 2, m, 1, m, n)
        elif method == "hdev":
            edfs[k] = _calc_edf_core(alpha, 3, m, m, m, n)
        elif method == "mhdev":
            edfs[k] = _calc_edf_core(alpha, 3, m, 1, m, n)
        elif method == "totdev":
            if alpha in (2, 1):
                edfs[k] = _calc_edf_core(alpha, 2, m, m, m, n)
            else:
                b, c = _coeff_totvar(alpha)
                edfs[k] = b * (t / tau) - c
        elif method == "mtotdev":
            if m < 16:
                edfs[k] = _calc_edf_core(alpha, 2, m, 1, m, n)
            else:
                b, c = _coeff_mtot(alpha)
                edfs[k] = b * (t / tau) - c
        elif method == "htotdev":
            if alpha in (2, 1):
                edfs[k] = _calc_edf_core(alpha, 3, m, m, m, n)
            else:
                b0, b1 = _coeff_htot(alpha)
                edfs[k] = (t / tau) / (b0 + b1 * (tau / t))
        elif method == "mhtotdev":
            b, c = _coeff_mhtot(alpha)
            edfs[k] = b * (t / tau) - c
        elif method == "pdev":
            edfs[k] = _pvar_edf(alpha, m, n)
        else:
            edfs[k] = math.nan
    return edfs


def bias_correction(noises: np.ndarray, var_type: str, taus: np.ndarray, t: float) -> np.ndarray:
    """Variance-scale bias factor B(α) = E[estimator²]/true_variance (apply as σ/√B)."""
    b = np.ones(len(noises), dtype=np.float64)
    for k in range(len(noises)):
        alpha = _alpha_from_noise(noises[k])
        tau = taus[k]
        if var_type == "totvar":
            a = (1.0 / (3.0 * math.log(2.0))) if alpha == -1 else (0.75 if alpha == -2 else 0.0)
            b[k] = 1.0 - a * (tau / t)
        elif var_type == "mtot":
            table = {2: 1.06, 1: 1.17, 0: 1.27, -1: 1.30, -2: 1.31}
            b[k] = table.get(_clamp(alpha, -2, 2), 1.0)
        elif var_type == "htot":
            a_table = {0: -0.005, -1: -0.149, -2: -0.229, -3: -0.283, -4: -0.321}
            b[k] = 1.0 + a_table[alpha] if -4 <= alpha <= 0 else 1.0
        elif var_type == "mhtot":
            b0_t = {2: 1.0644, 1: 0.9837, 0: 1.0193, -1: 1.2131, -2: 1.9432}
            b1_t = {2: 0.0165, 1: 0.0362, 0: -0.0477, -1: -0.3211, -2: -3.5880}
            aa = _clamp(alpha, -2, 2)
            b[k] = b0_t.get(aa, 1.0) + b1_t.get(aa, 0.0) * (tau / t)
    return b


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def confidence_intervals(
    devs: np.ndarray, edfs: np.ndarray, noises: np.ndarray, n: int, confidence: float
) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper confidence limits — χ² when edf ≥ 1, Gaussian fallback otherwise."""
    lower = np.empty(len(devs), dtype=np.float64)
    upper = np.empty(len(devs), dtype=np.float64)
    a_half = (1.0 - confidence) / 2.0
    z = float(norm.ppf(1.0 - a_half))

    for k in range(len(devs)):
        d = devs[k]
        if np.isnan(d):
            lower[k] = np.nan
            upper[k] = np.nan
            continue
        ef = edfs[k]
        if np.isfinite(ef) and ef >= 1.0:
            chi_lo = float(chi2.ppf(a_half, ef))
            chi_hi = float(chi2.ppf(1.0 - a_half, ef))
            lower[k] = d * math.sqrt(ef / chi_hi)
            upper[k] = d * math.sqrt(ef / chi_lo)
        else:
            kn = _kn_from_alpha(_alpha_from_noise(noises[k]))
            half = kn * d * z / math.sqrt(n)
            lower[k] = max(d - half, 0.0)
            upper[k] = d + half

    return lower, upper


__all__ = ["calculate_edf", "bias_correction", "confidence_intervals"]
