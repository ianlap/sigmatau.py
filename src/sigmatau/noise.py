"""Power-law noise identification — mirrors the lag-1/B1 portion of ``noise.jl``.

``identify_noise`` returns a power-law noise-type symbol per averaging factor,
which the EDF / CI / bias machinery in ``edf.py`` consumes. Per τ it uses the
lag-1 autocorrelation method when the decimated record is long enough
(``N/m ≥ NEFF_RELIABLE``) and the B1-ratio / R(n) fallback otherwise; an
unreliable τ inherits the last reliable classification.

``identify_noise`` is NumPy only (``np.std``/``np.var`` use ``ddof=1`` to match
Julia's ``Statistics`` convention). This module also holds ``noise_gen``, the
calibrated power-law clock-noise generator.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .kernels import _adev_core
from .types import FrequencyData, PhaseData

NEFF_RELIABLE = 30
_EPS = float(np.finfo(np.float64).eps)

_ALPHA_TO_SYMBOL = {2: "WHPM", 1: "FLPM", 0: "WHFM", -1: "FLFM", -2: "RWFM"}


def identify_noise(
    x: np.ndarray,
    m_values: Sequence[int],
    *,
    dmin: int = 0,
    dmax: int = 2,
    detrend: bool = True,
) -> np.ndarray:
    """Identify the dominant power-law noise type at each averaging factor.

    Returns an object array of symbols (``"WHPM"/"FLPM"/"WHFM"/"FLFM"/"RWFM"`` or
    ``"unknown"``). ``detrend`` (default ``True``, matching allantools) quadratically
    detrends each decimated subseries before classification.
    """
    x_clean = _preprocess(np.ascontiguousarray(x, dtype=np.float64))
    n = x_clean.size
    noises = np.empty(len(m_values), dtype=object)
    last_reliable = "unknown"

    for k, m in enumerate(m_values):
        n_eff = n // m
        alpha = np.nan
        try:
            if n_eff >= NEFF_RELIABLE:
                alpha = _noise_id_lag1acf(x_clean, m, dmin, dmax, detrend=detrend)[0]
            else:
                alpha = _noise_id_b1rn(x_clean, m, detrend=detrend)[0]
        except Exception:
            alpha = np.nan

        if not np.isnan(alpha):
            ai = int(round(alpha))
            sym = _ALPHA_TO_SYMBOL.get(ai, "unknown")
            noises[k] = sym
            last_reliable = sym
        else:
            noises[k] = last_reliable

    return noises


def _preprocess(x: np.ndarray) -> np.ndarray:
    """5σ outlier filter on the full record (no polynomial detrend here)."""
    x_std = float(np.std(x, ddof=1))
    if x_std < _EPS:
        return x
    x_mean = float(np.mean(x))
    keep = np.abs(x - x_mean) < 5.0 * x_std
    if bool(keep.all()):
        return x
    return x[keep]


def _detrend_quadratic(x: np.ndarray) -> np.ndarray:
    """Remove a least-squares quadratic (offset + drift + curvature), 1-indexed."""
    n = x.size
    nf = float(n)
    fi = np.arange(1, n + 1, dtype=np.float64)  # 1-indexed sample positions
    x1 = float(x.sum())
    x2 = float((fi * x).sum())
    x3 = float((fi * fi * x).sum())
    s1 = nf
    s2 = nf * (nf + 1.0) / 2.0
    s3 = nf * (nf + 1.0) * (2.0 * nf + 1.0) / 6.0
    s4 = nf**2 * (nf + 1.0) ** 2 / 4.0
    s5 = nf * (nf + 1.0) * (2.0 * nf + 1.0) * (3.0 * nf**2 + 3.0 * nf - 1.0) / 30.0
    mat = np.array([[s1, s2, s3], [s2, s3, s4], [s3, s4, s5]], dtype=np.float64)
    a, b, c = np.linalg.solve(mat, np.array([x1, x2, x3], dtype=np.float64))
    return x - (a + b * fi + c * fi * fi)


def _lag1_acf(x: np.ndarray) -> float:
    """Lag-1 autocorrelation with a scale-invariant degeneracy guard."""
    n = x.size
    if n < 2:
        return np.nan
    mu = float(x.mean())
    centered = x - mu
    ssx = float(np.dot(centered, centered))
    raw = float(np.dot(x, x))
    if raw > 0 and ssx <= _EPS * raw:
        return np.nan
    if ssx == 0.0:
        return np.nan
    num = float(np.dot(centered[:-1], centered[1:]))
    return num / ssx


def _noise_id_lag1acf(
    x: np.ndarray, m: int, dmin: int = 0, dmax: int = 2, *, detrend: bool = True
) -> tuple[float, int, int, float]:
    """Lag-1 ACF noise ID: difference until ρ < 0.25 (or dmax), then α = 2 − 2(ρ+d)."""
    x_dec = x[::m] if m > 1 else x
    x_det = _detrend_quadratic(x_dec) if detrend else x_dec.copy()

    d = 0
    while True:
        r1 = _lag1_acf(x_det)
        rho = r1 / (1.0 + r1)
        if d >= dmin and (rho < 0.25 or d >= dmax):
            p = -2.0 * (rho + d)
            alpha = p + 2.0
            if np.isnan(alpha):
                return (np.nan, 0, d, rho)
            return (alpha, int(round(alpha)), d, rho)
        x_det = np.diff(x_det)
        d += 1
        if x_det.size < 5:
            raise ValueError("Data too short after differencing")


def _noise_id_b1rn(x: np.ndarray, m: int, *, detrend: bool = True) -> tuple[float, int, float]:
    """B1-ratio noise ID with R(n) WPM/FLPM disambiguation (short-record fallback)."""
    x_dec = x[::m]
    if detrend:
        x_dec = _detrend_quadratic(x_dec)

    avar_val = _simple_avar(x_dec, 1) / float(m) ** 2
    n_avar = x.size - 1  # frequency-sample count of the full run (Howe/Barnes)

    dx = np.diff(x)
    nd = (dx.size // m) * m
    if nd < m:
        return (np.nan, -2, np.nan)

    y_avg = dx[:nd].reshape(-1, m).mean(axis=1)
    var_class = float(np.var(y_avg, ddof=1))

    if np.isnan(avar_val) or avar_val <= 0:
        return (np.nan, -2, np.nan)

    b1_obs = var_class / avar_val

    mu_list = [1, 0, -1, -2]
    alpha_list = [-2, -1, 0, 2]
    b1_vals = [_b1_theory(n_avar, mu) for mu in mu_list]

    mu_best = mu_list[-1]
    alpha_int = alpha_list[-1]
    for i in range(len(mu_list) - 1):
        boundary = np.sqrt(b1_vals[i] * b1_vals[i + 1])
        if b1_obs > boundary:
            mu_best = mu_list[i]
            alpha_int = alpha_list[i]
            break

    if mu_best == -2:
        adev_val = np.sqrt(avar_val)
        mdev_val = _simple_mdev(x, m, 1.0)
        if not np.isnan(mdev_val) and adev_val > 0:
            rn_obs = (mdev_val / adev_val) ** 2
            r_hi = _rn_theory(m, 0)
            r_lo = _rn_theory(m, -1)
            alpha_int = 1 if rn_obs > np.sqrt(r_hi * r_lo) else 2

    return (float(alpha_int), mu_best, b1_obs)


def _b1_theory(n: int, mu: int) -> float:
    """Howe/Barnes closed-form B1 ratio for averaging factor 1 at exponent ``mu``."""
    nf = float(n)
    if mu == 2:
        return nf * (nf + 1.0) / 6.0
    if mu == 1:
        return nf / 2.0
    if mu == 0:
        return nf * np.log(nf) / (2.0 * (nf - 1.0) * np.log(2.0))
    if mu == -1:
        return 1.0
    if mu == -2:
        return (nf**2 - 1.0) / (1.5 * nf * (nf - 1.0))
    return (nf * (1.0 - nf**mu)) / (2.0 * (nf - 1.0) * (1.0 - 2.0**mu))


def _rn_theory(af: int, b: int) -> float:
    """MVAR/AVAR ratio R(n) for the WPM/FLPM boundary (Howe/Greenhall)."""
    if b == 0:
        return 1.0 / af
    if b == -1:
        avar = (1.038 + 3.0 * np.log(2.0 * np.pi * 0.5 * af)) / (4.0 * np.pi**2)
        mvar = 3.0 * np.log(256.0 / 27.0) / (8.0 * np.pi**2)
        return mvar / avar
    return 1.0


def _simple_avar(x: np.ndarray, m: int) -> float:
    """Overlapping AVAR value (no τ₀ scaling) — helper for the B1 ratio."""
    n = x.size
    length = n - 2 * m
    if length <= 0:
        return np.nan
    d2 = x[2 * m : n] - 2.0 * x[m : n - m] + x[0:length]
    return float(np.dot(d2, d2)) / (2.0 * float(m) ** 2 * length)


def _simple_mdev(x: np.ndarray, m: int, tau0: float) -> float:
    """Modified Allan deviation value — helper for the R(n) disambiguation."""
    n = x.size
    ne = n - 3 * m + 1
    if ne <= 0:
        return np.nan
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(x, out=cs[1:])
    s1 = cs[m : m + ne] - cs[0:ne]
    s2 = cs[2 * m : 2 * m + ne] - cs[m : m + ne]
    s3 = cs[3 * m : 3 * m + ne] - cs[2 * m : 2 * m + ne]
    d = (s3 - 2.0 * s2 + s1) / m
    return float(np.sqrt(np.dot(d, d) / (2.0 * float(m) ** 2 * tau0**2 * ne)))


# ──────────────────────────────────────────────────────────────────────
# Calibrated power-law clock-noise generation (mirrors noise.jl synth + gen)
#
# Realizations cannot match the Julia oracle bit-for-bit (NumPy and Julia use
# different RNGs), so this is validated by its properties — calibration hits the
# target σ exactly, the h→σ analytic identities hold, independent components sum
# in quadrature — rather than against golden fixtures.


def _gen_powerlaw_y(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise shaped to a fractional-frequency PSD ∝ f^alpha.

    DC is zeroed (zero-mean output); the absolute level is whatever the f^(α/2)
    shaper produces on unit-variance white noise — callers rescale to calibrate.
    """
    w = rng.standard_normal(n)
    spec = np.fft.fft(w)
    f = np.abs(np.fft.fftfreq(n, 1.0))
    f[0] = 1.0  # placeholder so f^(α/2) is finite at DC; the bin is zeroed next
    shaped = spec * f ** (alpha / 2.0)
    shaped[0] = 0.0
    return np.fft.ifft(shaped).real


def _h_to_sigma1(alpha: int, h: float, tau0: float) -> float:
    """σ_y(τ=τ₀) from the PSD coefficient h_α (SP1065 Table 3, Nyquist f_h=1/2τ₀)."""
    if alpha == 2:
        return math.sqrt(3.0 * h / (8.0 * math.pi**2 * tau0**3))
    if alpha == 1:
        coef = 1.038 + 3.0 * math.log(math.pi)
        return math.sqrt(coef * h / (4.0 * math.pi**2 * tau0**2))
    if alpha == 0:
        return math.sqrt(h / (2.0 * tau0))
    if alpha == -1:
        return math.sqrt(2.0 * math.log(2.0) * h)
    if alpha == -2:
        return math.sqrt(2.0 * math.pi**2 * h * tau0 / 3.0)
    raise ValueError(f"noise_gen: unsupported α = {alpha}; expected ∈ {{-2,-1,0,1,2}}")


def _measure_sigma1(y: np.ndarray, tau0: float) -> float:
    """Empirical σ_y(τ=τ₀) of a frequency vector via overlapping ADEV at m=1."""
    x = np.cumsum(y) * tau0
    return float(_adev_core(x, [1], tau0)[0])


def _noise_gen_y(
    n: int,
    tau0: float,
    sigma1: Mapping[int, float],
    h: Mapping[int, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Composite fractional-frequency vector: independent per-α components, summed."""
    if sigma1 and h:
        raise ValueError("noise_gen: pass either `sigma1` or `h`, not both")
    if not sigma1 and not h:
        raise ValueError("noise_gen: noise mixture is empty; pass `sigma1` or `h`")
    if n < 4:
        raise ValueError(f"noise_gen: N must be ≥ 4 (got {n})")
    if not tau0 > 0:
        raise ValueError(f"noise_gen: tau0 must be > 0 (got {tau0})")

    targets: dict[int, float] = {}
    source = sigma1 if sigma1 else h
    for a, val in source.items():
        if not isinstance(a, int) or isinstance(a, bool):
            raise ValueError(f"noise_gen: α keys must be int, got {type(a)}")
        if not -2 <= a <= 2:
            raise ValueError(f"noise_gen: α must be ∈ {{-2,…,2}}, got {a}")
        if val < 0:
            raise ValueError(f"noise_gen: values must be ≥ 0, got {val} for α={a}")
        targets[a] = float(val) if sigma1 else _h_to_sigma1(a, float(val), tau0)

    y_total = np.zeros(n, dtype=np.float64)
    for a, sigma_target in targets.items():
        if sigma_target == 0:
            continue
        y_raw = _gen_powerlaw_y(a, n, rng)
        sigma_raw = _measure_sigma1(y_raw, tau0)
        if sigma_raw > 0:
            y_total += y_raw * (sigma_target / sigma_raw)
    return y_total


def noise_gen(
    kind: type,
    n: int,
    tau0: float = 1.0,
    *,
    sigma1: Mapping[int, float] | None = None,
    h: Mapping[int, float] | None = None,
    rng: int | np.random.Generator | None = None,
) -> PhaseData | FrequencyData:
    """Synthesize a length-``n`` clock record with a power-law frequency spectrum.

    ``kind`` is ``PhaseData`` (integrated to phase) or ``FrequencyData`` (raw ``y``).
    Specify the mixture by ``sigma1[α] = σ_y(τ₀)`` *or* ``h[α] = h_α`` (not both),
    with α ∈ {−2,−1,0,1,2}. Each component is rescaled to hit its target σ exactly
    for the drawn realization; components of different α are independent.

    ``rng`` accepts an int seed, a ``numpy.random.Generator``, or ``None`` (fresh).
    """
    generator = np.random.default_rng(rng)
    y = _noise_gen_y(n, float(tau0), sigma1 or {}, h or {}, generator)
    if kind is PhaseData:
        return PhaseData(np.cumsum(y) * float(tau0), float(tau0))
    if kind is FrequencyData:
        return FrequencyData(y, float(tau0))
    raise ValueError("noise_gen: kind must be PhaseData or FrequencyData")


__all__ = ["identify_noise", "noise_gen", "NEFF_RELIABLE"]
