"""Welch PSD core and the public Sy / Sx / L estimators — mirrors ``spectral.jl``.

One-sided "density" normalization (matching ``scipy.signal.welch`` with
``scaling="density"``) so that ``Σ psd·Δf ≈ var`` over the analysis band. NumPy
only (``np.fft.rfft``). Reference: IEEE Std 1139-2022 §3.3–3.5.
"""

from __future__ import annotations

import numpy as np

from .grids import _f64, _freq_to_phase, _phase_to_freq
from .types import FrequencyData, PhaseData, SpectralResult


def _window(kind: str, n: int) -> np.ndarray:
    """Length-``n`` periodic (DFT-even) analysis window."""
    if kind == "rectangular":
        return np.ones(n, dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    if kind == "hann":
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)
    if kind == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * k / n)
    raise ValueError(f"spectral: unknown window {kind}; expected hann, hamming, or rectangular")


def _welch_psd(
    z: np.ndarray, fs: float, *, nperseg: int, noverlap: int, window: str
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD of real signal ``z`` sampled at ``fs`` Hz."""
    n = z.size
    if nperseg < 2:
        raise ValueError(f"spectral: nperseg must be ≥ 2 (got {nperseg})")
    if nperseg > n:
        raise ValueError(f"spectral: nperseg={nperseg} exceeds signal length {n}")
    if not 0 <= noverlap < nperseg:
        raise ValueError(
            f"spectral: noverlap must be in [0, nperseg) (got {noverlap}, nperseg={nperseg})"
        )
    if not fs > 0:
        raise ValueError(f"spectral: fs must be > 0 (got {fs})")

    win = _window(window, nperseg)
    u = float(np.dot(win, win))  # window power
    step = nperseg - noverlap
    nseg = (n - noverlap) // step
    if nseg < 1:
        raise ValueError("spectral: no full segment fits; reduce nperseg or noverlap")

    nfreq = nperseg // 2 + 1
    scale = 1.0 / (fs * u)
    psd = np.zeros(nfreq, dtype=np.float64)
    for s in range(nseg):
        off = s * step
        seg = z[off : off + nperseg]
        seg = (seg - seg.mean()) * win
        spec = np.fft.rfft(seg)
        psd += (spec.real**2 + spec.imag**2) * scale
    psd /= nseg

    # Fold to one-sided: double all but DC and (for even nperseg) Nyquist.
    if nperseg % 2 == 0:
        psd[1 : nfreq - 1] *= 2.0
    else:
        psd[1:] *= 2.0

    freq = np.arange(nfreq, dtype=np.float64) * fs / nperseg
    return freq, psd


def _default_nperseg(n: int) -> int:
    return min(n, 256)


def _welch_params(n: int, nperseg: int | None, noverlap: int | None) -> tuple[int, int]:
    np_ = _default_nperseg(n) if nperseg is None else int(nperseg)
    no = np_ // 2 if noverlap is None else int(noverlap)
    return np_, no


def Sy(
    data: PhaseData | FrequencyData,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
) -> SpectralResult:
    """One-sided PSD of fractional frequency S_y(f) [1/Hz] (Welch, fs=1/τ₀)."""
    if isinstance(data, PhaseData):
        data = _phase_to_freq(data)
    y = _f64(data.y)
    np_, no = _welch_params(y.size, nperseg, noverlap)
    freq, psd = _welch_psd(y, 1.0 / data.tau0, nperseg=np_, noverlap=no, window=window)
    return SpectralResult("Sy", freq, psd, "per_Hz", np_, no, window)


def Sx(
    data: PhaseData | FrequencyData,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
) -> SpectralResult:
    """One-sided PSD of the phase residual S_x(f) [s²/Hz] (Welch, fs=1/τ₀)."""
    if isinstance(data, FrequencyData):
        data = _freq_to_phase(data)
    x = _f64(data.x)
    np_, no = _welch_params(x.size, nperseg, noverlap)
    freq, psd = _welch_psd(x, 1.0 / data.tau0, nperseg=np_, noverlap=no, window=window)
    return SpectralResult("Sx", freq, psd, "s2_per_Hz", np_, no, window)


def L(
    data: PhaseData | FrequencyData,
    *,
    f_carrier: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
) -> SpectralResult:
    """Single-sideband phase noise ℒ(f) [dBc/Hz] at carrier ``f_carrier`` (IEEE 1139)."""
    if not f_carrier > 0:
        raise ValueError(f"L: f_carrier must be > 0 Hz (got {f_carrier})")
    sx = Sx(data, nperseg=nperseg, noverlap=noverlap, window=window)
    # Drop DC; S_x → S_φ → ℒ(f) → dBc/Hz.
    freq = sx.freq[1:]
    sphi = (2.0 * np.pi * f_carrier) ** 2 * sx.psd[1:]
    script_l = 10.0 * np.log10(0.5 * sphi)
    return SpectralResult("L", freq, script_l, "dBc_per_Hz", sx.nperseg, sx.noverlap, window)


__all__ = ["Sy", "Sx", "L"]
