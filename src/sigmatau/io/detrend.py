"""Linear / endpoint / mean detrending — mirrors ``io/detrend.jl``."""

from __future__ import annotations

import numpy as np

from ..types import FrequencyData, PhaseData

_DETREND_MODES = ("none", "mean", "endpoint", "linear")


def _detrend_core(x: np.ndarray, mode: str) -> np.ndarray:
    """Return a detrended copy of ``x``. ``mode`` ∈ {none, mean, endpoint, linear}."""
    n = x.size
    if mode == "none" or n <= 1:
        return x.copy()
    if mode == "mean":
        return x - x.sum() / n
    if mode == "endpoint":
        slope = (x[-1] - x[0]) / (n - 1)
        idx = np.arange(n, dtype=np.float64)
        return x - (x[0] + slope * idx)
    if mode == "linear":
        # closed-form OLS for y = a + b·n on n = 0..N-1 (matches the oracle)
        sx = (n - 1) * n / 2.0
        sxx = (n - 1) * n * (2 * n - 1) / 6.0
        idx = np.arange(n, dtype=np.float64)
        sy = float(x.sum())
        sxy = float((idx * x).sum())
        denom = n * sxx - sx * sx
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        return x - (a + b * idx)
    raise ValueError(f"detrend: unknown method {mode} (expected one of {_DETREND_MODES})")


def detrend(
    data: PhaseData | FrequencyData, *, method: str = "linear"
) -> PhaseData | FrequencyData:
    """Return a new record with a detrended sample vector (the original is untouched).

    ``method`` ∈ {``linear`` (default), ``endpoint``, ``mean``, ``none``}.
    """
    if method not in _DETREND_MODES:
        raise ValueError(f"detrend: unknown method {method} (expected one of {_DETREND_MODES})")
    if isinstance(data, FrequencyData):
        return FrequencyData(_detrend_core(data.y, method), data.tau0)
    return PhaseData(_detrend_core(data.x, method), data.tau0)


__all__ = ["detrend"]
