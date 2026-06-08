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
