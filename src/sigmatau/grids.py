"""Averaging-factor grids and conversion helpers — mirrors ``grids.jl``.

``TauMode`` selects the averaging-factor spacing; ``tau_values`` resolves it to a
concrete integer grid bounded by each kernel's algorithmic m-max. Also holds the
``float64`` promotion and frequency→phase conversion used at the API boundary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum

import numpy as np

from .types import FrequencyData, PhaseData


class TauMode(Enum):
    """Averaging-factor grid spacing. Mirrors the Julia ``@enum TauMode``."""

    AllTaus = "AllTaus"
    Octave = "Octave"
    HalfOctave = "HalfOctave"
    QuarterOctave = "QuarterOctave"
    Decade = "Decade"
    HalfDecade = "HalfDecade"


# Module-level singletons so callers write `adev(pd, Octave)`, matching Julia.
AllTaus = TauMode.AllTaus
Octave = TauMode.Octave
HalfOctave = TauMode.HalfOctave
QuarterOctave = TauMode.QuarterOctave
Decade = TauMode.Decade
HalfDecade = TauMode.HalfDecade


def _kernel_m_max(n: int, kernel: str) -> int:
    """Largest averaging factor ``m`` for which ``kernel`` retains ≥2 windows.

    Mirrors ``_kernel_m_max`` in ``grids.jl`` (MVP subset of kernels).
    """
    if kernel in ("adev", "pdev"):
        m_max = (n - 2) // 2
    elif kernel == "totdev":
        m_max = (n - 1) // 2
    elif kernel in ("mdev", "tdev"):
        m_max = (n - 1) // 3
    elif kernel in ("mtotdev", "ttotdev"):
        m_max = n // 3
    elif kernel == "hdev":
        m_max = (n - 2) // 3
    elif kernel == "htotdev":
        m_max = (n - 1) // 3
    elif kernel in ("mhdev", "htdev"):
        m_max = (n - 1) // 4
    elif kernel == "mhtotdev":
        m_max = n // 4
    elif kernel == "mtie":
        m_max = n - 1
    else:
        raise ValueError(f"tau_values: unknown kernel symbol :{kernel}")
    if m_max < 1:
        raise ValueError(f"tau_values: N={n} is too short to support any m for :{kernel}")
    return m_max


def _grid(mode: TauMode, m_max: int) -> list[int]:
    """Averaging-factor grid for ``mode``, clamped to ``[1, m_max]``.

    ``Octave`` uses the exact integer formula so the default grid is byte-identical
    to the Julia oracle's; the other geometric modes round factor powers, dedupe,
    and clamp.
    """
    if mode is TauMode.AllTaus:
        return list(range(1, m_max + 1))
    if mode is TauMode.Octave:
        return [2**k for k in range(int(math.floor(math.log2(m_max))) + 1)]
    factor = {
        TauMode.HalfOctave: math.sqrt(2.0),
        TauMode.QuarterOctave: 2.0 ** (1 / 4),
        TauMode.Decade: 10.0,
        TauMode.HalfDecade: math.sqrt(10.0),
    }[mode]
    ms: list[int] = []
    k = 0
    while True:
        m = max(1, _round_half_even(factor**k))
        if m > m_max:
            break
        ms.append(m)
        k += 1
    # Deduplicate preserving order (matches Julia's `unique`).
    seen: set[int] = set()
    out: list[int] = []
    for m in ms:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _round_half_even(x: float) -> int:
    """Round to nearest integer, ties to even — matches Julia ``round(Int, x)``."""
    return int(np.rint(x))


def tau_values(mode: TauMode, n: int, kernel: str) -> list[int]:
    """Averaging factors ``m`` (τ = m·τ₀) for ``mode``, bounded by ``kernel``'s m-max."""
    return _grid(mode, _kernel_m_max(n, kernel))


def _default_m_values(n: int, kernel: str) -> list[int]:
    """Octave-spaced default grid bounded by ``kernel``'s algorithmic m-max."""
    return _grid(TauMode.Octave, _kernel_m_max(n, kernel))


def _f64(v: object) -> np.ndarray:
    """Promote a sample vector to a contiguous ``float64`` array (the kernel boundary)."""
    return np.ascontiguousarray(v, dtype=np.float64)


def _freq_to_phase(data: FrequencyData) -> PhaseData:
    """Convert fractional frequency to phase via ``x[k] = τ₀·Σⱼ y[j]`` (length preserved)."""
    return PhaseData(np.cumsum(data.y) * data.tau0, data.tau0)


def _phase_to_freq(data: PhaseData) -> FrequencyData:
    """Convert phase to fractional frequency via ``y[k] = (x[k+1]−x[k])/τ₀`` (N → N−1)."""
    return FrequencyData(np.diff(data.x) / data.tau0, data.tau0)


def _resolve_m(taus: TauMode | Sequence[int], n: int, kernel: str) -> list[int]:
    """Resolve the ``taus`` argument to an explicit list of integer averaging factors."""
    if isinstance(taus, TauMode):
        return tau_values(taus, n, kernel)
    return [int(m) for m in taus]


__all__ = [
    "TauMode",
    "AllTaus",
    "Octave",
    "HalfOctave",
    "QuarterOctave",
    "Decade",
    "HalfDecade",
    "tau_values",
]
