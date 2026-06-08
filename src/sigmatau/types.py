"""Shared data records and result types — mirrors ``types.jl``.

``PhaseData``/``FrequencyData`` are the input records; ``StabilityResult`` and
``StabilitySuite`` are the outputs. All are frozen dataclasses with NumPy array
fields, matching the Julia oracle's non-parametric ``Vector{Float64}`` design.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


def _as_samples(v: object, name: str) -> np.ndarray:
    """Promote a sample vector to a contiguous 1-D float64 array."""
    arr = np.ascontiguousarray(v, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name}: expected a 1-D sample vector, got {arr.ndim}-D")
    return arr


@dataclass(frozen=True)
class PhaseData:
    """Phase residuals ``x(t)`` sampled at uniform interval ``tau0`` (default 1.0 s)."""

    x: np.ndarray
    tau0: float = 1.0

    def __post_init__(self) -> None:
        x = _as_samples(self.x, "PhaseData")
        if not self.tau0 > 0:
            raise ValueError(f"PhaseData: tau0 must be positive, got {self.tau0}")
        if x.size < 2:
            raise ValueError(f"PhaseData: need at least 2 phase samples, got {x.size}")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "tau0", float(self.tau0))

    def __repr__(self) -> str:
        return f"PhaseData(N={self.x.size}, tau0={self.tau0} s)"


@dataclass(frozen=True)
class FrequencyData:
    """Fractional-frequency samples ``y(t)`` at uniform interval ``tau0`` (default 1.0 s)."""

    y: np.ndarray
    tau0: float = 1.0

    def __post_init__(self) -> None:
        y = _as_samples(self.y, "FrequencyData")
        if not self.tau0 > 0:
            raise ValueError(f"FrequencyData: tau0 must be positive, got {self.tau0}")
        if y.size < 2:
            raise ValueError(f"FrequencyData: need at least 2 frequency samples, got {y.size}")
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "tau0", float(self.tau0))

    def __repr__(self) -> str:
        return f"FrequencyData(N={self.y.size}, tau0={self.tau0} s)"


@dataclass(frozen=True)
class StabilityResult:
    """Unified return type for every stability calculation.

    ``noise_type``, ``ci_lower``, ``ci_upper``, and ``edf`` are empty arrays when
    the calculation was invoked with ``ci=False``.
    """

    deviation_type: str
    tau: np.ndarray
    dev: np.ndarray
    noise_type: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    edf: np.ndarray

    def __repr__(self) -> str:
        n = self.tau.size
        ci = "no CI" if self.edf.size == 0 else "with CI"
        if n == 0:
            return f"StabilityResult({self.deviation_type}, 0 pts, {ci})"
        return (
            f"StabilityResult({self.deviation_type}, {n} pts, "
            f"τ∈[{self.tau[0]}, {self.tau[-1]}] s, {ci})"
        )


@dataclass(frozen=True)
class StabilitySuite:
    """Ordered, symbol-indexable collection of ``StabilityResult``s + session metadata.

    Index by deviation name (``suite["adev"]``) or position (``suite[0]``).
    """

    results: tuple[StabilityResult, ...]
    data_kind: str
    tau0: float
    n: int
    confidence: float | None
    tau_mode: str

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[StabilityResult]:
        return iter(self.results)

    def keys(self) -> tuple[str, ...]:
        return tuple(r.deviation_type for r in self.results)

    def __contains__(self, key: str) -> bool:
        return any(r.deviation_type == key for r in self.results)

    def __getitem__(self, key: int | str) -> StabilityResult:
        if isinstance(key, str):
            for r in self.results:
                if r.deviation_type == key:
                    return r
            raise KeyError(key)
        return self.results[key]

    def __repr__(self) -> str:
        devs = ", ".join(self.keys())
        ci = "no CI" if self.confidence is None else f"CI@{self.confidence}"
        return (
            f"StabilitySuite({len(self.results)} devs [{devs}], "
            f"{self.data_kind}, N={self.n}, τ₀={self.tau0} s, {ci})"
        )


@dataclass(frozen=True)
class SpectralResult:
    """Unified return type for every spectral-density estimate (``Sy``/``Sx``/``L``).

    A flat, non-parametric record: the one-sided frequency grid, the estimated
    spectrum, and the Welch parameters that produced it.
    """

    spectral_type: str
    freq: np.ndarray
    psd: np.ndarray
    units: str
    nperseg: int
    noverlap: int
    window: str

    def __repr__(self) -> str:
        n = self.freq.size
        rng = "" if n == 0 else f", f∈[{self.freq[0]}, {self.freq[-1]}] Hz"
        return (
            f"SpectralResult({self.spectral_type}, {n} bins{rng}, "
            f"{self.units}, nperseg={self.nperseg})"
        )


__all__ = [
    "PhaseData",
    "FrequencyData",
    "StabilityResult",
    "StabilitySuite",
    "SpectralResult",
]
