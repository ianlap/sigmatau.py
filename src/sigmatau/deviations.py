"""Public deviation API — mirrors ``deviations.jl`` (MVP subset).

Each function takes ``PhaseData`` or ``FrequencyData`` and returns a
``StabilityResult``. The frequency path delegates via ``_freq_to_phase``, exactly
like the Julia oracle. ``taus`` accepts a ``TauMode`` (default ``Octave``) or an
explicit sequence of integer averaging factors.

CI is not implemented in this milestone: ``ci`` defaults to ``False`` and
``ci=True`` raises ``NotImplementedError``. The default flips to ``True`` once the
EDF/CI cycle lands. ``confidence`` is accepted for forward-compatible signatures.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .grids import Octave, TauMode, _f64, _freq_to_phase, _resolve_m
from .kernels import _adev_core, _hdev_core, _mdev_core, _mhdev_core
from .types import FrequencyData, PhaseData, StabilityResult

_CI_MESSAGE = (
    "ci=True is not implemented in this milestone (no EDF/CI machinery yet); "
    "call with ci=False. CI lands in a later release."
)


def _as_phase(data: PhaseData | FrequencyData) -> PhaseData:
    return _freq_to_phase(data) if isinstance(data, FrequencyData) else data


def _no_ci_result(name: str, tau: np.ndarray, dev: np.ndarray) -> StabilityResult:
    """Build a ``StabilityResult`` with the empty-CI contract (``ci=False`` path)."""
    return StabilityResult(
        deviation_type=name,
        tau=np.asarray(tau, dtype=np.float64),
        dev=np.asarray(dev, dtype=np.float64),
        noise_type=np.empty(0, dtype=object),
        ci_lower=np.empty(0, dtype=np.float64),
        ci_upper=np.empty(0, dtype=np.float64),
        edf=np.empty(0, dtype=np.float64),
    )


def _simple(
    name: str,
    core: Callable[[np.ndarray, Sequence[int], float], np.ndarray],
    data: PhaseData | FrequencyData,
    taus: TauMode | Sequence[int],
    ci: bool,
) -> StabilityResult:
    """Shared path for the direct (non-derived) deviations: adev/mdev/hdev/mhdev."""
    if ci:
        raise NotImplementedError(_CI_MESSAGE)
    pd = _as_phase(data)
    m = _resolve_m(taus, pd.x.size, name)
    devs = core(_f64(pd.x), m, pd.tau0)
    tau = np.asarray(m, dtype=np.float64) * pd.tau0
    return _no_ci_result(name, tau, devs)


def adev(
    data: PhaseData | FrequencyData,
    taus: TauMode | Sequence[int] = Octave,
    *,
    ci: bool = False,
    confidence: float = 0.683,
) -> StabilityResult:
    """Overlapping Allan deviation σ_y(τ)."""
    return _simple("adev", _adev_core, data, taus, ci)


def mdev(
    data: PhaseData | FrequencyData,
    taus: TauMode | Sequence[int] = Octave,
    *,
    ci: bool = False,
    confidence: float = 0.683,
) -> StabilityResult:
    """Modified Allan deviation Mod σ_y(τ)."""
    return _simple("mdev", _mdev_core, data, taus, ci)


def hdev(
    data: PhaseData | FrequencyData,
    taus: TauMode | Sequence[int] = Octave,
    *,
    ci: bool = False,
    confidence: float = 0.683,
) -> StabilityResult:
    """Overlapping Hadamard deviation."""
    return _simple("hdev", _hdev_core, data, taus, ci)


def mhdev(
    data: PhaseData | FrequencyData,
    taus: TauMode | Sequence[int] = Octave,
    *,
    ci: bool = False,
    confidence: float = 0.683,
) -> StabilityResult:
    """Modified Hadamard deviation."""
    return _simple("mhdev", _mhdev_core, data, taus, ci)


def tdev(
    data: PhaseData | FrequencyData,
    taus: TauMode | Sequence[int] = Octave,
    *,
    ci: bool = False,
    confidence: float = 0.683,
) -> StabilityResult:
    """Time deviation σ_x(τ) = (τ/√3)·Mod σ_y(τ). Units of seconds."""
    if ci:
        raise NotImplementedError(_CI_MESSAGE)
    r = mdev(data, taus, ci=False)
    factor = r.tau / np.sqrt(3.0)
    return _no_ci_result("tdev", r.tau, r.dev * factor)


def htdev(
    data: PhaseData | FrequencyData,
    taus: TauMode | Sequence[int] = Octave,
    *,
    ci: bool = False,
    confidence: float = 0.683,
) -> StabilityResult:
    """Hadamard time deviation σ_x,HT(τ) = (τ/√(10/3))·Mod σ_y,MH(τ). Units of seconds."""
    if ci:
        raise NotImplementedError(_CI_MESSAGE)
    r = mhdev(data, taus, ci=False)
    factor = r.tau / np.sqrt(10.0 / 3.0)
    return _no_ci_result("htdev", r.tau, r.dev * factor)
