"""Public deviation API — mirrors ``deviations.jl``.

Each function takes ``PhaseData`` or ``FrequencyData`` and returns a
``StabilityResult``. The frequency path delegates via ``_freq_to_phase``. ``taus``
accepts a ``TauMode`` (default ``Octave``) or an explicit sequence of integer
averaging factors.

With the stats layer in place the defaults match the Julia oracle: ``ci=True``
populates per-τ noise type, equivalent degrees of freedom, and χ²-based
confidence bounds; the total family applies bias correction by default
(``correct_bias=True``). ``mtie`` has no CI model, so its ``ci``/``confidence`` are
accepted but are no-ops.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .edf import bias_correction, calculate_edf, confidence_intervals
from .grids import Octave, TauMode, _f64, _freq_to_phase, _resolve_m
from .kernels import (
    _adev_core,
    _hdev_core,
    _htotdev_core,
    _mdev_core,
    _mhdev_core,
    _mhtotdev_core,
    _mtie_core,
    _mtotdev_core,
    _pdev_core,
    _totdev_core,
)
from .noise import identify_noise
from .types import FrequencyData, PhaseData, StabilityResult

_Core = Callable[[np.ndarray, Sequence[int], float], np.ndarray]
_DEFAULT_CONFIDENCE = 0.683


def _as_phase(data: PhaseData | FrequencyData) -> PhaseData:
    return _freq_to_phase(data) if isinstance(data, FrequencyData) else data


def _empty_f() -> np.ndarray:
    return np.empty(0, dtype=np.float64)


def _no_ci_result(name: str, tau: np.ndarray, dev: np.ndarray) -> StabilityResult:
    """A result with the empty-CI contract (no noise type, CI, or EDF)."""
    return StabilityResult(
        deviation_type=name,
        tau=np.asarray(tau, dtype=np.float64),
        dev=np.asarray(dev, dtype=np.float64),
        noise_type=np.empty(0, dtype=object),
        ci_lower=_empty_f(),
        ci_upper=_empty_f(),
        edf=_empty_f(),
    )


def _unbias_divisor(b: np.ndarray) -> np.ndarray:
    """√B with non-positive B mapped to NaN (only :totvar can go non-positive)."""
    out = np.full(b.shape, np.nan, dtype=np.float64)
    pos = b > 0
    out[pos] = np.sqrt(b[pos])
    return out


def _ci_result(
    name: str,
    x: np.ndarray,
    m: Sequence[int],
    tau: np.ndarray,
    dev: np.ndarray,
    tau0: float,
    dmax: int,
    confidence: float,
    noises: np.ndarray | None = None,
) -> StabilityResult:
    """Build a full result: identify noise (unless supplied), then EDF and CIs."""
    n = x.size
    t = (n - 1) * tau0
    if noises is None:
        noises = identify_noise(x, m, dmin=0, dmax=dmax)
    edfs = calculate_edf(name, dev, noises, m, tau, n, t)
    lower, upper = confidence_intervals(dev, edfs, noises, n, confidence)
    return StabilityResult(
        name,
        np.asarray(tau, dtype=np.float64),
        np.asarray(dev, dtype=np.float64),
        noises,
        lower,
        upper,
        edfs,
    )


def _direct(
    name: str,
    core: _Core,
    dmax: int,
    data: PhaseData | FrequencyData,
    taus: TauMode | Sequence[int],
    ci: bool,
    confidence: float,
) -> StabilityResult:
    """Shared path for the non-derived, non-bias deviations: adev/mdev/hdev/mhdev/pdev."""
    pd = _as_phase(data)
    m = _resolve_m(taus, pd.x.size, name)
    x = _f64(pd.x)
    dev = core(x, m, pd.tau0)
    tau = np.asarray(m, dtype=np.float64) * pd.tau0
    if not ci:
        return _no_ci_result(name, tau, dev)
    return _ci_result(name, x, m, tau, dev, pd.tau0, dmax, confidence)


def adev(data, taus=Octave, *, ci=True, confidence=_DEFAULT_CONFIDENCE):
    """Overlapping Allan deviation σ_y(τ)."""
    return _direct("adev", _adev_core, 2, data, taus, ci, confidence)


def mdev(data, taus=Octave, *, ci=True, confidence=_DEFAULT_CONFIDENCE):
    """Modified Allan deviation Mod σ_y(τ)."""
    return _direct("mdev", _mdev_core, 2, data, taus, ci, confidence)


def hdev(data, taus=Octave, *, ci=True, confidence=_DEFAULT_CONFIDENCE):
    """Overlapping Hadamard deviation."""
    return _direct("hdev", _hdev_core, 3, data, taus, ci, confidence)


def mhdev(data, taus=Octave, *, ci=True, confidence=_DEFAULT_CONFIDENCE):
    """Modified Hadamard deviation."""
    return _direct("mhdev", _mhdev_core, 3, data, taus, ci, confidence)


def pdev(data, taus=Octave, *, ci=True, confidence=_DEFAULT_CONFIDENCE):
    """Parabolic deviation σ_PDEV(τ) (Vernotte). PDEV(τ₀) ≡ ADEV(τ₀)."""
    return _direct("pdev", _pdev_core, 2, data, taus, ci, confidence)


def _scaled(name: str, base: StabilityResult, factor: np.ndarray, ci: bool) -> StabilityResult:
    """A σ_x deviation derived by rescaling a σ_y result (tdev/htdev/ttotdev)."""
    dev = base.dev * factor
    if not ci:
        return _no_ci_result(name, base.tau, dev)
    return StabilityResult(
        name,
        base.tau,
        dev,
        base.noise_type,
        base.ci_lower * factor,
        base.ci_upper * factor,
        base.edf,
    )


def tdev(data, taus=Octave, *, ci=True, confidence=_DEFAULT_CONFIDENCE):
    """Time deviation σ_x(τ) = (τ/√3)·Mod σ_y(τ). Units of seconds."""
    r = mdev(data, taus, ci=ci, confidence=confidence)
    return _scaled("tdev", r, r.tau / np.sqrt(3.0), ci)


def htdev(data, taus=Octave, *, ci=True, confidence=_DEFAULT_CONFIDENCE):
    """Hadamard time deviation σ_x,HT(τ) = (τ/√(10/3))·Mod σ_y,MH(τ). Units of seconds."""
    r = mhdev(data, taus, ci=ci, confidence=confidence)
    return _scaled("htdev", r, r.tau / np.sqrt(10.0 / 3.0), ci)


def _total(
    name: str,
    core: _Core,
    var_type: str,
    dmax: int,
    data: PhaseData | FrequencyData,
    taus: TauMode | Sequence[int],
    ci: bool,
    correct_bias: bool,
    confidence: float,
) -> StabilityResult:
    """Shared path for the total family: optional bias correction, then CIs."""
    pd = _as_phase(data)
    m = _resolve_m(taus, pd.x.size, name)
    x = _f64(pd.x)
    raw = core(x, m, pd.tau0)
    tau = np.asarray(m, dtype=np.float64) * pd.tau0
    n = x.size
    t = (n - 1) * pd.tau0

    need_noise = correct_bias or ci
    noises = identify_noise(x, m, dmin=0, dmax=dmax) if need_noise else np.empty(0, dtype=object)
    dev = raw / _unbias_divisor(bias_correction(noises, var_type, tau, t)) if correct_bias else raw

    if not ci:
        return StabilityResult(name, tau, dev, noises, _empty_f(), _empty_f(), _empty_f())
    return _ci_result(name, x, m, tau, dev, pd.tau0, dmax, confidence, noises=noises)


def totdev(data, taus=Octave, *, ci=True, correct_bias=True, confidence=_DEFAULT_CONFIDENCE):
    """Total deviation (Howe / SP1065 mean-flip extension)."""
    return _total("totdev", _totdev_core, "totvar", 2, data, taus, ci, correct_bias, confidence)


def mtotdev(data, taus=Octave, *, ci=True, correct_bias=True, confidence=_DEFAULT_CONFIDENCE):
    """Modified Total deviation (Greenhall extension)."""
    return _total("mtotdev", _mtotdev_core, "mtot", 2, data, taus, ci, correct_bias, confidence)


def htotdev(data, taus=Octave, *, ci=True, correct_bias=True, confidence=_DEFAULT_CONFIDENCE):
    """Hadamard Total deviation (Greenhall extension on y = diff(x))."""
    return _total("htotdev", _htotdev_core, "htot", 3, data, taus, ci, correct_bias, confidence)


def mhtotdev(data, taus=Octave, *, ci=True, correct_bias=True, confidence=_DEFAULT_CONFIDENCE):
    """Modified Hadamard Total deviation (SigmaTau-original; Greenhall methodology)."""
    return _total("mhtotdev", _mhtotdev_core, "mhtot", 3, data, taus, ci, correct_bias, confidence)


def ttotdev(data, taus=Octave, *, ci=True, correct_bias=True, confidence=_DEFAULT_CONFIDENCE):
    """Time Total deviation σ_x(τ) = (τ/√3)·Mod-Total σ_y(τ). Units of seconds."""
    r = mtotdev(data, taus, ci=ci, correct_bias=correct_bias, confidence=confidence)
    return _scaled("ttotdev", r, r.tau / np.sqrt(3.0), ci)


def mtie(data, taus=Octave, *, ci=True, confidence=_DEFAULT_CONFIDENCE):
    """Maximum Time Interval Error (ITU-T G.810). Units of seconds.

    MTIE has no published EDF/CI model, so ``ci``/``confidence`` are accepted for
    signature uniformity but are no-ops (the result always has empty CI fields).
    """
    pd = _as_phase(data)
    m = _resolve_m(taus, pd.x.size, "mtie")
    dev = _mtie_core(_f64(pd.x), m, pd.tau0)
    tau = np.asarray(m, dtype=np.float64) * pd.tau0
    return _no_ci_result("mtie", tau, dev)
