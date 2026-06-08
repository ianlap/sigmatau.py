"""sigmatau — clock-stability analysis, a NumPy port of SigmaTau.jl.

The public surface mirrors the Julia oracle one-to-one. All 13 deviations are
implemented with noise ID, EDF, χ² confidence intervals, and bias correction,
plus a calibrated power-law noise generator; see the project roadmap.
"""

from __future__ import annotations

from .deviations import (
    adev,
    hdev,
    htdev,
    htotdev,
    mdev,
    mhdev,
    mhtotdev,
    mtie,
    mtotdev,
    pdev,
    tdev,
    totdev,
    ttotdev,
)
from .edf import bias_correction, calculate_edf, confidence_intervals
from .grids import (
    AllTaus,
    Decade,
    HalfDecade,
    HalfOctave,
    Octave,
    QuarterOctave,
    TauMode,
    tau_values,
)
from .noise import identify_noise, noise_gen
from .spectral import L, Sx, Sy
from .types import FrequencyData, PhaseData, SpectralResult, StabilityResult, StabilitySuite

__version__ = "0.1.0"

__all__ = [
    # data + result types
    "PhaseData",
    "FrequencyData",
    "StabilityResult",
    "StabilitySuite",
    "SpectralResult",
    # tau grids
    "TauMode",
    "AllTaus",
    "Octave",
    "HalfOctave",
    "QuarterOctave",
    "Decade",
    "HalfDecade",
    "tau_values",
    # deviations
    "adev",
    "mdev",
    "tdev",
    "hdev",
    "mhdev",
    "htdev",
    "totdev",
    "mtotdev",
    "ttotdev",
    "htotdev",
    "mhtotdev",
    "mtie",
    "pdev",
    # statistics
    "identify_noise",
    "calculate_edf",
    "confidence_intervals",
    "bias_correction",
    # noise generation
    "noise_gen",
    # spectral estimators
    "Sy",
    "Sx",
    "L",
]
