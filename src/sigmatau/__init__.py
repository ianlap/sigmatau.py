"""sigmatau — clock-stability analysis, a NumPy port of SigmaTau.jl.

The public surface mirrors the Julia oracle one-to-one. This release implements
the bare overlapping deviations (``ci=False`` only); see the project roadmap.
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
from .types import FrequencyData, PhaseData, StabilityResult, StabilitySuite

__version__ = "0.1.0"

__all__ = [
    # data + result types
    "PhaseData",
    "FrequencyData",
    "StabilityResult",
    "StabilitySuite",
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
]
