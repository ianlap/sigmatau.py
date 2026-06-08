"""Compute-all entry point — mirrors ``suite.jl``.

``stability`` runs a set of deviations on one record and collects them into a
``StabilitySuite``, forwarding ``ci``/``confidence`` (and ``correct_bias`` to the
total family), exactly like the Julia oracle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

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
from .grids import Octave, TauMode, tau_values
from .types import FrequencyData, PhaseData, StabilityResult, StabilitySuite

DEFAULT_DEVIATIONS = ("adev", "mdev", "hdev", "tdev")

_DISPATCH: dict[str, Callable[..., StabilityResult]] = {
    "adev": adev,
    "mdev": mdev,
    "tdev": tdev,
    "hdev": hdev,
    "mhdev": mhdev,
    "htdev": htdev,
    "totdev": totdev,
    "mtotdev": mtotdev,
    "ttotdev": ttotdev,
    "htotdev": htotdev,
    "mhtotdev": mhtotdev,
    "mtie": mtie,
    "pdev": pdev,
}
_TOTAL = frozenset({"totdev", "mtotdev", "ttotdev", "htotdev", "mhtotdev"})


def stability(
    data: PhaseData | FrequencyData,
    *,
    devs: Sequence[str] = DEFAULT_DEVIATIONS,
    taus: TauMode | Sequence[int] = Octave,
    ci: bool = True,
    confidence: float = 0.683,
    correct_bias: bool = True,
) -> StabilitySuite:
    """Compute several deviations on one record and collect them into a suite.

    ``devs`` is a sequence of deviation names (default the Stable32-like core
    ``("adev", "mdev", "hdev", "tdev")``). ``taus`` is a ``TauMode`` or explicit
    averaging factors; ``ci``/``confidence`` and (for the total family)
    ``correct_bias`` flow through to each deviation.
    """
    if isinstance(data, FrequencyData):
        kind, n, tau0 = "frequency", data.y.size, data.tau0
    else:
        kind, n, tau0 = "phase", data.x.size, data.tau0

    results = []
    for sym in devs:
        fn = _DISPATCH.get(sym)
        if fn is None:
            raise ValueError(f"stability: unknown deviation {sym!r}")
        m = tau_values(taus, n, sym) if isinstance(taus, TauMode) else [int(v) for v in taus]
        if sym in _TOTAL:
            results.append(fn(data, m, ci=ci, confidence=confidence, correct_bias=correct_bias))
        else:
            results.append(fn(data, m, ci=ci, confidence=confidence))

    tau_mode = taus.value if isinstance(taus, TauMode) else "explicit"
    return StabilitySuite(tuple(results), kind, tau0, n, confidence if ci else None, tau_mode)


__all__ = ["stability", "DEFAULT_DEVIATIONS"]
