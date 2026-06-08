"""Read raw phase / frequency files into PhaseData / FrequencyData — mirrors ``io/read.jl``.

Optional preprocessing in one call: column scaling, detrend, and Howe gap fill.
"""

from __future__ import annotations

import os

import numpy as np

from ..types import FrequencyData, PhaseData
from .detrend import _detrend_core
from .fillgaps import _howe_fillgaps_core, _make_equispaced


def _read_columns(path: str, *, header: int = 0, delim: str | None = None) -> np.ndarray:
    """Load a numeric table as a 2-D float64 array; delimiter auto-detected by extension."""
    if not os.path.isfile(path):
        raise ValueError(f"read_columns: file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    d = delim if delim is not None else ("," if ext == ".csv" else "\t" if ext == ".tsv" else None)
    try:
        return np.loadtxt(path, delimiter=d, skiprows=header, ndmin=2, dtype=np.float64)
    except ValueError as err:
        raise ValueError(
            f"read_columns: failed to coerce {path} to a float matrix (non-numeric content?): {err}"
        ) from err


def _ingest(
    path: str,
    *,
    tau0: float | None,
    time_col: int,
    value_col: int,
    header: int,
    delim: str | None,
    scaling: float,
    detrend: str,
    fillgaps: bool,
) -> tuple[np.ndarray, float]:
    m = _read_columns(path, header=header, delim=delim)
    ncols = m.shape[1]
    if value_col < 1:
        raise ValueError("read: value_col must be ≥ 1")
    if value_col > ncols:
        raise ValueError(f"read: value_col={value_col} exceeds available columns ({ncols})")

    have_time = time_col >= 1
    t = None
    if have_time:
        if time_col > ncols:
            raise ValueError(f"read: time_col={time_col} exceeds available columns ({ncols})")
        t = m[:, time_col - 1]
    v = np.array(m[:, value_col - 1], dtype=np.float64)

    if scaling != 1:
        v = v * scaling

    if tau0 is not None:
        resolved = float(tau0)
    elif t is not None and t.size > 1:
        resolved = float(np.median(np.diff(t)))
    else:
        raise ValueError("read: tau0 must be supplied when no time column is present")

    if fillgaps:
        if t is None:
            raise ValueError("read: fillgaps=True requires a time column (got time_col=0)")
        _, v_eq = _make_equispaced(t, v)
        v, _ = _howe_fillgaps_core(v_eq)

    if detrend != "none":
        v = _detrend_core(v, detrend)

    return v, resolved


def read_phase(
    path: str,
    *,
    tau0: float | None = None,
    time_col: int = 1,
    value_col: int = 2,
    header: int = 0,
    delim: str | None = None,
    scaling: float = 1.0,
    detrend: str = "none",
    fillgaps: bool = False,
) -> PhaseData:
    """Read a phase-data file into a :class:`PhaseData` (see :func:`read_frequency`)."""
    v, t0 = _ingest(
        path,
        tau0=tau0,
        time_col=time_col,
        value_col=value_col,
        header=header,
        delim=delim,
        scaling=scaling,
        detrend=detrend,
        fillgaps=fillgaps,
    )
    return PhaseData(v, t0)


def read_frequency(
    path: str,
    *,
    tau0: float | None = None,
    time_col: int = 1,
    value_col: int = 2,
    header: int = 0,
    delim: str | None = None,
    scaling: float = 1.0,
    detrend: str = "none",
    fillgaps: bool = False,
) -> FrequencyData:
    """Read a fractional-frequency file into a :class:`FrequencyData`.

    ``tau0`` is auto-inferred from the time column (median Δt) when omitted, and
    required when ``time_col=0``. ``time_col``/``value_col`` are 1-based; ``delim``
    is auto-detected (``,`` for .csv, tab for .tsv, whitespace otherwise).
    ``scaling`` multiplies the samples; ``detrend`` ∈ {none, mean, endpoint,
    linear}; ``fillgaps`` equispaces on the min spacing and Howe-imputes NaNs.
    """
    v, t0 = _ingest(
        path,
        tau0=tau0,
        time_col=time_col,
        value_col=value_col,
        header=header,
        delim=delim,
        scaling=scaling,
        detrend=detrend,
        fillgaps=fillgaps,
    )
    return FrequencyData(v, t0)


__all__ = ["read_phase", "read_frequency"]
