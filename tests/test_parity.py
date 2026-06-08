"""Parity against the Julia oracle.

For every row in the vendored ``julia_reference.csv``, recompute the deviation in
Python on the *same* input record and require agreement to ``rtol=1e-11`` — both
libraries use Float64 and the identical algorithm, so any larger gap is a real
discrepancy, not floating-point noise.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import sigmatau as st

FIX = Path(__file__).parent / "fixtures" / "julia"

DEVS = {
    "adev": st.adev,
    "mdev": st.mdev,
    "tdev": st.tdev,
    "hdev": st.hdev,
    "mhdev": st.mhdev,
    "htdev": st.htdev,
    "totdev": st.totdev,
    "mtie": st.mtie,
    "pdev": st.pdev,
}


def _load_rows() -> dict[tuple[str, str, str, str], list[tuple[int, float, float]]]:
    rows: dict[tuple[str, str, str, str], list[tuple[int, float, float]]] = {}
    with open(FIX / "julia_reference.csv") as f:
        for r in csv.DictReader(f):
            key = (r["input"], r["data_kind"], r["deviation"], r["grid"])
            rows.setdefault(key, []).append((int(r["m"]), float(r["tau"]), float(r["dev"])))
    return rows


def _load_input(name: str, kind: str) -> st.PhaseData | st.FrequencyData:
    arr = np.loadtxt(FIX / "inputs" / f"{name}.txt")
    return st.FrequencyData(arr, 1.0) if kind == "frequency" else st.PhaseData(arr, 1.0)


ROWS = _load_rows()


@pytest.mark.parametrize("key", list(ROWS), ids=lambda k: "/".join(k))
def test_matches_julia(key: tuple[str, str, str, str]) -> None:
    name, kind, dev, _grid = key
    data = _load_input(name, kind)
    triples = sorted(ROWS[key])
    m = [t[0] for t in triples]
    julia_tau = np.array([t[1] for t in triples])
    julia_dev = np.array([t[2] for t in triples])

    res = DEVS[dev](data, m, ci=False)
    np.testing.assert_allclose(res.tau, julia_tau, rtol=1e-11, atol=0.0)
    np.testing.assert_allclose(res.dev, julia_dev, rtol=1e-11, atol=0.0, equal_nan=True)


def test_fixtures_present() -> None:
    assert ROWS, "no fixture rows loaded — run tools/export_python_fixtures.jl"
