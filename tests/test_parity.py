"""Parity against the Julia oracle (full ci=true output).

For every row in the vendored ``julia_reference.csv``, recompute the deviation in
Python on the *same* input record at the default settings (``ci=True``,
``correct_bias=True``) and require:

- ``tau``/``dev``/``edf`` to rtol 1e-11 (pure arithmetic, identical algorithm),
- ``noise_type`` to match exactly (the classifier must reproduce α identically),
- ``ci_lower``/``ci_upper`` to rtol 1e-9 (scipy vs Distributions.jl χ² quantiles
  differ slightly).

Deviations without a CI model (mtie) carry empty noise/EDF/CI columns and only
``tau``/``dev`` are checked.
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
    "mtotdev": st.mtotdev,
    "ttotdev": st.ttotdev,
    "htotdev": st.htotdev,
    "mhtotdev": st.mhtotdev,
    "mtie": st.mtie,
    "pdev": st.pdev,
}


def _load_rows():
    rows: dict[tuple[str, str, str, str], list[dict]] = {}
    with open(FIX / "julia_reference.csv") as f:
        for r in csv.DictReader(f):
            key = (r["input"], r["data_kind"], r["deviation"], r["grid"])
            rows.setdefault(key, []).append(
                {
                    "m": int(r["m"]),
                    "tau": float(r["tau"]),
                    "dev": float(r["dev"]),
                    "noise": r["noise_type"],
                    "edf": float(r["edf"]),
                    "lo": float(r["ci_lower"]),
                    "hi": float(r["ci_upper"]),
                }
            )
    return rows


def _load_input(name: str, kind: str) -> st.PhaseData | st.FrequencyData:
    arr = np.loadtxt(FIX / "inputs" / f"{name}.txt")
    return st.FrequencyData(arr, 1.0) if kind == "frequency" else st.PhaseData(arr, 1.0)


ROWS = _load_rows()


@pytest.mark.parametrize("key", list(ROWS), ids=lambda k: "/".join(k))
def test_matches_julia(key: tuple[str, str, str, str]) -> None:
    name, kind, dev, _grid = key
    data = _load_input(name, kind)
    triples = sorted(ROWS[key], key=lambda d: d["m"])
    m = [t["m"] for t in triples]

    res = DEVS[dev](data, m)  # default settings = oracle behavior

    np.testing.assert_allclose(res.tau, [t["tau"] for t in triples], rtol=1e-11, atol=0.0)
    np.testing.assert_allclose(
        res.dev, [t["dev"] for t in triples], rtol=1e-11, atol=0.0, equal_nan=True
    )

    has_ci = triples[0]["noise"] != ""
    if not has_ci:  # mtie: no noise/EDF/CI
        assert res.noise_type.size == 0
        return

    assert list(res.noise_type) == [t["noise"] for t in triples]
    np.testing.assert_allclose(
        res.edf, [t["edf"] for t in triples], rtol=1e-11, atol=0.0, equal_nan=True
    )
    np.testing.assert_allclose(
        res.ci_lower, [t["lo"] for t in triples], rtol=1e-9, atol=0.0, equal_nan=True
    )
    np.testing.assert_allclose(
        res.ci_upper, [t["hi"] for t in triples], rtol=1e-9, atol=0.0, equal_nan=True
    )


def test_fixtures_present() -> None:
    assert ROWS, "no fixture rows loaded — run tools/export_python_fixtures.jl"
