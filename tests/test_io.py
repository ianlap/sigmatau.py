"""IO: detrend, fillgaps (parity vs oracle), readers, and TSV round-trip."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import sigmatau as st

FIX = Path(__file__).parent / "fixtures" / "julia"


# --- detrend ---------------------------------------------------------------


def test_detrend_linear_removes_line() -> None:
    n = 256
    x = 3.0 + 0.5 * np.arange(n) + np.zeros(n)  # pure line, no noise
    r = st.detrend(st.PhaseData(x, 1.0), method="linear")
    np.testing.assert_allclose(r.x, 0.0, atol=1e-9)


def test_detrend_linear_matches_numpy_lsq() -> None:
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(300)) + 0.01 * np.arange(300)
    r = st.detrend(st.FrequencyData(x, 1.0), method="linear")
    idx = np.arange(300, dtype=float)
    coef = np.polynomial.polynomial.polyfit(idx, x, 1)
    np.testing.assert_allclose(r.y, x - np.polynomial.polynomial.polyval(idx, coef), rtol=1e-9)


def test_detrend_mean_and_endpoint() -> None:
    x = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    assert st.detrend(st.PhaseData(x), method="mean").x.mean() == pytest.approx(0.0, abs=1e-12)
    ep = st.detrend(st.PhaseData(x), method="endpoint").x
    assert ep[0] == pytest.approx(0.0, abs=1e-12)
    assert ep[-1] == pytest.approx(0.0, abs=1e-12)


def test_detrend_unknown_raises() -> None:
    with pytest.raises(ValueError):
        st.detrend(st.PhaseData(np.arange(10.0)), method="cubic")


# --- fillgaps (parity vs Julia oracle) -------------------------------------


def test_fillgaps_matches_julia() -> None:
    idx, inp, filled = [], [], []
    with open(FIX / "fillgaps_reference.csv") as f:
        for r in csv.DictReader(f):
            idx.append(int(r["idx"]))
            inp.append(float(r["input"]))  # "NaN" parses to nan
            filled.append(float(r["filled"]))
    x_in = np.array(inp)
    assert np.isnan(x_in).any()  # the fixture really has gaps
    out = st.fillgaps(st.PhaseData(x_in, 1.0))
    np.testing.assert_allclose(out.x, filled, rtol=1e-8, atol=1e-22)
    assert not np.isnan(out.x).any()


def test_fillgaps_no_gaps_is_identity() -> None:
    x = np.cumsum(np.random.default_rng(1).standard_normal(64)) * 1e-9
    out = st.fillgaps(st.PhaseData(x, 1.0))
    np.testing.assert_array_equal(out.x, x)


# --- readers ---------------------------------------------------------------


def test_read_phase_csv_roundtrip(tmp_path: Path) -> None:
    t = np.arange(50.0) * 0.1
    v = np.cumsum(np.random.default_rng(2).standard_normal(50)) * 1e-9
    p = tmp_path / "phase.csv"
    np.savetxt(p, np.column_stack([t, v]), delimiter=",")
    pd = st.read_phase(str(p))
    np.testing.assert_allclose(pd.x, v, rtol=1e-12)
    assert pd.tau0 == pytest.approx(0.1)  # inferred from median Δt


def test_read_no_time_column_requires_tau0(tmp_path: Path) -> None:
    v = np.arange(20.0)
    p = tmp_path / "v.txt"
    np.savetxt(p, v)
    with pytest.raises(ValueError):
        st.read_phase(str(p), time_col=0, value_col=1)
    pd = st.read_phase(str(p), time_col=0, value_col=1, tau0=2.0, scaling=1e-9)
    assert pd.tau0 == 2.0
    np.testing.assert_allclose(pd.x, v * 1e-9, rtol=1e-12)


def test_read_with_fillgaps(tmp_path: Path) -> None:
    # Irregular time grid (a missing sample) → equispace + Howe fill.
    t = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    v = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0]) * 1e-9
    p = tmp_path / "gap.csv"
    np.savetxt(p, np.column_stack([t, v]), delimiter=",")
    pd = st.read_phase(str(p), fillgaps=True)
    assert pd.x.size == 9  # t spans 0..8 at dt=1 → 9 samples
    assert not np.isnan(pd.x).any()


# --- result / suite TSV round-trip + cross-load ----------------------------


def _synth_phase() -> st.PhaseData:
    return st.PhaseData(np.loadtxt(FIX / "inputs" / "synth_phase.txt"), 1.0)


def test_result_python_roundtrip(tmp_path: Path) -> None:
    r = st.adev(_synth_phase())
    p = tmp_path / "r.tsv"
    st.save_result(str(p), r)
    loaded = st.load_result(str(p))
    np.testing.assert_array_equal(loaded.tau, r.tau)
    np.testing.assert_array_equal(loaded.dev, r.dev)
    np.testing.assert_array_equal(loaded.edf, r.edf)
    assert list(loaded.noise_type) == list(r.noise_type)


def test_cross_load_julia_result() -> None:
    loaded = st.load_result(str(FIX / "adev_result.tsv"))
    expected = st.adev(_synth_phase())
    assert loaded.deviation_type == "adev"
    np.testing.assert_allclose(loaded.dev, expected.dev, rtol=1e-11)
    np.testing.assert_allclose(loaded.edf, expected.edf, rtol=1e-11)
    np.testing.assert_allclose(loaded.ci_lower, expected.ci_lower, rtol=1e-9)
    assert list(loaded.noise_type) == list(expected.noise_type)


def test_cross_load_julia_suite() -> None:
    suite = st.load_suite(str(FIX / "suite.tsv"))
    assert suite.keys() == ("adev", "mdev", "hdev", "tdev")
    expected = st.stability(_synth_phase())
    for sym in suite.keys():
        np.testing.assert_allclose(suite[sym].dev, expected[sym].dev, rtol=1e-11)


def test_load_result_rejects_suite_file() -> None:
    with pytest.raises(ValueError):
        st.load_result(str(FIX / "suite.tsv"))
