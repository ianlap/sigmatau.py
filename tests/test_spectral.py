"""Spectral estimators: parity vs the Julia oracle + density-normalization checks."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import sigmatau as st

FIX = Path(__file__).parent / "fixtures" / "julia"
F_CARRIER = 1.0e7


def _load_spectral():
    rows: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
    with open(FIX / "spectral_reference.csv") as f:
        for r in csv.DictReader(f):
            rows.setdefault((r["input"], r["estimator"]), []).append(
                (int(r["idx"]), float(r["freq"]), float(r["psd"]))
            )
    return rows


def _load_input(name: str) -> st.PhaseData | st.FrequencyData:
    arr = np.loadtxt(FIX / "inputs" / f"{name}.txt")
    return st.PhaseData(arr, 1.0) if "phase" in name else st.FrequencyData(arr, 1.0)


SPECTRAL = _load_spectral()


@pytest.mark.parametrize("key", list(SPECTRAL), ids=lambda k: "/".join(k))
def test_spectral_matches_julia(key: tuple[str, str]) -> None:
    name, estimator = key
    data = _load_input(name)
    if estimator == "Sy":
        r = st.Sy(data)
    elif estimator == "Sx":
        r = st.Sx(data)
    else:
        r = st.L(data, f_carrier=F_CARRIER)

    triples = sorted(SPECTRAL[key])
    np.testing.assert_allclose(r.freq, [t[1] for t in triples], rtol=1e-9, atol=0.0)
    np.testing.assert_allclose(r.psd, [t[2] for t in triples], rtol=1e-9, atol=0.0)


def test_onesided_grid_shape() -> None:
    fd = st.FrequencyData(np.random.default_rng(0).standard_normal(1024) * 1e-12, 1.0)
    r = st.Sy(fd, nperseg=256)
    assert r.freq.size == 256 // 2 + 1
    assert r.freq[0] == 0.0
    assert r.freq[-1] == pytest.approx(0.5)  # Nyquist = fs/2 at tau0=1


def test_density_normalization_recovers_variance() -> None:
    # Σ S_y(f)·Δf ≈ var(y) over [0, fs/2] (the density convention).
    rng = np.random.default_rng(1)
    y = rng.standard_normal(8192) * 1e-12
    r = st.Sy(st.FrequencyData(y, 1.0))
    df = r.freq[1] - r.freq[0]
    integral = r.psd.sum() * df
    assert integral == pytest.approx(np.var(y), rel=0.1)


def test_bad_window_and_carrier_raise() -> None:
    fd = st.FrequencyData(np.ones(64) + np.arange(64) * 1e-9, 1.0)
    with pytest.raises(ValueError):
        st.Sy(fd, window="blackman")
    with pytest.raises(ValueError):
        st.L(fd, f_carrier=0.0)
