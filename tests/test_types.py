"""Data/result type behavior: defaults, validation, repr, suite indexing, grids."""

from __future__ import annotations

import numpy as np
import pytest

import sigmatau as st
from sigmatau.grids import _default_m_values, _kernel_m_max


def test_tau0_defaults_to_one() -> None:
    assert st.PhaseData(np.arange(4.0)).tau0 == 1.0
    assert st.FrequencyData(np.arange(4.0)).tau0 == 1.0


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        st.PhaseData(np.array([1.0]))  # need >= 2 samples
    with pytest.raises(ValueError):
        st.PhaseData(np.arange(4.0), 0.0)  # tau0 must be positive
    with pytest.raises(ValueError):
        st.FrequencyData(np.arange(4.0), -1.0)


def test_integer_input_promoted_to_float64() -> None:
    pd = st.PhaseData(np.arange(4))  # int input
    assert pd.x.dtype == np.float64


def test_result_repr() -> None:
    pd = st.PhaseData(np.cumsum(np.random.default_rng(0).standard_normal(256)) * 1e-9)
    text = repr(st.adev(pd, st.Octave))
    assert text.startswith("StabilityResult(adev,")
    assert "no CI" in text


def test_suite_indexing() -> None:
    pd = st.PhaseData(np.cumsum(np.random.default_rng(1).standard_normal(256)) * 1e-9)
    a, h = st.adev(pd, [1, 2, 4]), st.hdev(pd, [1, 2, 4])
    suite = st.StabilitySuite((a, h), "phase", 1.0, 256, None, "Octave")
    assert len(suite) == 2
    assert suite["adev"] is a
    assert suite[1] is h
    assert "hdev" in suite
    assert suite.keys() == ("adev", "hdev")
    with pytest.raises(KeyError):
        suite["nope"]


def test_octave_grid_matches_formula() -> None:
    # m_max = (1024 - 2)//2 = 511 -> octave grid [1,2,...,256].
    assert _kernel_m_max(1024, "adev") == 511
    assert _default_m_values(1024, "adev") == [1, 2, 4, 8, 16, 32, 64, 128, 256]


def test_unknown_kernel_raises() -> None:
    with pytest.raises(ValueError):
        _kernel_m_max(1024, "bogus")
