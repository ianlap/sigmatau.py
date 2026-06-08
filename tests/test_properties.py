"""Algebraic identities and contracts that don't need the oracle."""

from __future__ import annotations

import numpy as np
import pytest

import sigmatau as st


def _phase(n: int = 512, seed: int = 0) -> st.PhaseData:
    rng = np.random.default_rng(seed)
    return st.PhaseData(np.cumsum(rng.standard_normal(n)) * 1e-9, 1.0)


def test_tdev_is_mdev_scaled() -> None:
    pd = _phase()
    m = [1, 2, 4, 8, 16]
    rm = st.mdev(pd, m)
    rt = st.tdev(pd, m)
    np.testing.assert_allclose(rt.dev, rm.tau / np.sqrt(3.0) * rm.dev, rtol=1e-13)


def test_htdev_is_mhdev_scaled() -> None:
    pd = _phase()
    m = [1, 2, 4, 8]
    rmh = st.mhdev(pd, m)
    rht = st.htdev(pd, m)
    np.testing.assert_allclose(rht.dev, rmh.tau / np.sqrt(10.0 / 3.0) * rmh.dev, rtol=1e-13)


def test_constant_phase_is_zero() -> None:
    # Constant phase has zero differences; mdev/mhdev accumulate the constant
    # through prefix sums, leaving machine-precision residual (~1e-14 at this
    # scale) — the same the Julia oracle produces. Allow a tiny absolute floor.
    pd = st.PhaseData(np.full(128, 3.14), 1.0)
    for fn in (st.adev, st.mdev, st.hdev, st.mhdev):
        np.testing.assert_allclose(fn(pd, [1, 2, 4]).dev, 0.0, atol=1e-12)


def test_linear_phase_zero_adev() -> None:
    # Linear phase = constant frequency: ADEV's second difference annihilates it.
    pd = st.PhaseData(np.arange(128, dtype=float) * 2.5, 1.0)
    np.testing.assert_allclose(st.adev(pd, [1, 2, 4, 8]).dev, 0.0, atol=1e-20)


def test_frequency_dispatch_matches_phase_integration() -> None:
    rng = np.random.default_rng(7)
    y = rng.standard_normal(256) * 1e-12
    fd = st.FrequencyData(y, 1.0)
    pd = st.PhaseData(np.cumsum(y) * 1.0, 1.0)
    m = [1, 2, 4, 8]
    for fn in (st.adev, st.mdev, st.hdev, st.mhdev):
        np.testing.assert_allclose(fn(fd, m).dev, fn(pd, m).dev, rtol=1e-13)


def test_undersampled_is_nan() -> None:
    pd = st.PhaseData(np.arange(8.0), 1.0)
    # adev needs N - 2m >= 2; m=4 -> 0 windows -> NaN.
    assert np.isnan(st.adev(pd, [4]).dev[0])


def test_ci_false_leaves_empty_fields() -> None:
    pd = _phase(64)
    r = st.adev(pd, [1, 2, 4])
    assert r.noise_type.size == 0
    assert r.ci_lower.size == 0
    assert r.ci_upper.size == 0
    assert r.edf.size == 0


def test_ci_true_not_implemented() -> None:
    pd = _phase(64)
    for fn in (st.adev, st.mdev, st.tdev, st.hdev, st.mhdev, st.htdev):
        with pytest.raises(NotImplementedError):
            fn(pd, [1, 2], ci=True)


def test_taumode_and_explicit_grid_agree() -> None:
    pd = _phase(1024)
    from_mode = st.adev(pd, st.Octave)
    from_explicit = st.adev(pd, st.tau_values(st.Octave, 1024, "adev"))
    np.testing.assert_array_equal(from_mode.tau, from_explicit.tau)
    np.testing.assert_array_equal(from_mode.dev, from_explicit.dev)
