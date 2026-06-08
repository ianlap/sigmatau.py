"""Property tests for the calibrated power-law noise generator.

``noise_gen`` can't be bit-parity-checked against the Julia oracle (different
RNGs), so it is validated by the same properties the oracle's own test suite
checks: per-component calibration is exact, the h→σ analytic identities hold,
independent components sum in quadrature, output is deterministic under a seed,
and the argument validation matches.
"""

from __future__ import annotations

import numpy as np
import pytest

import sigmatau as st


def _sigma1(data: st.FrequencyData) -> float:
    return float(st.adev(data, [1], ci=False).dev[0])


def test_single_component_calibration_is_exact() -> None:
    # Each component is rescaled to hit its target σ_y(τ₀) for the drawn record.
    for alpha in (2, 1, 0, -1, -2):
        fd = st.noise_gen(st.FrequencyData, 4096, 1.0, sigma1={alpha: 1e-12}, rng=7)
        assert _sigma1(fd) == pytest.approx(1e-12, rel=1e-9)


def test_phase_and_frequency_agree() -> None:
    # Same seed, same mixture → the PhaseData path is the cumsum of the y path.
    y = st.noise_gen(st.FrequencyData, 1024, 2.0, sigma1={0: 1e-12}, rng=3)
    p = st.noise_gen(st.PhaseData, 1024, 2.0, sigma1={0: 1e-12}, rng=3)
    np.testing.assert_allclose(p.x, np.cumsum(y.y) * 2.0, rtol=1e-13)


def test_composite_sums_in_quadrature() -> None:
    # Independent components → total variance ≈ sum of component variances.
    targets = {0: 1e-12, -2: 3e-13}
    fd = st.noise_gen(st.FrequencyData, 200_000, 1.0, sigma1=targets, rng=11)
    expected = np.sqrt(sum(s**2 for s in targets.values()))
    assert _sigma1(fd) == pytest.approx(expected, rel=0.05)


def test_h_mode_ffm_identity() -> None:
    # FFM (α=-1): σ_y(τ₀)² = 2·ln2·h_{-1}, independent of τ₀.
    h = 4e-26
    fd = st.noise_gen(st.FrequencyData, 4096, 1.0, h={-1: h}, rng=5)
    assert _sigma1(fd) ** 2 == pytest.approx(2.0 * np.log(2.0) * h, rel=1e-9)


def test_h_mode_rwfm_scales_with_tau0() -> None:
    # RWFM (α=-2): σ_y(τ₀)² = (2π²/3)·h_{-2}·τ₀.
    h, tau0 = 1e-28, 10.0
    fd = st.noise_gen(st.FrequencyData, 4096, tau0, h={-2: h}, rng=9)
    assert _sigma1(fd) ** 2 == pytest.approx(2.0 * np.pi**2 * h * tau0 / 3.0, rel=1e-9)


def test_determinism_and_independent_streams() -> None:
    a = st.noise_gen(st.FrequencyData, 512, 1.0, sigma1={0: 1e-12}, rng=42)
    b = st.noise_gen(st.FrequencyData, 512, 1.0, sigma1={0: 1e-12}, rng=42)
    c = st.noise_gen(st.FrequencyData, 512, 1.0, sigma1={0: 1e-12}, rng=43)
    np.testing.assert_array_equal(a.y, b.y)  # same seed → identical
    assert not np.array_equal(a.y, c.y)  # different seed → different


def test_zero_amplitude_is_zero_vector() -> None:
    fd = st.noise_gen(st.FrequencyData, 64, 1.0, sigma1={0: 0.0}, rng=1)
    np.testing.assert_array_equal(fd.y, np.zeros(64))


def test_argument_validation() -> None:
    with pytest.raises(ValueError):  # both sigma1 and h
        st.noise_gen(st.FrequencyData, 64, 1.0, sigma1={0: 1e-12}, h={0: 1e-24})
    with pytest.raises(ValueError):  # empty mixture
        st.noise_gen(st.FrequencyData, 64, 1.0)
    with pytest.raises(ValueError):  # N too small
        st.noise_gen(st.FrequencyData, 2, 1.0, sigma1={0: 1e-12})
    with pytest.raises(ValueError):  # α out of range
        st.noise_gen(st.FrequencyData, 64, 1.0, sigma1={3: 1e-12})
    with pytest.raises(ValueError):  # negative amplitude
        st.noise_gen(st.FrequencyData, 64, 1.0, sigma1={0: -1.0})
    with pytest.raises(ValueError):  # bad kind
        st.noise_gen(int, 64, 1.0, sigma1={0: 1e-12})
