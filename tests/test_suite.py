"""stability() compute-all suite: composition matches calling deviations directly."""

from __future__ import annotations

import numpy as np

import sigmatau as st


def _phase(n: int = 1024) -> st.PhaseData:
    return st.PhaseData(np.cumsum(np.random.default_rng(0).standard_normal(n)) * 1e-9, 1.0)


def test_default_devs_and_metadata() -> None:
    pd = _phase()
    s = st.stability(pd)
    assert s.keys() == st.DEFAULT_DEVIATIONS == ("adev", "mdev", "hdev", "tdev")
    assert s.data_kind == "phase"
    assert s.n == 1024
    assert s.tau0 == 1.0
    assert s.tau_mode == "Octave"
    assert s.confidence == 0.683  # ci=True default


def test_suite_results_match_direct_calls() -> None:
    pd = _phase()
    s = st.stability(pd, devs=("adev", "hdev", "pdev"), taus=st.Octave)
    for sym, fn in (("adev", st.adev), ("hdev", st.hdev), ("pdev", st.pdev)):
        direct = fn(pd, st.Octave)
        np.testing.assert_array_equal(s[sym].dev, direct.dev)
        np.testing.assert_array_equal(s[sym].edf, direct.edf)
        assert list(s[sym].noise_type) == list(direct.noise_type)


def test_ci_false_suite_has_no_confidence() -> None:
    pd = _phase(256)
    s = st.stability(pd, devs=("adev",), ci=False)
    assert s.confidence is None
    assert s["adev"].edf.size == 0


def test_total_family_correct_bias_flows() -> None:
    pd = _phase(512)
    biased = st.stability(pd, devs=("totdev",))["totdev"]
    raw = st.stability(pd, devs=("totdev",), correct_bias=False)["totdev"]
    # Bias correction changes the deviation values at long τ (atol=0: the
    # values are ~1e-10, far below np.allclose's default atol).
    assert not np.allclose(biased.dev, raw.dev, rtol=1e-9, atol=0.0)


def test_unknown_deviation_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        st.stability(_phase(256), devs=("nope",))
