"""Both backends agree: the Numba njit kernels and the pure-NumPy fallback must
produce identical results for the loop-bound deviations.

test_parity.py already pins whichever backend is installed against the Julia
oracle; this pins the *other* path to it, so both are oracle-validated.
"""

from __future__ import annotations

import numpy as np
import pytest

import sigmatau as st
from sigmatau import kernels

_TOTAL = {"mtotdev": st.mtotdev, "htotdev": st.htotdev, "mhtotdev": st.mhtotdev}
_M = [1, 2, 4, 8, 16, 32]


def _phase() -> st.PhaseData:
    return st.PhaseData(np.cumsum(np.random.default_rng(0).standard_normal(2000)) * 1e-9, 1.0)


@pytest.mark.skipif(not kernels._HAS_NUMBA, reason="numba absent; only one backend to test")
@pytest.mark.parametrize("name", [*_TOTAL, "pdev"])
def test_numba_and_fallback_agree(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    pd = _phase()
    if name == "pdev":
        njit_dev = st.pdev(pd, _M, ci=False).dev
        monkeypatch.setattr(kernels, "_HAS_NUMBA", False)
        fallback_dev = st.pdev(pd, _M, ci=False).dev
    else:
        fn = _TOTAL[name]
        njit_dev = fn(pd, _M, ci=False, correct_bias=False).dev
        monkeypatch.setattr(kernels, "_HAS_NUMBA", False)
        fallback_dev = fn(pd, _M, ci=False, correct_bias=False).dev
    np.testing.assert_allclose(fallback_dev, njit_dev, rtol=1e-11, atol=0.0, equal_nan=True)


def test_dispatch_reads_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # The *_core dispatchers must consult _HAS_NUMBA at call time (so the fallback
    # is reachable), not bind a backend at import.
    monkeypatch.setattr(kernels, "_HAS_NUMBA", False)
    x = np.cumsum(np.random.default_rng(1).standard_normal(512)) * 1e-9
    out = kernels._mtotdev_core(x, [1, 2, 4], 1.0)
    assert out.shape == (3,) and np.all(np.isfinite(out))
