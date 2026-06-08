"""Smoke tests for the matplotlib plotting helpers (no parity contract — visual)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sigmatau as st  # noqa: E402
from sigmatau.plotting import plot_deviation, plot_spectral, plot_suite  # noqa: E402


def _phase() -> st.PhaseData:
    return st.PhaseData(np.cumsum(np.random.default_rng(0).standard_normal(1024)) * 1e-9, 1.0)


def test_plot_deviation_with_ci_errorbars() -> None:
    ax = plot_deviation(st.adev(_phase()))
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert len(ax.lines) >= 1  # errorbar draws at least the data line
    plt.close(ax.figure)


def test_plot_deviation_ci_band_and_no_ci() -> None:
    pd = _phase()
    ax = plot_deviation(st.adev(pd), ci_band=True)
    assert ax.collections  # fill_between adds a PolyCollection
    plt.close(ax.figure)
    ax2 = plot_deviation(st.adev(pd, ci=False))
    assert len(ax2.lines) == 1
    plt.close(ax2.figure)


def test_plot_suite_overlays_each_deviation() -> None:
    ax = plot_suite(st.stability(_phase()))
    assert ax.get_legend() is not None
    assert len(ax.get_legend().get_texts()) == 4  # default 4 deviations
    plt.close(ax.figure)


def test_plot_spectral_sy_and_l() -> None:
    fd = st.FrequencyData(np.random.default_rng(1).standard_normal(1024) * 1e-12, 1.0)
    ax = plot_spectral(st.Sy(fd))
    assert ax.get_yscale() == "log"
    assert ax.get_xscale() == "log"
    plt.close(ax.figure)
    ax2 = plot_spectral(st.L(fd, f_carrier=1e7))
    assert ax2.get_yscale() == "linear"  # dB ordinate
    assert "dBc" in ax2.get_ylabel()
    plt.close(ax2.figure)


def test_plot_accepts_existing_axes() -> None:
    _, ax = plt.subplots()
    out = plot_deviation(st.adev(_phase()), ax=ax)
    assert out is ax
    plt.close(ax.figure)
