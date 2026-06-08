"""Matplotlib plotting helpers — mirrors ``SigmaTauRecipesBaseExt.jl``.

Not imported by the top-level package (matplotlib is an optional dependency,
mirroring the Julia weakdep extension). Use ``from sigmatau.plotting import …``.
Install the extra with ``pip install sigmatau[plot]``.

- :func:`plot_deviation` — a σ(τ) curve on log-log axes, with CI as error bars
  (default) or a filled band (``ci_band=True``).
- :func:`plot_suite` — overlay every deviation in a suite.
- :func:`plot_spectral` — S_y/S_x on log-log axes; ℒ(f) on a log-frequency,
  linear-dB axis.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

from .types import SpectralResult, StabilityResult, StabilitySuite


def _ensure_ax(ax: Any | None) -> Any:
    return ax if ax is not None else plt.subplots()[1]


def plot_deviation(
    result: StabilityResult,
    ax: Any | None = None,
    *,
    ci_band: bool = False,
    label: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot a single deviation on log-log axes; returns the Axes."""
    ax = _ensure_ax(ax)
    name = result.deviation_type.upper()
    lab = name if label is None else label
    has_ci = result.ci_lower.size > 0 and result.ci_upper.size > 0
    if has_ci and ci_band:
        (line,) = ax.plot(result.tau, result.dev, label=lab, **kwargs)
        ax.fill_between(
            result.tau, result.ci_lower, result.ci_upper, alpha=0.25, color=line.get_color()
        )
    elif has_ci:
        yerr = [result.dev - result.ci_lower, result.ci_upper - result.dev]
        ax.errorbar(result.tau, result.dev, yerr=yerr, label=lab, capsize=2, **kwargs)
    else:
        ax.plot(result.tau, result.dev, label=lab, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Averaging Time τ (s)")
    ax.set_ylabel(name)
    return ax


def plot_suite(suite: StabilitySuite, ax: Any | None = None, **kwargs: Any) -> Any:
    """Overlay every result in a suite on one set of log-log axes; returns the Axes."""
    ax = _ensure_ax(ax)
    for r in suite.results:
        plot_deviation(r, ax=ax, label=r.deviation_type.upper(), **kwargs)
    ax.set_ylabel("Deviation")
    ax.legend()
    return ax


def plot_spectral(
    result: SpectralResult, ax: Any | None = None, *, label: str | None = None, **kwargs: Any
) -> Any:
    """Plot a spectral-density estimate (drops the DC bin); returns the Axes."""
    ax = _ensure_ax(ax)
    keep = result.freq > 0
    f, psd = result.freq[keep], result.psd[keep]
    ax.set_xscale("log")
    ax.set_xlabel("Frequency f (Hz)")
    if result.spectral_type == "L":
        ax.plot(f, psd, label=label or "ℒ(f)", **kwargs)
        ax.set_ylabel("ℒ(f) (dBc/Hz)")
    else:
        ax.plot(f, psd, label=label or result.spectral_type, **kwargs)
        ax.set_yscale("log")
        ax.set_ylabel("S_y(f) (1/Hz)" if result.spectral_type == "Sy" else "S_x(f) (s²/Hz)")
    return ax


__all__ = ["plot_deviation", "plot_suite", "plot_spectral"]
