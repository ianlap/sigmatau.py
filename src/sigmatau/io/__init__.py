"""File IO: readers, detrend, gap fill, and result round-trip — mirrors Julia's ``io/``."""

from __future__ import annotations

from .detrend import detrend
from .fillgaps import fillgaps
from .read import read_frequency, read_phase
from .results import load_result, load_suite, save_result, save_suite

__all__ = [
    "read_phase",
    "read_frequency",
    "detrend",
    "fillgaps",
    "save_result",
    "load_result",
    "save_suite",
    "load_suite",
]
