"""Save/load StabilityResult and StabilitySuite as tab-separated text.

Mirrors ``io/results.jl`` — a self-describing, stdlib-only TSV format that round
trips with the Julia oracle. The on-disk ``# calc_ci=`` token is the v1/v2 format
key (kept distinct from the public ``ci`` kwarg), so files written by either
language load in the other.
"""

from __future__ import annotations

import datetime
import re
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from ..types import StabilityResult, StabilitySuite

_IO_VERSION = "2"


def _pkg_version() -> str:
    try:
        return version("sigmatau")
    except PackageNotFoundError:
        return "unknown"


def _write_result_table(lines: list[str], r: StabilityResult) -> None:
    has_ci = r.ci_lower.size > 0
    lines.append(f"# calc_ci={'true' if has_ci else 'false'}")
    lines.append("tau\tdev\tnoise_type\tci_lower\tci_upper\tedf")
    for k in range(r.tau.size):
        noise_s = str(r.noise_type[k]) if has_ci else ""
        ci_lo = str(float(r.ci_lower[k])) if has_ci else "NaN"
        ci_hi = str(float(r.ci_upper[k])) if has_ci else "NaN"
        edf_s = str(float(r.edf[k])) if has_ci else "NaN"
        lines.append(f"{float(r.tau[k])}\t{float(r.dev[k])}\t{noise_s}\t{ci_lo}\t{ci_hi}\t{edf_s}")


def _parse_result_rows(deviation_type: str, has_ci: bool, data_lines: list[str]) -> StabilityResult:
    n = len(data_lines)
    tau = np.empty(n, dtype=np.float64)
    dev = np.empty(n, dtype=np.float64)
    noise = np.empty(n, dtype=object)
    ci_lower = np.empty(n, dtype=np.float64)
    ci_upper = np.empty(n, dtype=np.float64)
    edf = np.empty(n, dtype=np.float64)
    for k, line in enumerate(data_lines):
        parts = line.split("\t")
        if len(parts) != 6:
            raise ValueError(f"malformed row {k} (expected 6 columns, got {len(parts)})")
        tau[k] = float(parts[0])
        dev[k] = float(parts[1])
        noise[k] = parts[2]
        ci_lower[k] = float(parts[3])
        ci_upper[k] = float(parts[4])
        edf[k] = float(parts[5])
    if not has_ci:
        empty_f = np.empty(0, dtype=np.float64)
        return StabilityResult(
            deviation_type,
            tau,
            dev,
            np.empty(0, dtype=object),
            empty_f,
            empty_f.copy(),
            empty_f.copy(),
        )
    return StabilityResult(deviation_type, tau, dev, noise, ci_lower, ci_upper, edf)


def save_result(path: str, r: StabilityResult) -> str:
    """Write a single ``StabilityResult`` to a tab-delimited text file (format v1)."""
    lines = ["# SigmaTau StabilityResult v1", f"# deviation_type={r.deviation_type}"]
    _write_result_table(lines, r)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def load_result(path: str) -> StabilityResult:
    """Read a single ``StabilityResult`` written by :func:`save_result`."""
    with open(path) as f:
        lines = f.read().splitlines()
    if lines and "StabilitySuite" in lines[0]:
        raise ValueError(
            f"load_result: {path!r} is a StabilitySuite file — use load_suite instead."
        )

    deviation_type = "unknown"
    has_ci = True  # v1 compat: assume CI present if header absent
    for line in lines:
        if line.startswith("# deviation_type="):
            deviation_type = line[len("# deviation_type=") :]
        elif line.startswith("# calc_ci="):
            has_ci = line[len("# calc_ci=") :].strip() == "true"

    data_lines = [ln for ln in lines if not ln.startswith("#") and not ln.startswith("tau")]
    if not data_lines:
        raise ValueError(f"load_result: no data rows found in {path!r}")
    return _parse_result_rows(deviation_type, has_ci, data_lines)


def save_suite(path: str, suite: StabilitySuite, *, source_file: str = "") -> str:
    """Write a whole analysis session (results + metadata) to a TSV file (format v2)."""
    conf = "NaN" if suite.confidence is None else str(suite.confidence)
    lines = [
        f"# SigmaTau StabilitySuite v{_IO_VERSION}",
        f"# package_version={_pkg_version()}",
        f"# timestamp={datetime.datetime.now().isoformat()}",
        f"# source_file={source_file}",
        f"# data_kind={suite.data_kind}",
        f"# tau0={suite.tau0}",
        f"# n={suite.n}",
        f"# confidence={conf}",
        f"# tau_mode={suite.tau_mode}",
        f"# deviations={','.join(r.deviation_type for r in suite.results)}",
    ]
    for r in suite.results:
        lines.append(f"# --- result {r.deviation_type} ---")
        _write_result_table(lines, r)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def load_suite(path: str) -> StabilitySuite:
    """Read a ``StabilitySuite`` written by :func:`save_suite` (format v2)."""
    with open(path) as f:
        lines = f.read().splitlines()
    if not (lines and "StabilitySuite" in lines[0]):
        raise ValueError(f"load_suite: {path!r} is not a StabilitySuite file — use load_result.")

    meta: dict[str, str] = {}
    delim_idxs = [i for i, ln in enumerate(lines) if ln.startswith("# --- result ")]
    for ln in lines:
        if ln.startswith("# --- result "):
            break
        m = re.match(r"^# (\w+)=(.*)$", ln)
        if m:
            meta[m.group(1)] = m.group(2).strip()
    if not delim_idxs:
        raise ValueError(f"load_suite: no result blocks found in {path!r}")

    results = []
    for b, start in enumerate(delim_idxs):
        stop = delim_idxs[b + 1] if b + 1 < len(delim_idxs) else len(lines)
        block = lines[start:stop]
        header_match = re.match(r"^# --- result (\S+) ---", block[0])
        assert header_match is not None  # delim_idxs only holds lines matching this
        dev_type = header_match.group(1)
        ci_lines = [ln for ln in block if ln.startswith("# calc_ci=")]
        has_ci = bool(ci_lines) and ci_lines[0].split("=", 1)[1].strip() == "true"
        data = [
            ln for ln in block if not ln.startswith("#") and not ln.startswith("tau") and ln.strip()
        ]
        results.append(_parse_result_rows(dev_type, has_ci, data))

    conf_raw = meta.get("confidence", "NaN")
    confidence = None if conf_raw in ("NaN", "None") else float(conf_raw)
    return StabilitySuite(
        tuple(results),
        meta.get("data_kind", "phase"),
        float(meta.get("tau0", "nan")),
        int(meta.get("n", "0")),
        confidence,
        meta.get("tau_mode", "explicit"),
    )


__all__ = ["save_result", "load_result", "save_suite", "load_suite"]
