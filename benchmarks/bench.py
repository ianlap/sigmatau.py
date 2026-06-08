#!/usr/bin/env python3
"""Three-way deviation benchmark: allantools vs sigmatau (this) vs SigmaTau.jl.

All three libraries run the same overlapping deviations on the *same* phase
record over the *same* octave averaging-factor grid. Each kernel is warmed once
and timed as the best of N repetitions (wall-clock). SigmaTau.jl is timed in a
subprocess (JIT warmed out of band). Writes a Markdown table to RESULTS.md.

Usage:
    python benchmarks/bench.py [N] [reps]

Requires `allantools` (pip) and a `julia` on PATH with SigmaTau.jl available.
Set SIGMATAU_JL to the SigmaTau.jl checkout (default ~/Projects/SigmaTau.jl).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import allantools
import numpy as np

import sigmatau as st

HERE = Path(__file__).parent
JL_PROJECT = os.environ.get("SIGMATAU_JL", str(Path.home() / "Projects" / "SigmaTau.jl"))


def _best(fn, reps: int) -> float:
    fn()  # warm
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    x = np.cumsum(np.random.default_rng(0).standard_normal(n)) * 1e-9
    record = HERE / "_record.txt"
    np.savetxt(record, x)

    cap = n // 4
    ms = [2**k for k in range(int(np.floor(np.log2(cap))) + 1)]
    taus = np.array(ms, dtype=float)
    pd = st.PhaseData(x, 1.0)

    allan = {
        "adev": lambda: allantools.oadev(x, rate=1.0, data_type="phase", taus=taus),
        "mdev": lambda: allantools.mdev(x, rate=1.0, data_type="phase", taus=taus),
        "hdev": lambda: allantools.ohdev(x, rate=1.0, data_type="phase", taus=taus),
        "totdev": lambda: allantools.totdev(x, rate=1.0, data_type="phase", taus=taus),
    }
    sigp = {
        "adev": lambda: st.adev(pd, ms, ci=False),
        "mdev": lambda: st.mdev(pd, ms, ci=False),
        "hdev": lambda: st.hdev(pd, ms, ci=False),
        "totdev": lambda: st.totdev(pd, ms, ci=False, correct_bias=False),
        "pdev": lambda: st.pdev(pd, ms, ci=False),
    }

    at = {k: _best(f, reps) for k, f in allan.items()}
    sp = {k: _best(f, reps) for k, f in sigp.items()}
    jl = _run_julia(record, ms, reps)

    kernels = ["adev", "mdev", "hdev", "totdev", "pdev"]
    lines = [
        f"# Deviation benchmark — N = {n:,}, {len(ms)} octave τ (best of {reps})",
        "",
        "Wall-clock seconds per one-shot call (full τ grid). Lower is better.",
        "",
        "| kernel | allantools | sigmatau (py) | SigmaTau.jl | py vs allantools | py vs jl |",
        "|--------|-----------:|--------------:|------------:|-----------------:|---------:|",
    ]
    for k in kernels:
        a = at.get(k)
        p = sp.get(k)
        j = jl.get(k)
        a_s = f"{a * 1e3:.2f} ms" if a else "—"
        p_s = f"{p * 1e3:.2f} ms" if p else "—"
        j_s = f"{j * 1e3:.2f} ms" if j else "—"
        pva = f"{a / p:.1f}×" if (a and p) else "—"
        pvj = f"{j / p:.1f}×" if (j and p) else "—"
        lines.append(f"| {k} | {a_s} | {p_s} | {j_s} | {pva} | {pvj} |")
    lines += [
        "",
        "`py vs allantools` / `py vs jl` are speedup factors (>1 means sigmatau is",
        "faster). allantools has no parabolic deviation, so `pdev` is Python-vs-Julia",
        "only. Absolute times are hardware-dependent; the ratios are the portable result.",
    ]
    report = "\n".join(lines) + "\n"
    (HERE / "RESULTS.md").write_text(report)
    print(report)


def _run_julia(record: Path, ms: list[int], reps: int) -> dict[str, float]:
    if not Path(JL_PROJECT).is_dir():
        print(f"[warn] SigmaTau.jl not found at {JL_PROJECT}; skipping Julia timings.")
        return {}
    cmd = [
        "julia",
        f"--project={JL_PROJECT}",
        str(HERE / "bench_julia.jl"),
        str(record),
        ",".join(map(str, ms)),
        str(reps),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=900)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as err:
        print(f"[warn] Julia timing failed ({err}); skipping.")
        return {}
    timings: dict[str, float] = {}
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            timings[parts[0]] = float(parts[1])
    return timings


if __name__ == "__main__":
    main()
