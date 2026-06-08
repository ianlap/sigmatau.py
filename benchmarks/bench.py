#!/usr/bin/env python3
"""Three-way deviation benchmark: allantools vs sigmatau (this) vs SigmaTau.jl.

All three libraries run the same deviations on the same phase record over the
same octave averaging-factor grid; each kernel is warmed once and timed as the
best of N repetitions. SigmaTau.jl runs in a subprocess (JIT paid out of band).

Two tables: the fast O(N) kernels at a large N, and the heavier modified-total
family at a smaller N (those are O(N·m) in every library). Writes RESULTS.md.

Usage:
    python benchmarks/bench.py [N] [reps] [N_modtotal]

Requires `allantools` (pip) and a `julia` on PATH with SigmaTau.jl available
(set SIGMATAU_JL; default ~/Projects/SigmaTau.jl).
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

FAST = ["adev", "mdev", "hdev", "totdev", "pdev"]
MODTOTAL = ["mtotdev", "htotdev", "mhtotdev"]


def _best(fn, reps: int) -> float:
    fn()  # warm
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _allan_calls(x, taus):
    return {
        "adev": lambda: allantools.oadev(x, rate=1.0, data_type="phase", taus=taus),
        "mdev": lambda: allantools.mdev(x, rate=1.0, data_type="phase", taus=taus),
        "hdev": lambda: allantools.ohdev(x, rate=1.0, data_type="phase", taus=taus),
        "totdev": lambda: allantools.totdev(x, rate=1.0, data_type="phase", taus=taus),
        "mtotdev": lambda: allantools.mtotdev(x, rate=1.0, data_type="phase", taus=taus),
        "htotdev": lambda: allantools.htotdev(x, rate=1.0, data_type="phase", taus=taus),
    }


def _sig_calls(pd, ms):
    tot = {"correct_bias": False}
    return {
        "adev": lambda: st.adev(pd, ms, ci=False),
        "mdev": lambda: st.mdev(pd, ms, ci=False),
        "hdev": lambda: st.hdev(pd, ms, ci=False),
        "pdev": lambda: st.pdev(pd, ms, ci=False),
        "totdev": lambda: st.totdev(pd, ms, ci=False, **tot),
        "mtotdev": lambda: st.mtotdev(pd, ms, ci=False, **tot),
        "htotdev": lambda: st.htotdev(pd, ms, ci=False, **tot),
        "mhtotdev": lambda: st.mhtotdev(pd, ms, ci=False, **tot),
    }


def _run_julia(record: Path, ms: list[int], reps: int, kernels: list[str]) -> dict[str, float]:
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
        ",".join(kernels),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as err:
        print(f"[warn] Julia timing failed ({err}); skipping.")
        return {}
    timings: dict[str, float] = {}
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            timings[parts[0]] = float(parts[1])
    return timings


def _table(title: str, n: int, reps: int, names: list[str]) -> str:
    x = np.cumsum(np.random.default_rng(0).standard_normal(n)) * 1e-9
    record = HERE / "_record.txt"
    np.savetxt(record, x)
    ms = [2**k for k in range(int(np.floor(np.log2(n // 4))) + 1)]
    taus = np.array(ms, dtype=float)
    pd = st.PhaseData(x, 1.0)

    ac, sc = _allan_calls(x, taus), _sig_calls(pd, ms)
    at = {k: _best(ac[k], reps) for k in names if k in ac}
    sp = {k: _best(sc[k], reps) for k in names if k in sc}
    jl = _run_julia(record, ms, reps, names)

    def fmt(v: float | None) -> str:
        return f"{v * 1e3:.2f} ms" if v else "—"

    lines = [
        f"## {title} — N = {n:,}, {len(ms)} octave τ (best of {reps})",
        "",
        "| kernel | allantools | sigmatau (py) | SigmaTau.jl | py vs allantools | py vs jl |",
        "|--------|-----------:|--------------:|------------:|-----------------:|---------:|",
    ]
    for k in names:
        a, p, j = at.get(k), sp.get(k), jl.get(k)
        pva = f"{a / p:.1f}×" if (a and p) else "—"
        pvj = f"{j / p:.2f}×" if (j and p) else "—"
        lines.append(f"| {k} | {fmt(a)} | {fmt(p)} | {fmt(j)} | {pva} | {pvj} |")
    return "\n".join(lines)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_mt = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000

    blocks = [
        _table("Fast (O(N)) kernels", n, reps, FAST),
        _table("Modified-total family", n_mt, reps, MODTOTAL),
    ]
    report = (
        "# Deviation benchmark — allantools vs sigmatau (py) vs SigmaTau.jl\n\n"
        "Wall-clock seconds per one-shot call (full τ grid). Lower is better. "
        "`py vs allantools` / `py vs jl` are speedup factors (>1 = sigmatau faster). "
        "allantools has no `pdev`/`mhtotdev`. Absolute times are hardware-dependent; "
        "the ratios are the portable result.\n\n" + "\n\n".join(blocks) + "\n"
    )
    (HERE / "RESULTS.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
