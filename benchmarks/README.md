# Benchmarks

Three-way wall-clock comparison of overlapping deviations across:

- **allantools** (2024.06) — the established Python reference,
- **sigmatau** (this package) — NumPy port,
- **SigmaTau.jl** — the compiled Julia oracle.

All three run the same deviations on the same phase record over the same octave
averaging-factor grid; each kernel is warmed once then timed as the best of N
repetitions. Julia is timed in a subprocess with JIT paid out of band.

```bash
pip install -e ".[dev]" allantools
python benchmarks/bench.py [N] [reps]     # default N=100000, reps=5
```

Set `SIGMATAU_JL` to your SigmaTau.jl checkout (default `~/Projects/SigmaTau.jl`);
if Julia or the project is missing, those columns are skipped. Results are
written to `RESULTS.md`. Absolute times are hardware-dependent — the **ratios**
are the portable result.
