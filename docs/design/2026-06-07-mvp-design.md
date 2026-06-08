# sigmatau — MVP design (milestone 2)

Date: 2026-06-07. Status: approved, implementing.

## Context

SigmaTau.jl 0.5.0 is frozen as the reference oracle for clock-stability math.
This package is the Python port: a NumPy-native re-implementation that is easier
to use and faster on long records than allantools, with rich result objects, and
an API that mirrors the Julia oracle one-to-one. The math is validated against
golden fixtures exported from the oracle.

The full port is large (deviation kernels + stats + noise + IO + spectral +
plotting), so it is built in sub-projects. This spec covers the **first
milestone**: the bare overlapping deviations, end-to-end, with the parity
pipeline that every later cycle reuses.

## Decisions

- **Pure Python (NumPy)** kernels. The six MVP kernels are overlapping
  second/third-difference sums and vectorize cleanly in NumPy — no Numba needed
  yet. Numba enters in the next cycle for the loop-heavy kernels (total family,
  MTIE deque, PDEV recurrence). MVP runtime dependency: `numpy` only.
- **Mirror the flattened Julia `src/`**: one module per Julia file, same names
  (`types.py`, `grids.py`, `kernels.py`, `deviations.py`).
- **Vendor fixtures**: golden CSVs + shared input records are committed into this
  repo, so tests are self-contained and CI needs no Julia.
- **`ci=False` default for the MVP** — the one intentional, temporary divergence
  from Julia (which defaults `True`); flips once the EDF/CI cycle lands.

## Scope

In: scaffold; `PhaseData`/`FrequencyData`/`StabilityResult`/`StabilitySuite`;
`TauMode` + `tau_values`; `adev`/`mdev`/`tdev`/`hdev`/`mhdev`/`htdev` at
`ci=False`; Julia golden-fixture export + parity/property tests.

Out (later cycles): CI/EDF/noise-ID, total family, MTIE, PDEV, `noise_gen`, IO,
spectral, plotting, `stability` suite.

## Architecture

```
src/sigmatau/
  __init__.py        public exports
  types.py           PhaseData, FrequencyData, StabilityResult, StabilitySuite
  grids.py           TauMode + Octave/…, tau_values, _kernel_m_max, _grid,
                     _default_m_values, _freq_to_phase, _f64
  kernels.py         _adev_core, _mdev_core, _tdev_core, _hdev_core, _mhdev_core
  deviations.py      adev, mdev, tdev, hdev, mhdev, htdev
```

Core/API split mirrors Julia: `kernels.py` takes `float64` arrays → arrays;
`deviations.py` takes `PhaseData`/`FrequencyData` → `StabilityResult`. The
frequency path delegates via `_freq_to_phase` (`cumsum(y)·tau0`, length
preserved). `htdev = mhdev · τ/√(10/3)`, `tdev = mdev · τ/√3`.

### Kernel formulas (ported verbatim from `kernels.jl`)

- `adev`:  `L = N − 2m`; need `L ≥ 2` else NaN; `σ = sqrt(Σ d2² / (2·L·m²·τ0²))`,
  `d2 = x[i+2m] − 2x[i+m] + x[i]`.
- `mdev`:  prefix sums `X` (len N+1, `X[0]=0`); `Ne = N − 3m + 1 ≥ 2`;
  `d = X[i+3m] − 3X[i+2m] + 3X[i+m] − X[i]`; `σ = sqrt(Σ d² / (2·Ne·m⁴·τ0²))`.
- `tdev`:  `τ/√3 · mdev`.
- `hdev`:  `L = N − 3m ≥ 2`; `d3 = x[i+3m] − 3x[i+2m] + 3x[i+m] − x[i]`;
  `σ = sqrt(Σ d3² / (6·L·m²·τ0²))`.
- `mhdev`: `Ne = N − 4m + 1 ≥ 2`; `d = X[i+4m] − 4X[i+3m] + 6X[i+2m] − 4X[i+m] +
  X[i]`; `σ = sqrt(Σ d² / (6·Ne·m⁴·τ0²))`.
- `htdev`: `τ/√(10/3) · mhdev`.

### Grids (ported from `grids.jl`)

`_kernel_m_max(N, kernel)`: adev `(N−2)//2`, mdev/tdev `(N−1)//3`, hdev
`(N−2)//3`, mhdev/htdev `(N−1)//4`. `_grid(Octave, m_max) = [2^k for k in
0..floor(log2(m_max))]`; other modes round geometric factor powers (√2, 2^¼, 10,
√10), dedupe, clamp; `AllTaus = 1..m_max`. `_default_m_values = _grid(Octave,
_kernel_m_max)`.

## API

```python
adev(data, taus=Octave, *, ci=False, confidence=0.683) -> StabilityResult
```
`taus` accepts a `TauMode` (default `Octave`) or an explicit integer m-sequence.
Dispatch on `isinstance(data, FrequencyData)`. Validation → `ValueError`
(`tau0 > 0`, `len ≥ 2`). `StabilityResult.deviation_type` is a `str`; `tau`/`dev`
are `np.ndarray`; `noise_type`/`ci_lower`/`ci_upper`/`edf` are empty arrays when
`ci=False`. `__repr__` mirrors Julia `show`:
`StabilityResult(adev, 6 pts, τ∈[1.0, 32.0] s, no CI)`.

## Parity pipeline

`tools/export_python_fixtures.jl` (in SigmaTau.jl) runs the six deviations at
`ci=False` on fixed inputs (`stable32gen.DAT` + deterministic synthetic phase &
frequency records) over octave and all-tau grids, writing `julia_reference.csv`
(`deviation, data_kind, m, tau, dev`) plus the raw input records. Those are
vendored into `tests/fixtures/julia/`. `test_parity.py` reads the same input,
computes in Python, and asserts `np.allclose(py, julia, rtol=1e-11, atol=0)`.

## Testing

- **Parity**: every fixture row vs Julia, `rtol=1e-11`.
- **Properties**: `tdev = mdev·τ/√3`; `htdev = mhdev·τ/√(10/3)`; constant phase →
  0; frequency dispatch ≈ phase integration; under-sampled → NaN; `ci=False`
  empties.
- **Validation**: `ValueError` on bad `tau0`/length; `float64` promotion.
- **Tooling**: pytest, ruff, mypy; GitHub Actions on 3.11–3.13.
