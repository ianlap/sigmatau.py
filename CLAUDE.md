# sigmatau (Python) — agent context

Python port of [SigmaTau.jl](https://github.com/ianlap/SigmaTau.jl), authored by
Ian Lapinski. The Julia package is the **reference oracle**: its tagged release
defines the trusted numerical results, and this package is validated against
golden fixtures exported from it.

## Authorship and attribution rules

- All changes are authored by Ian. Do not add yourself as a co-author or
  attribute work to "Claude" or "AI" anywhere.
- Do not add "Co-authored-by: Claude" or similar trailers to commits.
- Do not add "Generated with Claude Code" footers, signatures, or comments to
  code, commit messages, PRs, changelogs, or docs.
- Commit messages, CHANGELOG entries, and code comments are written in Ian's
  voice. No first-person from you. No "# added by AI" comments.

## What this package is

A NumPy-native re-implementation of SigmaTau's validated clock-stability math —
easier to use and faster on long records than allantools, with richer result
objects. The API deliberately **mirrors the Julia oracle** one-to-one (same
names: `stability`, `devs`, `taus`, `ci`, `tau0`, `confidence`; same result
fields), differing only in Python casing/conventions.

## Architecture — mirrors the flattened Julia `src/`

One Python module per Julia source file, same names:

- `src/sigmatau/types.py`      ← `types.jl`      — `PhaseData`, `FrequencyData`,
  `StabilityResult`, `StabilitySuite` (frozen dataclasses; `tau0` defaults to 1.0).
- `src/sigmatau/grids.py`      ← `grids.jl`      — `TauMode` + `Octave`/…,
  `tau_values`, `_kernel_m_max`, `_grid`, `_default_m_values`, `_freq_to_phase`, `_f64`.
- `src/sigmatau/kernels.py`    ← `kernels.jl`    — internal `_*_core` array
  kernels (plain `np.ndarray` in, `np.ndarray` out). NumPy-vectorized.
- `src/sigmatau/deviations.py` ← `deviations.jl` — public deviations
  (`PhaseData`|`FrequencyData` → `StabilityResult`).
- `src/sigmatau/edf.py`        ← `edf.jl`        — EDF, bias correction, CIs (scipy χ²).
- `src/sigmatau/noise.py`      ← `noise.jl`      — `identify_noise` + `noise_gen`.
- `src/sigmatau/spectral.py`   ← `spectral.jl`   — Welch PSD + `Sy`/`Sx`/`L`.
- `src/sigmatau/suite.py`      ← `suite.jl`      — `stability` compute-all.
- `src/sigmatau/io/`           ← `io/`           — `read_phase`/`read_frequency`,
  `detrend`, `fillgaps`, `save_result`/`load_result`/`save_suite`/`load_suite`.

Only the plotting helpers (the RecipesBase extension equivalent) remain to port.

### Critical conventions — do not violate

- Keep the **core / API split** firm, exactly like Julia: `_*_core` take plain
  `float64` arrays and return arrays; public functions take `PhaseData` /
  `FrequencyData` and return `StabilityResult`. Never collapse the two layers.
- `StabilityResult` CI fields (`noise_type`, `ci_lower`, `ci_upper`, `edf`) are
  **empty arrays** when `ci=False`. Preserve this contract.
- Inputs are promoted to contiguous `float64` at the API boundary (`_f64`).

## Validation — the parity contract

`tests/fixtures/julia/` holds golden CSVs exported from the Julia oracle plus the
**shared input records** (so Python reads the identical input Julia did — RNGs
differ between languages, so inputs must be shared, not regenerated). Since both
use Float64 and the same algorithm, parity holds tight: `rtol = 1e-11`.

Regenerate fixtures by running `tools/export_python_fixtures.jl` in the SigmaTau.jl
repo and copying its output here.

## Status / roadmap

Through milestone 4: all 13 deviations with full statistics — `adev`, `mdev`,
`tdev`, `hdev`, `mhdev`, `htdev`, `totdev`, `mtotdev`, `ttotdev`, `htotdev`,
`mhtotdev`, `mtie`, `pdev`. `ci=True` (default) reports per-τ noise type,
equivalent degrees of freedom, and χ²-based confidence intervals;
`correct_bias=True` (default) applies the SP1065/FCS bias correction to the
total family. Defaults now match the Julia oracle exactly — no temporary
divergences remain. `mtie` has no CI model (`ci`/`confidence` are no-ops).

Stats live in `noise.py` (`identify_noise` + lag-1/B1 helpers) and `edf.py`
(`calculate_edf`, `bias_correction`, `confidence_intervals`; `scipy.stats` for
χ²/normal quantiles). The modified-total family uses a per-subsequence loop with
vectorized inner reductions (parity exported on the synthetic N=1024 records,
octave grid).

Parity tolerances: deviation/EDF `rtol=1e-11`, noise type exact, CI `rtol=1e-9`
(scipy vs Distributions.jl quantiles).

Temporary divergences from Julia (all flip when the stats cycle lands):
- `ci` defaults to **False** (Julia: `True`); `ci=True` raises `NotImplementedError`
  except on `mtie`, where `ci`/`confidence` are genuine no-ops (no EDF model).
- `totdev` `correct_bias` defaults to **False** (Julia: `True`); `correct_bias=True`
  raises `NotImplementedError` (the SP1065 unbias correction needs noise-ID).

Remaining: modified-total family (`mtotdev`/`ttotdev`/`htotdev`/`mhtotdev`) →
EDF/CI/noise-ID/bias → `noise_gen` → IO → spectral → plotting.

**On Numba:** not a dependency. Numba only accelerates explicit scalar loops;
every kernel so far vectorizes in NumPy (and `llvmlite` has no wheel for current
Python here anyway). Reconsider it — surgically, never package-wide — only if a
future kernel needs a scalar loop NumPy can't express.

## Testing

- `pytest` — parity tests (vs `julia_reference.csv`) + property/identity tests.
- `ruff check` / `ruff format` — lint + format.
- `mypy` — types.
