# sigmatau — milestone 4 design (EDF, CI, noise-ID, bias)

Date: 2026-06-07. Status: approved, implementing.

## Context

Through milestone 3 all 13 deviations are ported as raw kernels (`ci=False`,
`correct_bias=False`, both raising `NotImplementedError` if requested). Milestone
4 ports the statistics layer so the deviations finally report error bars and
match the Julia oracle's defaults exactly. Decision (user): do it as **one
slice** — noise identification + all EDF + bias correction + confidence
intervals together — and flip `ci`/`correct_bias` to `True` across the board.

## Components (mirror the Julia stats stack)

Dependency chain: `identify_noise → α → (calculate_edf, bias_correction) →
confidence_intervals`.

- **`noise.py`** ← `noise.jl`: `identify_noise(x, m_values, *, dmin=0, dmax=2,
  detrend=True)` (public, mirroring Julia's export) + privates `_preprocess`
  (5σ outlier filter), `_noise_id_lag1acf`, `_lag1_acf`, `_noise_id_b1rn`,
  `_b1_theory`, `_rn_theory`, `_detrend_quadratic`, `_simple_avar`,
  `_simple_mdev`; `NEFF_RELIABLE = 30`. Per τ: lag-1 ACF when `N/m ≥ 30`, else
  the B1/R(n) fallback; α mapped to `"WHPM"/"FLPM"/"WHFM"/"FLFM"/"RWFM"`; an
  unreliable τ inherits the last reliable symbol (`last_reliable` carry-forward).
  (`noise_gen` synthesis stays milestone 5.)
- **`edf.py`** ← `edf.jl`: `calculate_edf(method, devs, noises, m_values, taus,
  n, t)`, `_calc_edf_core` (Greenhall–Riley) + `_compute_sw/_sx/_sz` spectral
  integrals, coefficient tables `_coeff_totvar/_mtot/_mhtot/_htot`, `_pvar_A`/
  `_pvar_edf`, `_kn_from_alpha`, `_alpha_from_noise`, `bias_correction(noises,
  var_type, taus, t)`, `confidence_intervals(devs, edfs, noises, n, confidence)`.

`_alpha_from_noise` maps the noise-type strings to α ∈ {2,1,0,−1,−2}.

## Dependency

Add **`scipy`** (runtime). `confidence_intervals` uses `scipy.stats.chi2.ppf`
and `scipy.stats.norm.ppf` where Julia uses `Distributions.jl`
(`quantile(Chisq(ν), p)`, `quantile(Normal(), p)`).

## Wiring into `deviations.py`

The `ci=True` path becomes: `noises = identify_noise(x, m, dmin=…, dmax=…)`
(per-kernel `dmax`: 2 for the Allan/total/pdev family, 3 for Hadamard) →
`edfs = calculate_edf(method, devs, noises, m, taus, N, T)` → `lower, upper =
confidence_intervals(devs, edfs, noises, N, confidence)`. `T = (N − 1)·τ₀`. The
total family additionally applies `devs /= sqrt(bias_correction(noises,
var_type, taus, T))` (with the `_unbias_divisor` non-positive→NaN guard) when
`correct_bias=True`. The derived wrappers (`tdev`, `htdev`, `ttotdev`) scale the
CI bounds by the same τ-factor as the deviation, matching Julia.

Result fields: `noise_type` becomes a string array; `ci_lower`/`ci_upper`/`edf`
are populated. `StabilityResult.__repr__` shows "with CI".

**Defaults flip to match the oracle:** `ci=True` for every CI-bearing deviation
(`adev mdev tdev hdev mhdev htdev totdev mtotdev ttotdev htotdev mhtotdev pdev`);
`correct_bias=True` for the total family; `mtie` unchanged (`ci`/`confidence`
remain no-ops). The `NotImplementedError`s are removed.

## Parity strategy & tolerances

Regenerate fixtures with `ci=true` (now the default) from the oracle, adding
`noise_type, edf, ci_lower, ci_upper` columns (same records/grids as before;
modified-total family on the N=1024 synth records).

- **noise_type (α): exact string match** — the heuristic must reproduce
  identically.
- **edf: `rtol = 1e-11`** (pure arithmetic).
- **ci_lower/ci_upper: `rtol = 1e-9`** — `scipy` vs `Distributions.jl` χ²-quantile
  algorithms differ slightly; tighten if it passes tighter.

The deviation-value (`dev`) columns now reflect bias correction for the total
family (`correct_bias=True`), so existing `dev` parity rows update accordingly.

## Testing

- Parity: every fixture row — `noise_type` exact, `edf` 1e-11, `dev`/`ci` per
  tolerances above.
- Properties: `lower ≤ dev ≤ upper`; `edf > 0` where finite; `noise_type` ∈ the
  five power-law symbols; Gaussian-fallback branch when `edf < 1` (short record,
  high `m`); `ci=False` still yields empty CI fields; `mtie` CI still a no-op.
- Tooling: pytest, ruff, mypy; CI on 3.11–3.13.

## Main risk

`identify_noise` is the long pole — a multi-branch heuristic (ACF-vs-B1 at
`N/m = 30`, per-τ quadratic detrend, `last_reliable` carry). Any α mismatch
cascades into that τ's EDF/CI. Port it line-for-line and confirm α parity before
trusting the CI numbers.
