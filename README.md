# sigmatau

Clock-stability analysis in Python — Allan/Hadamard deviations and friends, with
rich result objects. A NumPy-native port of
[SigmaTau.jl](https://github.com/ianlap/SigmaTau.jl), which serves as the
reference oracle: every kernel is validated against golden fixtures exported
from the tagged Julia release.

> **Status:** all 13 deviations with full statistics (through milestone 4) —
> `adev`, `mdev`, `tdev`, `hdev`, `mhdev`, `htdev`, `totdev`, `mtotdev`,
> `ttotdev`, `htotdev`, `mhtotdev`, `mtie`, `pdev`, each reporting per-τ noise
> type, equivalent degrees of freedom, and χ²-based confidence intervals (and
> bias correction for the total family). Every value is validated against the
> Julia oracle. The API mirrors SigmaTau.jl one-to-one, defaults included.

## Install

```bash
pip install -e ".[dev]"   # from a checkout; only runtime dep is numpy
```

## Usage

```python
import numpy as np
from sigmatau import PhaseData, FrequencyData, Octave, adev, mdev, hdev

x = np.cumsum(np.random.randn(4096)) * 1e-9      # phase residuals (seconds)
pd = PhaseData(x)                                # tau0 defaults to 1.0 s

r = adev(pd)                                     # octave grid, ci=True by default
print(r)                                         # StabilityResult(adev, 11 pts, τ∈[1.0, 1024.0] s, with CI)
r.tau, r.dev, r.edf                              # numpy arrays
r.noise_type, r.ci_lower, r.ci_upper            # per-τ noise type + χ² confidence bounds
adev(pd, ci=False).edf                           # opt out of CI -> empty arrays

adev(pd, Octave)                                 # explicit grid mode
adev(pd, [1, 2, 4, 8])                           # explicit averaging factors

fd = FrequencyData(np.random.randn(4096) * 1e-12)
hdev(fd)                                         # frequency input also works
```

## Relationship to SigmaTau.jl

| | SigmaTau.jl (oracle) | sigmatau (this) |
|---|---|---|
| `stability` / `devs` / `taus` / `ci` / `tau0` / `confidence` | ✓ | ✓ (mirrored) |
| Result fields `tau`, `dev`, `noise_type`, `ci_lower`, `ci_upper`, `edf` | ✓ | ✓ |
| Numerics | reference | deviations/EDF to `rtol = 1e-11`, noise type exact, CI to `1e-9` vs the oracle |

## Roadmap

1. ✅ Bare deviations: `adev`, `mdev`, `tdev`, `hdev`, `mhdev`, `htdev`.
2. ✅ `totdev`, `mtie`, `pdev`.
3. ✅ Modified-total family: `mtotdev`, `ttotdev`, `htotdev`, `mhtotdev`.
4. ✅ Stats: noise identification, EDF, confidence intervals, bias correction
   (`ci`/`correct_bias` default to `True`, matching the oracle).
5. `noise_gen` calibrated power-law generator.
6. IO, spectral estimators (`Sy`/`Sx`/`L`), plotting.

Numba was originally slated to accelerate the long-record kernels, but every
kernel so far vectorizes in NumPy (the modified-total family uses a
per-subsequence loop with vectorized inner reductions), so it isn't a
dependency. It will be reconsidered (optionally, never package-wide) only if a
future kernel genuinely needs a scalar loop NumPy can't express.

## License

MIT © Ian Lapinski
