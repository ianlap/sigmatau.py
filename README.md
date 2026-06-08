# sigmatau

Clock-stability analysis in Python — Allan/Hadamard deviations and friends, with
rich result objects. A NumPy-native port of
[SigmaTau.jl](https://github.com/ianlap/SigmaTau.jl), which serves as the
reference oracle: every kernel is validated against golden fixtures exported
from the tagged Julia release.

> **Status:** early (through milestone 3a). Implements `adev`, `mdev`, `tdev`,
> `hdev`, `mhdev`, `htdev`, plus `totdev`, `mtie`, and `pdev` — all validated
> against the Julia oracle. No confidence intervals or bias correction yet (raw
> kernels; see the roadmap). The API mirrors SigmaTau.jl one-to-one.

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

r = adev(pd)                                     # octave-spaced grid by default
print(r)                                         # StabilityResult(adev, 11 pts, τ∈[1.0, 1024.0] s, no CI)
r.tau, r.dev                                     # numpy arrays

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
| Numerics | reference | validated to `rtol = 1e-11` vs the oracle |

## Roadmap

1. ✅ Bare deviations: `adev`, `mdev`, `tdev`, `hdev`, `mhdev`, `htdev`.
2. ✅ `totdev`, `mtie`, `pdev` (raw kernels) — all pure NumPy.
3. The modified-total family: `mtotdev`, `ttotdev`, `htotdev`, `mhtotdev`.
4. Stats: EDF, confidence intervals, noise identification, bias correction
   (flips the `ci`/`correct_bias` defaults to `True`).
5. `noise_gen` calibrated power-law generator.
6. IO, spectral estimators (`Sy`/`Sx`/`L`), plotting.

Numba was originally slated to accelerate the long-record kernels, but the
milestone-2/3a kernels all vectorize cleanly in NumPy, so it isn't a dependency.
It will be reconsidered (optionally) only if a later kernel genuinely needs a
scalar loop that NumPy can't express.

## License

MIT © Ian Lapinski
