# Deviation benchmark — allantools vs sigmatau (py) vs SigmaTau.jl

Wall-clock seconds per one-shot call (full τ grid). Lower is better. `py vs allantools` / `py vs jl` are speedup factors (>1 = sigmatau faster). allantools has no `pdev`/`mhtotdev`. Absolute times are hardware-dependent; the ratios are the portable result.

## Fast (O(N)) kernels — N = 100,000, 15 octave τ (best of 3)

| kernel | allantools | sigmatau (py) | SigmaTau.jl | py vs allantools | py vs jl |
|--------|-----------:|--------------:|------------:|-----------------:|---------:|
| adev | 2.48 ms | 1.73 ms | 0.28 ms | 1.4× | 0.16× |
| mdev | 7.59 ms | 2.75 ms | 0.79 ms | 2.8× | 0.29× |
| hdev | 3.05 ms | 2.88 ms | 0.56 ms | 1.1× | 0.19× |
| totdev | 5.95 ms | 9.02 ms | 0.51 ms | 0.7× | 0.06× |
| pdev | — | 30.23 ms | 2.79 ms | — | 0.09× |

## Modified-total family — N = 8,000, 11 octave τ (best of 3)

| kernel | allantools | sigmatau (py) | SigmaTau.jl | py vs allantools | py vs jl |
|--------|-----------:|--------------:|------------:|-----------------:|---------:|
| mtotdev | 117483.22 ms | 2703.06 ms | 202.32 ms | 43.5× | 0.07× |
| htotdev | 128322.43 ms | 2687.68 ms | 185.22 ms | 47.7× | 0.07× |
| mhtotdev | — | 1307.04 ms | 150.43 ms | — | 0.12× |
