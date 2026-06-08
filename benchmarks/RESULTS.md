# Deviation benchmark — N = 100,000, 15 octave τ (best of 5)

Wall-clock seconds per one-shot call (full τ grid). Lower is better.

| kernel | allantools | sigmatau (py) | SigmaTau.jl | py vs allantools | py vs jl |
|--------|-----------:|--------------:|------------:|-----------------:|---------:|
| adev | 2.48 ms | 1.71 ms | 0.29 ms | 1.4× | 0.2× |
| mdev | 7.52 ms | 2.74 ms | 0.82 ms | 2.7× | 0.3× |
| hdev | 3.23 ms | 2.48 ms | 0.36 ms | 1.3× | 0.1× |
| totdev | 5.39 ms | 8.82 ms | 0.44 ms | 0.6× | 0.0× |
| pdev | — | 31.18 ms | 2.72 ms | — | 0.1× |

`py vs allantools` / `py vs jl` are speedup factors (>1 means sigmatau is
faster). allantools has no parabolic deviation, so `pdev` is Python-vs-Julia
only. Absolute times are hardware-dependent; the ratios are the portable result.
