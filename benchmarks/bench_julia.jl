# bench_julia.jl — time SigmaTau.jl deviations on a shared record.
#
# Invoked by bench.py as:
#   julia --project=<SigmaTau.jl> bench_julia.jl <record.txt> <m1,m2,...> <reps> <k1,k2,...>
# Reads a single-column phase record, warms each requested kernel (JIT out of
# band), then prints "kernel<TAB>best_seconds".

using SigmaTau
using Printf
using DelimitedFiles
using Random

path    = ARGS[1]
ms      = parse.(Int, split(ARGS[2], ","))
reps    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5
kernels = length(ARGS) >= 4 ? split(ARGS[4], ",") : ["adev", "mdev", "hdev", "totdev", "pdev"]

const FNS = Dict(
    "adev" => adev, "mdev" => mdev, "hdev" => hdev, "tdev" => tdev,
    "mhdev" => mhdev, "htdev" => htdev, "totdev" => totdev, "mtotdev" => mtotdev,
    "ttotdev" => ttotdev, "htotdev" => htotdev, "mhtotdev" => mhtotdev,
    "mtie" => mtie, "pdev" => pdev,
)
const TOTAL = Set(["totdev", "mtotdev", "ttotdev", "htotdev", "mhtotdev"])

x  = vec(readdlm(path, Float64))
pd = PhaseData(x, 1.0)

warm   = PhaseData(cumsum(randn(MersenneTwister(0), 2048)) .* 1e-9, 1.0)
warm_m = [1, 2, 4, 8]

run(name, fn, data, m) =
    name in TOTAL ? fn(data, m; ci=false, correct_bias=false) : fn(data, m; ci=false)

for name in kernels
    fn = FNS[name]
    run(name, fn, warm, warm_m)                # warm-start
    best = Inf
    for _ in 1:reps
        best = min(best, @elapsed run(name, fn, pd, ms))
    end
    @printf("%s\t%.6e\n", name, best)
end
