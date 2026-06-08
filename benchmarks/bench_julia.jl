# bench_julia.jl — time SigmaTau.jl deviations on a shared record.
#
# Invoked by bench.py as:
#   julia --project=<SigmaTau.jl> bench_julia.jl <record.txt> <m1,m2,...> <reps>
# Reads a single-column phase record, warms each kernel (JIT out of band), then
# prints "kernel<TAB>best_seconds" for adev/mdev/hdev/totdev/pdev.

using SigmaTau
using Printf
using DelimitedFiles
using Random

path = ARGS[1]
ms   = parse.(Int, split(ARGS[2], ","))
reps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5

x  = vec(readdlm(path, Float64))
pd = PhaseData(x, 1.0)

warm   = PhaseData(cumsum(randn(MersenneTwister(0), 2048)) .* 1e-9, 1.0)
warm_m = [1, 2, 4, 8]

run(fn, data, m; bias=false) =
    fn === totdev ? fn(data, m; ci=false, correct_bias=bias) : fn(data, m; ci=false)

for (name, fn) in (("adev", adev), ("mdev", mdev), ("hdev", hdev),
                   ("totdev", totdev), ("pdev", pdev))
    run(fn, warm, warm_m)                      # warm-start
    best = Inf
    for _ in 1:reps
        best = min(best, @elapsed run(fn, pd, ms))
    end
    @printf("%s\t%.6e\n", name, best)
end
