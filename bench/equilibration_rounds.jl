# The cost and tightness improvement of the balancing alternation.
#
# `balance_cover` is an unbounded, data-dependent iteration: it alternates until the per-round
# product change falls below `rtol`, and errors when it does not. Its round count is therefore a
# property of the problem, not of the code, and any quoted cost has to carry it.
#
# Everything is scored as *excess* — `max_kj (a_k*b_j)/Ũ_kj` over the nonzero entries, the
# quantity `jacobian_minimax` minimizes. That makes three references directly comparable on one
# scale: the unbalanced pair (`a = ones`, one merge b-step), the balanced pair, and the exact
# rank-1 optimum from `jacobian_minimax`. `jacobian_representability` reports the 2-cycle lower
# bound alongside it.

using DerivativeCovers, LinearAlgebra, Random, Statistics, Printf
using DerivativeCovers: jacobian_surrogate, balance_cover, jacobian_minimax,
    jacobian_representability, tape_function, MergeRule, SumMerge, QuadMerge, MaxMerge

const MERGES = (SumMerge(), QuadMerge(), MaxMerge())
const ARULES = (:marginal, :geomean)
const STATUSES = (:converged, :diverging, :stalled, :overbudget, :zeroweight)

"""
    excess(U, a, b) -> e

Worst-case looseness `max_kj (a_k*b_j)/U_kj` over the nonzero entries of `U`: the reciprocal view
of `jacobian_ratio`, and the quantity [`jacobian_minimax`](@ref) minimizes.
"""
excess(U, a, b) = maximum(a[k] * b[j] / U[k, j] for k in axes(U, 1), j in axes(U, 2) if U[k, j] > 0)

"One b-step of `merge` from `a = ones`: the unbalanced pair `cover_jacobian` returns."
bstep_from_ones(U, ::SumMerge) = vec(sum(U; dims = 1))
bstep_from_ones(U, ::QuadMerge) = sqrt.(vec(sum(abs2, U; dims = 1)))
bstep_from_ones(U, ::MaxMerge) = vec(maximum(U; dims = 1))

"""
    attempt(U, merge, arule; kwargs...) -> (; status, ...)

Run `balance_cover` and score the pair it returns. Reaching a fixed point is one outcome among
four, so the three failures are classified rather than raised:

- `:converged` — carries `iters`, `delta` and `excess`;
- `:diverging` — the weights are separating without bound, so a longer run returns a worse cover;
- `:stalled` — `delta` plateaued above `rtol`, the limit cycle a larger budget cannot fix;
- `:overbudget` — `maxiter` rounds were not enough, `delta` still falling.

A fourth, `:zeroweight`, is drift left to run past the floating-point exponent range until a step
zeroes the weight of a live output or input. The divergence check reports drift long before that,
so it stands as a guard rather than an expected outcome.

The failures carry `msg`, truncated at the first `;` to drop the remedy hint.
"""
function attempt(U, merge::MergeRule, arule::Symbol; kwargs...)
    return try
        r = balance_cover(U; merge, arule, kwargs...)
        (; status = :converged, r.iters, r.delta, excess = excess(U, r.a, r.b))
    catch e
        e isa ErrorException && return (;
            status = occursin("stalled", e.msg) ? :stalled :
                occursin("diverged", e.msg) ? :diverging : :overbudget,
            msg = String(first(split(e.msg, ";"))),
        )
        e isa ArgumentError && occursin("is zero on live", e.msg) &&
            return (; status = :zeroweight, msg = String(first(split(e.msg, ";"))))
        rethrow()
    end
end

"""
    fixture_table(name, U)

Excess of the unbalanced and balanced pairs for every `(arule, merge)` combination, against the
exact optimum. `:geomean` fits `log Ũ_kj ≈ log a_k + log b_j` and applies the merge only in the
closing b-step, so its round count is the same for all three merges.
"""
function fixture_table(name, U)
    m, n = size(U)
    mm = jacobian_minimax(U)
    @printf(
        "--- %s (%d x %d, %d nonzeros) ---\n", name, m, n, count(>(0), U)
    )
    @printf(
        "minimax excess %.4g, representability bound %.4g\n",
        mm.excess, jacobian_representability(U)
    )
    @printf("%-10s %-9s %11s %8s %11s %11s\n", "merge", "arule", "unbalanced", "iters", "balanced", "/minimax")
    for merge in MERGES
        e0 = excess(U, ones(m), bstep_from_ones(U, merge))
        for arule in ARULES
            r = attempt(U, merge, arule)
            if r.status === :converged
                @printf(
                    "%-10s %-9s %11.4g %8d %11.4g %11.3f\n",
                    nameof(typeof(merge)), arule, e0, r.iters, r.excess, r.excess / mm.excess
                )
            else
                @printf(
                    "%-10s %-9s %11.4g %8s %11s %11s   %s: %s\n",
                    nameof(typeof(merge)), arule, e0, "-", "-", "-", r.status, r.msg
                )
            end
        end
    end
    return println()
end

"""
    rtol_curve(name, U, rtols)

Rounds and excess as the convergence tolerance tightens. `rtol` is the only public handle on where
the alternation stops, so this is the cost-vs-tightness trade a caller can actually make. The
alternation optimizes the fixed point, not the excess, so the curve need not be monotone: an
intermediate iterate can be tighter than the pair the iteration converges to.
"""
function rtol_curve(name, U, rtols; merge::MergeRule = SumMerge(), arule::Symbol = :marginal, maxiter = 200_000)
    runs = [attempt(U, merge, arule; rtol = rt, maxiter) for rt in rtols]
    @printf("%-26s %-10s %-9s\n", name, nameof(typeof(merge)), arule)
    @printf("%16s", "rtol:")
    for rt in rtols
        @printf(" %9.0e", rt)
    end
    println()
    @printf("%16s", "iters")
    for r in runs
        r.status === :converged ? @printf(" %9d", r.iters) : @printf(" %9s", r.status)
    end
    println()
    @printf("%16s", "excess")
    for r in runs
        r.status === :converged ? @printf(" %9.4g", r.excess) : @printf(" %9s", "-")
    end
    println()
    return println()
end

"""
    pattern_survey(; nsamp, m, n, density, seed) -> nothing

Convergence outcome and tightness over random sparse magnitude patterns, at default settings.
Patterns with an all-zero row or column are skipped: those have an exact zero weight and balance
trivially. Tightness is compared only on the patterns where both a-rules converge, so the two
columns are paired.
"""
function pattern_survey(; nsamp = 200, m = 6, n = 8, density = 0.5, seed = 1)
    println("--- random sparse patterns ($m x $n, density $density, $nsamp draws, default settings) ---")
    @printf(
        "%-10s %-9s %7s %7s %7s %11s %11s %8s %8s %10s\n",
        "merge", "arule", "conv", "drift", "stall", "overbudget", "zeroweight",
        "med it", "max it", "/minimax"
    )
    paired = Dict(nameof(typeof(mg)) => Float64[] for mg in MERGES)
    for merge in MERGES, arule in ARULES
        tally = Dict(s => 0 for s in STATUSES)
        its, ratios = Int[], Float64[]
        for s in 1:nsamp
            U = _random_pattern(m, n, density, seed + 1000s)
            U === nothing && continue
            r = attempt(U, merge, arule)
            tally[r.status] += 1
            r.status === :converged || continue
            push!(its, r.iters)
            push!(ratios, r.excess / jacobian_minimax(U).excess)
        end
        @printf(
            "%-10s %-9s %7d %7d %7d %11d %11d %8s %8s %10s\n",
            nameof(typeof(merge)), arule, tally[:converged], tally[:diverging],
            tally[:stalled], tally[:overbudget], tally[:zeroweight],
            isempty(its) ? "-" : string(round(Int, median(its))),
            isempty(its) ? "-" : string(maximum(its)),
            isempty(ratios) ? "-" : @sprintf("%.3f", median(ratios))
        )
    end
    for merge in MERGES
        for s in 1:nsamp
            U = _random_pattern(m, n, density, seed + 1000s)
            U === nothing && continue
            rm, rg = attempt(U, merge, :marginal), attempt(U, merge, :geomean)
            (rm.status === :converged && rg.status === :converged) || continue
            push!(paired[nameof(typeof(merge))], rm.excess / rg.excess)
        end
    end
    println()
    println("marginal/geomean excess on the patterns where both converge (>1 favors :geomean):")
    for merge in MERGES
        v = paired[nameof(typeof(merge))]
        @printf(
            "%-10s n=%4d  median %6.3f  min %6.3f  max %8.3f\n",
            nameof(typeof(merge)), length(v), median(v), minimum(v), maximum(v)
        )
    end
    return println()
end

function _random_pattern(m, n, density, seed)
    rng = MersenneTwister(seed)
    U = abs.(randn(rng, m, n))
    U[rand(rng, m, n) .> density] .= 0
    all(any(>(0), U; dims = 2)) && all(any(>(0), U; dims = 1)) || return nothing
    return U
end

"""
    failure_modes()

Minimal patterns for two of the ways the marginal alternation fails to reach a fixed point, and
what `:geomean` returns on each. Neither failure is a failure of the cover: every marginal iterate
is valid, so the pair that would have been returned is sound and only the stopping test refuses it.
"""
function failure_modes()
    println("--- non-convergence, and the :geomean pair on the same pattern ---")
    cases = (
        "drift (no total support)" => ([1.0 1.0; 0.0 1.0], SumMerge()),
        "limit cycle (max merge)" => ([1.0 1.0 1.0; 1.0 1.0 0.0; 0.0 0.0 1.0], MaxMerge()),
    )
    for (name, (U, merge)) in cases
        r = attempt(U, merge, :marginal; maxiter = 5000)
        g = attempt(U, merge, :geomean)
        @printf(
            "%-26s marginal: %s\n", name,
            r.status === :converged ? "converged in $(r.iters), excess $(r.excess)" : r.msg
        )
        @printf(
            "%-26s geomean:  iters %d, excess %.4g (minimax %.4g)\n\n",
            "", g.iters, g.excess, jacobian_minimax(U).excess
        )
    end
    return nothing
end

"Dense two-layer map, isotropic output scales."
mlp_flat(W1, W2) = x -> W2 * tanh.(W1 * x)

"Dense two-layer map whose output scales span four decades."
mlp_spread(W1, W2, s) = x -> (s .* (W2 * tanh.(W1 * x)))

"Banded map: output `i` depends on inputs `i` and `i+1` only."
chain(x) = [tanh(x[i]) * x[i + 1] for i in 1:(length(x) - 1)]

function surrogate_of(f, x)
    t, outidxs = tape_function(f, x)
    return jacobian_surrogate(t, outidxs)
end

function main()
    rng = MersenneTwister(7)
    W1, W2 = randn(rng, 12, 8), randn(rng, 6, 12)
    x0 = 0.5 .* randn(rng, 8)
    s = exp10.(range(-2, 2; length = 6))
    xc = 0.7 .* randn(MersenneTwister(3), 16)

    fixtures = (
        "dense MLP" => surrogate_of(mlp_flat(W1, W2), x0),
        "dense MLP, spread outputs" => surrogate_of(mlp_spread(W1, W2, s), x0),
        "chain" => surrogate_of(chain, xc),
    )

    println("Excess is max_kj (a_k*b_j)/Utilde_kj: 1 is exact on the surrogate, larger is looser.")
    println()
    for (name, U) in fixtures
        fixture_table(name, U)
    end

    rtols = (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10)
    println("--- rounds against tolerance ---")
    for (name, U) in fixtures, merge in (SumMerge(), MaxMerge())
        rtol_curve(name, U, rtols; merge)
    end

    pattern_survey()
    return failure_modes()
end

main()
