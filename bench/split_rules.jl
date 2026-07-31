# What choosing a split rule is worth, in each tangent channel.
#
# Some second-derivative blocks admit a one-parameter family of covers, and a `SplitRule` picks one
# member of it from a scale reference. Two of the three rules are exact optima of something:
# `SensitivitySplit` minimizes the sum merge, `VectorNormSplit` the quadrature merge. `ValueSplit`
# minimizes nothing and is the default, because parent values are the only reference an `O(L)`
# sweep already has -- a sensitivity costs an extra sweep and a tangent norm costs `O(n*L)`.
#
# Each rule is scored in both tangent channels, since the merge a rule optimizes is the merge as
# performed over some channel, and the channel is now a separate choice. Looseness is
# `1/cover_ratio` against ForwardDiff's Hessian: 1 is exact, larger is looser.
#
# Two things the table cannot show on its own are reported beside it. A per-node optimum is not a
# global one -- the rules are compared against the best split found by sweeping the family, so a
# rule can be beaten -- and a scale reference can be degenerate at the point of evaluation, which
# looseness at that point does not reveal and a jitter of the point does. The four
# The `OptimizationProblems` cases are here for the second reason: their covers are tight under
# every rule and it is the stability of the split that separates them.

using ADNLPModels, DerivativeCovers, DerivativeCoverProblems, ForwardDiff, LinearAlgebra,
    OptimizationProblems, Random, Printf, Statistics
using DerivativeCovers: cover_ratio, value

const RULES = (
    "value" => ValueSplit(),
    "sensitivity" => SensitivitySplit(),
    "norm" => VectorNormSplit(),
)

const CHANNELS = ("absolute" => AbsoluteTangents(), "signed" => SignedTangents())

"""
    covers(f, x0, rule, channel) -> (; sum_loose, quad_loose, quad_objective)

Looseness of both merges under one `(rule, channel)` pair, and the quadrature objective
`sum(h_quad.^2)` that [`VectorNormSplit`](@ref) minimizes per node. The objective is the quantity
that rule is *trying* to reduce, so reporting it separates "chose the wrong member of the family"
from "the family itself is loose here".
"""
function covers(f, x0, rule, channel)
    H = ForwardDiff.hessian(f, x0)
    res = cover_hessian(WithDiagnostics(), f, x0; rule, channel)
    return (
        sum_loose = 1 / cover_ratio(H, res.h_sum),
        quad_loose = 1 / cover_ratio(H, res.h_quad),
        quad_objective = sum(abs2, res.h_quad),
    )
end

"""
    sampled_best(f, x0, channel; nw=9) -> objective

The smallest quadrature objective found by moving one input weight of `SensitivitySplit` at a time
over eight decades. A rule enters `balanced_split` only through the ratio of two scale references,
so varying the weights walks the family — but along one coordinate at a time this samples a thin
slice of it, and the result is an upper bound on the family's true optimum, not the optimum.

It is still the relevant check: the norm split's optimum is per node, and per-node optima across
different nodes need not be simultaneously achievable, so a cheap search finding anything better
would refute the claim that the rule is worth its cost.
"""
function sampled_best(f, x0, channel; nw = 9)
    n = length(x0)
    best = Inf
    for w in exp10.(range(-4, 4; length = nw))
        for a in 1:n
            weights = ones(n)
            weights[a] = w
            try
                best = min(best, covers(f, x0, SensitivitySplit(weights), channel).quad_objective)
            catch
                # A weight extreme enough to underflow a scale reference is not a family member.
            end
        end
    end
    return best
end

function rule_table(fixtures)
    @printf("%-22s %-9s", "case", "channel")
    for (label, _) in RULES
        @printf(" %10s %10s", "$label sum", "$label quad")
    end
    @printf(" %10s\n", "found/norm")
    for (name, f, x0) in fixtures
        for (clabel, channel) in CHANNELS
            @printf("%-22s %-9s", name, clabel)
            objectives = Float64[]
            for (_, rule) in RULES
                c = covers(f, x0, rule, channel)
                push!(objectives, c.quad_objective)
                @printf(" %10.4g %10.4g", c.sum_loose, c.quad_loose)
            end
            # Below 1 would mean the search beat the norm split's per-node optimum.
            @printf(" %10.3f\n", sampled_best(f, x0, channel) / objectives[end])
        end
    end
    return println()
end

"""
    singular_split(ratio; eps=1.0e-8)

`(x[1] - 1) * (x[2] - 1)` at `[1 + eps, 1 + ratio*eps]`. Both operands vanish as `eps` does, and
the cover of that product node is the one seam a value reference cannot resolve: the split it picks
depends on the *ratio* of two numbers that are both a hair from zero, so an arbitrarily small
change of direction swings it arbitrarily far, and at the singular point itself it falls back.

Sensitivity and norm references read derivatives, which are `1` on both operands throughout, so
they return the same split at every `ratio` and at the point itself.
"""
singular_split(ratio; eps = 1.0e-8) = (
    x -> (x[1] - 1) * (x[2] - 1),
    [1.0 + eps, 1.0 + ratio * eps],
)

function singularity_table(ratios)
    @printf("%12s", "ratio")
    for (label, _) in RULES
        @printf(" %21s", "$label h_quad")
    end
    println()
    for r in ratios
        f, x0 = singular_split(r)
        @printf("%12.3g", r)
        for (_, rule) in RULES
            h = cover_hessian(WithDiagnostics(), f, x0; rule).h_quad
            @printf(" %10.4g %10.4g", h[1], h[2])
        end
        println()
    end
    # The singular point itself, where no ratio exists at all.
    f = x -> (x[1] - 1) * (x[2] - 1)
    @printf("%12s", "exactly 0")
    for (_, rule) in RULES
        h = cover_hessian(WithDiagnostics(), f, [1.0, 1.0]; rule).h_quad
        @printf(" %10.4g %10.4g", h[1], h[2])
    end
    println()
    return println()
end

"""
    cutest_fixtures(; n=10)

Four problems from `OptimizationProblems.ADNLPProblems`, chosen because their splits are unstable
rather than because their covers are loose. `ADNLPModel` keeps the objective it was given, so the
generic `f` is taken straight off the model rather than transcribed.
"""
function cutest_fixtures(; n = 10)
    AP = OptimizationProblems.ADNLPProblems
    models = [
        ("hs255", AP.hs255()),
        ("beale", AP.beale()),
        ("brybnd", AP.brybnd(; n)),
        ("tointgss", AP.tointgss(; n)),
    ]
    return [(name, m.f, m.meta.x0) for (name, m) in models]
end

"""
    split_jitter(f, x0, rule; ndraws=24, releps=1.0e-12, seed=7) -> departure

Worst relative departure of `h_quad` from its unperturbed value, over `ndraws` perturbations of
relative size `releps` and over all coordinates.

A scale reference is a function of the evaluation point, so the split it picks is too, and nothing
requires that function to be continuous. This measures it: a rule whose reference varies smoothly
returns a departure of order `releps`, and one whose reference is degenerate at `x0` returns
whatever the discontinuity is worth. Validity is not at stake — every draw is a valid cover — only
reproducibility of the number a caller gets.
"""
function split_jitter(f, x0, rule; ndraws = 24, releps = 1.0e-12, seed = 7)
    rng = MersenneTwister(seed)
    base = cover_hessian(WithDiagnostics(), f, x0; rule).h_quad
    worst = 0.0
    for _ in 1:ndraws
        xp = x0 .* (1 .+ releps .* randn(rng, length(x0))) .+ releps .* randn(rng, length(x0))
        h = cover_hessian(WithDiagnostics(), f, xp; rule).h_quad
        for i in eachindex(base, h)
            base[i] == 0 && continue
            worst = max(worst, abs(h[i] - base[i]) / base[i])
        end
    end
    return worst
end

function stability_table(fixtures)
    @printf("%-22s", "case")
    for (label, _) in RULES
        @printf(" %12s", label)
    end
    println()
    for (name, f, x0) in fixtures
        @printf("%-22s", name)
        for (_, rule) in RULES
            @printf(" %12.3e", split_jitter(f, x0, rule))
        end
        println()
    end
    return println()
end

"Test problems drawn from the `DerivativeCoverProblems` families, at a common input count."
function family_fixtures(n; seed = 3)
    return map(families(; seed)) do (name, generate)
        p = generate(n)
        (name, p.f, p.x0)
    end
end

"`exp.(x) ./ sum(exp.(x))`, with the max detached: subtracting a constant leaves softmax unchanged."
function softmax(x::AbstractVector)
    e = exp.(x .- maximum(value, x))
    return e ./ sum(e)
end

"`(x .- mean(x)) ./ sqrt(var(x))` with the biased variance."
function layernorm(x::AbstractVector)
    n = length(x)
    d = x .- sum(x) / n
    return d ./ sqrt(sum(abs2, d) / n)
end

"Array-op heads, the recording style the other scripts measure."
function array_fixtures(rng, n)
    W1, W2 = randn(rng, 2n, n), randn(rng, n, 2n)
    A = let M = randn(rng, n, n)
        M + M'
    end
    x = 0.5 .* randn(rng, n)
    return [
        ("mlp 2-layer", z -> sum(abs2, W2 * tanh.(W1 * z)), x),
        ("softmax head", z -> sum(abs2, softmax(z)), x),
        ("layernorm head", z -> sum(abs2, tanh.(layernorm(z))), x),
        ("quadratic form", z -> dot(z, A * z) / 2, x),
    ]
end

function main()
    n = 12
    fixtures = vcat(family_fixtures(n), array_fixtures(MersenneTwister(20240607), n))
    cutest = cutest_fixtures()

    println("Looseness is 1/cover_ratio against ForwardDiff's Hessian: 1 is exact, larger is looser.")
    println("`found/norm` compares the smallest quadrature objective a one-coordinate weight sweep")
    println("finds against the norm split's; below 1 would mean the search beat its per-node optimum.")
    println()

    println("--- what a rule is worth, in each tangent channel ---")
    rule_table(fixtures)

    println("--- and on four problems from OptimizationProblems ---")
    rule_table(cutest)

    println("--- how far a split moves under a 1e-12 perturbation of the point ---")
    println("Order `releps` is a reference that varies smoothly; anything larger is a split that")
    println("does not survive a rounding difference in where the program was evaluated.")
    println()
    stability_table(cutest)

    println("--- the seam a value reference cannot resolve ---")
    println("Both operands of the product sit 1e-8 from zero; only their ratio varies down the")
    println("column, and only the value split answers differently at each one.")
    println()
    return singularity_table((100.0, 4.0, 1.0, 0.25, 0.01))
end

main()
