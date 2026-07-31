# What a declared curvature costs, measured against the decomposed scalar tape.
#
# An op whose within-operand curvature is a dense block larger than 2x2 has no verified cover:
# `local_cover_slots` rejects it, leaving `DeclaredCurvature` (via `submap`) as the only way to
# record it as a single node. Recording it that way moves two quantities at once. The node's
# first-order partials become exact, removing the path cancellation an absolute-value sweep cannot
# see on the decomposed form; the node's curvature becomes one rank-1 bound per output, discarding
# the structure the decomposed form keeps. This script scores both forms against ForwardDiff's
# Hessian so the two channels can be read separately.
#
# Regimes are per-op because the ops degenerate differently: softmax is governed by how peaked `p`
# is, while layernorm is invariant under `z -> a*z + b`, so a spread axis carries no information
# there, and an outlier entry is the analogous parameter. The quadratic outer function is weighted because
# `sum(abs2, layernorm(z))` is identically `n`, whose zero Hessian no cover can be tight on.

using DerivativeCovers, ForwardDiff, LinearAlgebra, Random, Statistics, Printf
using DerivativeCovers: value, cover_ratio, jacobian_surrogate, submap

"""
    softmax(x)

`exp.(x) ./ sum(exp.(x))`, with the max detached from the tape. Subtracting a constant from every
entry leaves softmax unchanged, so the detachment is exact and keeps the recorded graph identical
to the unshifted form.
"""
function softmax(x::AbstractVector)
    m = maximum(value, x)
    e = exp.(x .- m)
    return e ./ sum(e)
end

"""
    layernorm(x)

`(x .- mean(x)) ./ sqrt(var(x))` with the biased variance. The mean and variance carry real
derivatives and are recorded, so unlike `softmax`'s max they contribute path cancellation that an
absolute-value sweep bounds term by term.
"""
function layernorm(x::AbstractVector)
    n = length(x)
    mu = sum(x) / n
    d = x .- mu
    s2 = sum(abs2, d) / n
    return d ./ sqrt(s2)
end

tanhlayernorm(x::AbstractVector) = tanh.(layernorm(x))

"`log(sum(exp, x))` as an explicit chain: a scalar-output op, used as a control."
function logsumexp(x)
    s = exp(x[1])
    for i in 2:length(x)
        s = s + exp(x[i])
    end
    return log(s)
end

spreadz(s) = (n, rng) -> s .* randn(rng, n)
outlierz(o) = (n, rng) -> (z = randn(rng, n); z[1] += o; z)

const SOFTMAX_REGIMES = ("flat" => spreadz(0.5), "mid" => spreadz(2.0), "peaked" => spreadz(5.0))
const NORM_REGIMES = ("even" => outlierz(0.0), "skewed" => outlierz(3.0), "spiked" => outlierz(10.0))

const OPS = (
    ("softmax", softmax, SOFTMAX_REGIMES),
    ("layernorm", layernorm, NORM_REGIMES),
    ("tanh(layernorm)", tanhlayernorm, NORM_REGIMES),
)

const HEADS = (
    "linear" => (c0, y) -> dot(c0, y),
    "quadratic" => (c0, y) -> sum(abs2, c0 .* y) / 2,
    "tanh" => (c0, y) -> dot(c0, tanh.(y)),
)

const SIZES = (8, 32)

geo(v) = isempty(v) ? NaN : exp(mean(log, v))

"""
    firstorder_gap(F, z) -> (; max, geomean)

Entrywise `Ũ/abs(J)` over the nonzeros of `J`: the path-cancellation loss the decomposed tape
suffers. A one-node form has `ds == J`, so its ratio is `1` everywhere.
"""
function firstorder_gap(F, z)
    t, outidxs = tape_function(F, z)
    U = jacobian_surrogate(t, outidxs)
    J = ForwardDiff.jacobian(F, z)
    rs = Float64[]
    for j in axes(J, 2), i in axes(J, 1)
        abs(J[i, j]) > 1.0e-12 && push!(rs, U[i, j] / abs(J[i, j]))
    end
    return (max = maximum(rs), geomean = geo(rs))
end

"""
    looseness(fdec, fprim, z) -> (; scale, sum_dec, sum_prim, quad_dec, quad_prim)

`1/cover_ratio` of both merges for two recordings of one function. `scale` is the largest entry of
the reference Hessian; a `scale` near zero means the ratios describe a vanishing matrix and carry
no information.
"""
function looseness(fdec, fprim, z)
    H = ForwardDiff.hessian(fdec, z)
    dd = cover_hessian(WithDiagnostics(), fdec, z)
    dp = cover_hessian(WithDiagnostics(), fprim, z)
    return (
        scale = maximum(abs, H),
        sum_dec = 1 / cover_ratio(H, dd.h_sum), sum_prim = 1 / cover_ratio(H, dp.h_sum),
        quad_dec = 1 / cover_ratio(H, dd.h_quad), quad_prim = 1 / cover_ratio(H, dp.h_quad),
    )
end

function firstorder_table()
    println("=== First-order loss of the decomposed tape: Utilde / abs(J) ===")
    @printf("%16s %5s %8s %12s %10s\n", "op", "n", "regime", "max", "geomean")
    for (name, F, regimes) in OPS, n in SIZES, (rlabel, gen) in regimes
        g = firstorder_gap(F, gen(n, MersenneTwister(11 + n)))
        @printf("%16s %5d %8s %12.4g %10.4g\n", name, n, rlabel, g.max, g.geomean)
    end
    return nothing
end

penalty_header(extra) = @printf(
    "%16s %10s %5s %8s %10s %10s %6s %10s %10s %6s\n",
    "op", "head", "n", extra, "sum_dec", "sum_prim", "pen", "quad_dec", "quad_prim", "pen"
)

function penalty_row(opname, hname, n, label, r)
    flag = r.scale < 1.0e-10 ? "  <- degenerate H" : ""
    return @printf(
        "%16s %10s %5d %8s %10.4g %10.4g %6.2f %10.4g %10.4g %6.2f%s\n",
        opname, hname, n, label, r.sum_dec, r.sum_prim, r.sum_prim / r.sum_dec,
        r.quad_dec, r.quad_prim, r.quad_prim / r.quad_dec, flag
    )
end

"`head(op(z))`: the op sees the inputs directly."
function penalty_table()
    println("=== Declaration penalty: looseness of head(op(z)), decomposed vs one submap node ===")
    penalty_header("regime")
    for (opname, F, regimes) in OPS, (hname, head) in HEADS, n in SIZES, (rlabel, gen) in regimes
        rng = MersenneTwister(11 + n)
        z = gen(n, rng)
        c0 = randn(rng, n)
        r = looseness(z -> head(c0, F(z)), z -> head(c0, submap(F, z)), z)
        penalty_row(opname, hname, n, rlabel, r)
    end
    return nothing
end

"`head(op(A*z))`: only the op is declared, with a recorded matvec upstream of it."
function embedded_table()
    println("=== Embedded downstream of a matvec: head(op(A*z)), only the op declared ===")
    penalty_header("")
    for (opname, F, _) in OPS, (hname, head) in HEADS, n in SIZES
        rng = MersenneTwister(11 + n)
        z = randn(rng, n)
        c0 = randn(rng, n)
        A = randn(rng, n, n) ./ sqrt(n)
        r = looseness(z -> head(c0, F(A * z)), z -> head(c0, submap(F, A * z)), z)
        penalty_row(opname, hname, n, "", r)
    end
    return nothing
end

"""
Two properties the sweep rests on.

A scalar-output op has no first-order channel to recover, and `declare_submap` builds its
declaration as the sum-merge cover of the same curvature the decomposed tape sweeps, so the two
forms agree exactly under the sum merge. They do not agree under quadrature: the declared node is
a single atom, so its quadrature merge collapses onto its sum merge, while the decomposed form
combines many atoms. That gap is the declaration penalty in its purest form — everything else the
sweep measures is this plus the first-order channel.

Softmax's off-diagonal entries reach an input by a single path, making the surrogate exact there;
the diagonal carries two opposing paths and loses `(1+p_i)/(1-p_i)`.
"""
function controls()
    println("=== Controls ===")
    for n in SIZES, spread in (0.5, 2.0, 5.0)
        z = spread .* randn(MersenneTwister(11 + n), n)
        r = looseness(logsumexp, z -> only(submap(logsumexp, z)), z)
        @printf(
            "logsumexp n=%2d spread=%.1f  sum forms agree: %-5s  quad: %.4g vs %.4g (pen %.2f)\n",
            n, spread, r.sum_dec ≈ r.sum_prim, r.quad_dec, r.quad_prim, r.quad_prim / r.quad_dec
        )
    end
    for n in SIZES, spread in (2.0, 5.0)
        z = spread .* randn(MersenneTwister(11 + n), n)
        t, outidxs = tape_function(softmax, z)
        U = jacobian_surrogate(t, outidxs)
        J = ForwardDiff.jacobian(softmax, z)
        p = softmax(z)
        diagr = [U[i, i] / abs(J[i, i]) for i in 1:n]
        offr = [U[i, j] / abs(J[i, j]) for i in 1:n, j in 1:n if i != j]
        @printf(
            "softmax   n=%2d spread=%.1f  diag == (1+p)/(1-p): %s  max off-diag ratio: %.12f\n",
            n, spread, diagr ≈ [(1 + p[i]) / (1 - p[i]) for i in 1:n], maximum(offr)
        )
    end
    return nothing
end

function main()
    firstorder_table()
    println()
    penalty_table()
    println()
    embedded_table()
    println()
    controls()
    return nothing
end

main()
