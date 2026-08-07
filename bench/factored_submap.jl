# What a sub-map costs when its Jacobian is factored rather than stored dense.
#
# `submapnode` carries an exact `q x p` Jacobian and a `q x p` curvature bound, built by one
# adjoint sweep per output. A traced backend has neither the storage nor the per-output sweeps: the
# representation intended for traced backends is a vjp closure plus a factored cover `(a, b, S)`, which the package
# already reaches through `compose_jacobian_cover` and `compose_hessian_cover` — no joint tape and
# no dense `J`. What that replacement costs is a tightness question, and this script measures it.
#
# Three routes carry the same composite. Decomposed records the sub-map's internals on the outer
# tape; declared records it as one `submap` node per output; factored covers the two factors
# separately and combines the covers. Only the last is available without values at every node, so
# the declared-to-factored ratio measures the loss from this change, and the decomposed column shows where
# both sit relative to keeping everything on one tape.

using DerivativeCovers, ForwardDiff, LinearAlgebra, Random, Printf
using DerivativeCovers: cover_ratio, jacobian_minimax, declare_submap, submap,
    compose_jacobian_cover, compose_hessian_cover, value

# Use the same `softmax` and `layernorm` definitions as the tests.
include(joinpath(@__DIR__, "..", "DerivativeCovers", "test", "composites.jl"))

tanhlayernorm(x::AbstractVector) = tanh.(layernorm(x))

const OPS = (("softmax", softmax), ("layernorm", layernorm), ("tanh(layernorm)", tanhlayernorm))
const SIZES = (8, 32)

const HEADS = (
    "linear" => (c0, y) -> dot(c0, y),
    "quadratic" => (c0, y) -> sum(abs2, c0 .* y) / 2,
    "tanh" => (c0, y) -> dot(c0, tanh.(y)),
)

"""
    excess(J, a, b) -> e

Worst-case looseness `max_kj (a_k*b_j)/abs(J_kj)` over the nonzero entries of the exact Jacobian:
the same scale `jacobian_minimax` minimizes, so its optimum is the floor for any rank-1 pair.
"""
excess(J, a, b) =
    maximum(a[k] * b[j] / abs(J[k, j]) for k in axes(J, 1), j in axes(J, 2) if !iszero(J[k, j]))

"""
    jacobian_routes(F, G, x0) -> (; decomposed, declared, factored, floor)

Excess of a rank-1 cover of the Jacobian of `G ∘ F` by each of the three routes, against the exact
optimum over all rank-1 pairs. The factored route never sees the composite: it covers each factor
at its own point and combines those covers.
"""
function jacobian_routes(F, G, x0)
    J = ForwardDiff.jacobian(x -> G(F(x)), x0)
    rdec = cover_jacobian(x -> G(F(x)), x0)
    rdcl = cover_jacobian(x -> G(submap(F, x)), x0)
    rF = cover_jacobian(F, x0)
    rG = cover_jacobian(G, F(x0))
    rfac = compose_jacobian_cover(rG.a, rG.b_sum, rF.a, rF.b_sum)
    return (;
        decomposed = excess(J, rdec.a, rdec.b_sum),
        declared = excess(J, rdcl.a, rdcl.b_sum),
        factored = excess(J, rfac.a, rfac.b),
        floor = jacobian_minimax(abs.(J)).excess,
    )
end

"""
    hessian_routes(F, g, x0) -> (; sum, quad)

Looseness `1/cover_ratio` of the Hessian of `g ∘ F` by each of the three routes, under both merges.
The factored route takes the head's own cover and gradient from a tape of `g` at `F(x0)`, the
Jacobian cover from a tape of `F` at `x0`, and the curvature bound from `declare_submap` — the same
`S` the declared route embeds, so the two routes differ only in how the `J'∇²g J` term is bounded.
"""
function hessian_routes(F, g, x0)
    H = ForwardDiff.hessian(x -> g(F(x)), x0)
    dec = cover_hessian(WithDiagnostics(), x -> g(F(x)), x0)
    dcl = cover_hessian(WithDiagnostics(), x -> g(submap(F, x)), x0)
    hd = cover_hessian(WithDiagnostics(), g, F(x0))
    rF = cover_jacobian(F, x0)
    S = declare_submap(F, x0).S
    fsum = compose_hessian_cover(hd.h_sum, abs.(hd.grad), rF.a, rF.b_sum, S)
    fquad = compose_hessian_cover(hd.h_quad, abs.(hd.grad), rF.a, rF.b_sum, S)
    loose(h) = 1 / cover_ratio(H, h)
    return (
        sum = (
            decomposed = loose(dec.h_sum), declared = loose(dcl.h_sum),
            factored = loose(fsum.h_sum),
        ),
        quad = (
            decomposed = loose(dec.h_quad), declared = loose(dcl.h_quad),
            factored = loose(fquad.h_quad),
        ),
    )
end

function jacobian_table()
    println("=== Jacobian of G(F(z)), G(y) = B*tanh.(y): excess against the rank-1 optimum ===")
    @printf(
        "%16s %5s %12s %12s %12s %9s %12s\n",
        "op", "n", "decomposed", "declared", "factored", "fact/decl", "rank-1 floor"
    )
    for (name, F) in OPS, n in SIZES
        rng = MersenneTwister(11 + n)
        z = randn(rng, n)
        B = randn(rng, n, n) ./ sqrt(n)
        r = jacobian_routes(F, y -> B * tanh.(y), z)
        @printf(
            "%16s %5d %12.4g %12.4g %12.4g %9.2f %12.4g\n",
            name, n, r.decomposed, r.declared, r.factored, r.factored / r.declared, r.floor
        )
    end
    return println()
end

function hessian_table()
    println("=== Hessian of head(F(z)): looseness 1/cover_ratio ===")
    @printf(
        "%16s %10s %5s %10s %10s %10s %7s %10s %10s %10s %7s\n",
        "op", "head", "n", "sum_dec", "sum_decl", "sum_fact", "pen",
        "quad_dec", "quad_decl", "quad_fact", "pen"
    )
    for (name, F) in OPS, (hname, head) in HEADS, n in SIZES
        rng = MersenneTwister(11 + n)
        z = randn(rng, n)
        c0 = randn(rng, n)
        r = hessian_routes(F, y -> head(c0, y), z)
        @printf(
            "%16s %10s %5d %10.4g %10.4g %10.4g %7.2f %10.4g %10.4g %10.4g %7.2f\n",
            name, hname, n,
            r.sum.decomposed, r.sum.declared, r.sum.factored, r.sum.factored / r.sum.declared,
            r.quad.decomposed, r.quad.declared, r.quad.factored, r.quad.factored / r.quad.declared
        )
    end
    return println()
end

"""
    control()

Where the factoring penalty lives. A head linear in the sub-map's outputs has `∇²g = 0`, which
deletes the `J'∇²g J` term from the chain rule and leaves only the declared curvature `S` — the
same bound both routes carry. The two covers then agree to the last digit, so the entire penalty
in the table above comes from bounding `J'∇²g J` with `(a⋅hg)²*b*b'` instead of the exact `J`.
"""
function control()
    println("=== Control: a linear head leaves nothing to factor ===")
    for (name, F) in OPS, n in SIZES
        rng = MersenneTwister(11 + n)
        z = randn(rng, n)
        c0 = randn(rng, n)
        r = hessian_routes(F, y -> dot(c0, y), z)
        @printf(
            "%16s n=%2d  sum %s  quad %s\n", name, n,
            r.sum.declared == r.sum.factored ? "identical" : "DIFFER",
            r.quad.declared == r.quad.factored ? "identical" : "DIFFER"
        )
    end
    return println()
end

function main()
    println("Both scales report looseness: 1 is exact, larger is looser.")
    println()
    jacobian_table()
    hessian_table()
    return control()
end

main()
