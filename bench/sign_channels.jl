# What each sign channel is worth to a Hessian cover.
#
# A cover's tightness turns on two channels. The tangents `∇v_j` carry an input's influence forward
# to each node; the adjoints `vbar_k` carry the output's sensitivity back. Either can be propagated
# with signs or in absolute-value arithmetic, and the absolute form is a bound wherever the exact
# one cancels. `cover_lattice` computes the cover with all four combinations:
#
#     quad    absolute tangents, signed adjoints    what `cover_hessian` computes
#     signed  signed tangents, signed adjoints      the tightest corner
#     fwd-G   absolute tangents, absolute adjoints  the loosest corner
#     fwd     signed tangents, absolute adjoints    what a tapeless forward pass reaches
#
# Looseness is `1/cover_ratio(H, h)` against the exact Hessian from ForwardDiff, the scale the
# other scripts use: 1 is exact, larger is looser. The two mixed corners are incomparable, so
# `fwd/quad` above 1 says the default cover is tighter on the worst entry; below 1 says the tapeless
# one does.
#
# A third loss survives all four corners: signs that cancel *between* elementals, which the
# triangle inequality over nodes discards before any channel is chosen. The cancellation family
# isolates it.

using DerivativeCovers, DerivativeCoverProblems, ForwardDiff, LinearAlgebra, Random, Printf,
    Statistics
using DerivativeCovers: cover_ratio, tape_function, nnodes, natoms, adjoints, abs_reverse,
    atom_slots, value, sensitivities, tangents, local_covers, merge_atoms, hessian_bound,
    magnitude

"""
    cover_lattice(t, outidx) -> (; B, B_signed, h_quad, h_quad_absadj, h_quad_signed, h_quad_forward)

The quadrature merge of `t`'s Hessian cover in all four combinations of a signed or absolute
intermediate-gradient channel with a signed or absolute adjoint channel. The absolute
intermediate gradients are `sensitivities(t)`, the signed ones the magnitudes of `tangents(t)`;
the signed adjoints are `abs.(adjoints(t, outidx))`, the absolute ones `acheck`, the unsigned sum
over paths from each node to the output. `B` and `B_signed` are the elementwise bounds formed
over the two intermediate-gradient channels with signed adjoints.
"""
function cover_lattice(t, outidx)
    vbar = adjoints(t, outidx)
    seed = zeros(eltype(t), nnodes(t))
    seed[outidx] = 1
    acheck = abs_reverse(t, seed)
    utilde = sensitivities(t)
    w = [magnitude.(g) for g in tangents(t)]
    rhos_signed, _ = local_covers(t, vbar)
    rhos_abs, _ = local_covers(t, acheck)
    return (;
        B = hessian_bound(t, vbar, utilde),
        B_signed = hessian_bound(t, vbar, w),
        h_quad = merge_atoms(t, rhos_signed, utilde)[2],
        h_quad_absadj = merge_atoms(t, rhos_abs, utilde)[2],
        h_quad_signed = merge_atoms(t, rhos_signed, w)[2],
        h_quad_forward = merge_atoms(t, rhos_abs, w)[2],
    )
end

"Geometric mean and maximum of `num ./ den` over the entries where `den` is positive."
function ratio_stats(num, den)
    rs = [num[i] / den[i] for i in eachindex(num, den) if den[i] > 0]
    isempty(rs) && return (geo = 1.0, max = 1.0)
    return (geo = exp(mean(log, rs)), max = maximum(rs))
end

"""
    adjoint_index(t, outidx) -> (; geo, max, ndead)

The unsigned adjoint `acheck[k]` divided by `abs(vbar[k])`, over the nodes that carry curvature.
This is the node-level quantity the adjoint channel discards, and it bounds on its own what a
tapeless construction gives up: `acheck` is the factor a forward propagation applies to each
node's atoms, `abs(vbar)` the factor a reverse sweep supplies.

`ndead` counts the curvature nodes the signed adjoint switches off entirely — `vbar[k] == 0` while
`acheck[k] > 0`, where the ratio is infinite — and those are excluded from `geo` and `max`.
"""
function adjoint_index(t, outidx)
    vbar = adjoints(t, outidx)
    seed = zeros(eltype(t), nnodes(t))
    seed[outidx] = 1
    acheck = abs_reverse(t, seed)
    rs, ndead = Float64[], 0
    for k in (t.ninputs + 1):nnodes(t)
        isempty(atom_slots(t.Cs[k], length(t.parents[k]))) && continue
        acheck[k] > 0 || continue
        vbar[k] == 0 ? (ndead += 1) : push!(rs, acheck[k] / abs(vbar[k]))
    end
    isempty(rs) && return (geo = 1.0, max = 1.0, ndead)
    return (geo = exp(mean(log, rs)), max = maximum(rs), ndead)
end

"""
    corners(f, x0) -> (; ...)

Looseness of the four lattice corners, the per-coordinate gain each sign channel delivers, the
`sqrt(B)`-scale split of the elementwise bound, and the tape's adjoint index. One tape, one
Hessian, so every number below describes the same program at the same point.
"""
function corners(f, x0)
    t, outidx = tape_function(f, x0)
    lat = cover_lattice(t, outidx)
    H = ForwardDiff.hessian(f, x0)
    loose(h) = 1 / cover_ratio(H, h)

    # Both channels are scored the same way: how much the cover shrinks when the signs are kept.
    tangent = ratio_stats(lat.h_quad, lat.h_quad_signed)
    adjoint = ratio_stats(lat.h_quad_absadj, lat.h_quad)
    # The looseness of the elementwise bound factors on the sqrt scale a rank-1 cover lives on:
    # path cancellation inside each tangent, then cancellation across elementals.
    within = ratio_stats(sqrt.(diag(lat.B)), sqrt.(diag(lat.B_signed)))
    cross = ratio_stats(sqrt.(diag(lat.B_signed)), sqrt.(abs.(diag(H))))
    return (;
        n = t.ninputs, nodes = nnodes(t), natoms = natoms(t),
        quad = loose(lat.h_quad), signed = loose(lat.h_quad_signed),
        fwdg = loose(lat.h_quad_absadj), fwd = loose(lat.h_quad_forward),
        tangent, adjoint, within, cross, index = adjoint_index(t, outidx),
    )
end

function corner_table(fixtures)
    @printf(
        "%-22s %5s %6s %6s %9s %9s %9s %9s %8s\n",
        "case", "n", "nodes", "atoms", "quad", "signed", "fwd-G", "fwd", "fwd/quad"
    )
    for (name, f, x0) in fixtures
        c = corners(f, x0)
        @printf(
            "%-22s %5d %6d %6d %9.4g %9.4g %9.4g %9.4g %8.3f\n",
            name, c.n, c.nodes, c.natoms, c.quad, c.signed, c.fwdg, c.fwd, c.fwd / c.quad
        )
    end
    return println()
end

function decomposition_table(fixtures)
    @printf(
        "%-22s %15s %15s %15s %15s %13s\n",
        "case", "tangent geo/max", "adjoint geo/max", "within-path", "cross-elem",
        "acheck/vbar"
    )
    anydead = false
    for (name, f, x0) in fixtures
        c = corners(f, x0)
        anydead |= c.index.ndead > 0
        dead = c.index.ndead == 0 ? "" : @sprintf(" +%d", c.index.ndead)
        @printf(
            "%-22s %6.3f /%7.3f %6.3f /%7.3f %6.3f /%7.3f %6.3f /%7.3f %6.2f%s\n",
            name, c.tangent.geo, c.tangent.max, c.adjoint.geo, c.adjoint.max,
            c.within.geo, c.within.max, c.cross.geo, c.cross.max, c.index.geo, dead
        )
    end
    anydead &&
        println("`+k`: k curvature nodes whose signed adjoint is zero and unsigned adjoint is not.")
    return println()
end

"""
    sharing_sweep(n, sharings; ndraws=12)

Reconvergence is what gives either channel anything to cancel. `random_dag`'s `sharing` sets how
often an operand is an existing node rather than a fresh one: at `sharing = 0` the program is a
tree, every input reaches every node along one path, and no sign anywhere can cancel.

Every sharing level replays the same `ndraws` seeds, so the draws are paired across the sweep and
the medians differ only in this parameter. A single program says little here — the looseness of one
random DAG is dominated by which elementals it happened to draw.
"""
function sharing_sweep(n, sharings; ndraws = 12)
    @printf(
        "%8s %6s %9s %9s %9s %9s %9s %9s\n",
        "sharing", "atoms", "quad", "signed", "fwd-G", "fwd", "tangent", "adjoint"
    )
    for s in sharings
        cs = map(1:ndraws) do j
            p = random_dag(n; sharing = s, nops = 8n, rng = MersenneTwister(1000 + j))
            corners(p.f, p.x0)
        end
        med(g) = median(map(g, cs))
        @printf(
            "%8.2f %6.0f %9.4g %9.4g %9.4g %9.4g %9.3f %9.3f\n",
            s, med(c -> c.natoms), med(c -> c.quad), med(c -> c.signed), med(c -> c.fwdg),
            med(c -> c.fwd), med(c -> c.tangent.geo), med(c -> c.adjoint.geo)
        )
    end
    return println()
end

"""
    adjoint_sweep(deltas; ns=(2, 8, 64))

What the *adjoint* channel is worth. [`sharing_sweep`](@ref) varies reconvergence, which feeds the
tangent channel; an adjoint cancels only where the output is nearly stationary in an interior node,
and no amount of reconvergence through `+` produces that.

`adjoint_cancellation`'s `delta` sets how nearly stationary, so `adjoint` should track `2/delta`
while `tangent` stays pinned at 1 — the family has one path from each input to the product node, so
it isolates the channel. Each row repeats at several `ns` because the mechanism belongs to the
node, not the program: the separation is the same at every size.
"""
function adjoint_sweep(deltas; ns = (2, 8, 64))
    @printf("%9s %5s %6s %9s %9s %9s %9s %9s %9s\n",
            "delta", "n", "atoms", "quad", "signed", "fwd-G", "fwd", "tangent", "adjoint")
    for delta in deltas, n in ns
        p = adjoint_cancellation(n; delta)
        c = corners(p.f, p.x0)
        @printf("%9.3g %5d %6.0f %9.4g %9.4g %9.4g %9.4g %9.3f %9.3f\n",
                delta, n, c.natoms, c.quad, c.signed, c.fwdg, c.fwd,
                c.tangent.geo, c.adjoint.geo)
    end
    return println()
end

"""
    inner_solve_sweep(steps)

The same condition reached by the envelope theorem rather than by construction. Each Newton step on
the inner problem makes the output more nearly stationary in the final iterate, so the adjoint gain
climbs with `steps` — and unlike [`adjoint_sweep`](@ref) the tangent channel is engaged too, which
is what differentiating through a converged solve actually looks like.
"""
function inner_solve_sweep(steps)
    @printf("%6s %6s %9s %9s %9s %9s %9s %9s\n",
            "steps", "atoms", "quad", "signed", "fwd-G", "fwd", "tangent", "adjoint")
    for k in steps
        p = converged_inner_solve(; steps = k)
        c = corners(p.f, p.x0)
        @printf("%6d %6.0f %9.4g %9.4g %9.4g %9.4g %9.3f %9.3f\n",
                k, c.natoms, c.quad, c.signed, c.fwdg, c.fwd, c.tangent.geo, c.adjoint.geo)
    end
    return println()
end

"""
    cross_elemental(n, ms)

What no corner of the lattice recovers. `cancellation`'s two variants differ only in the sign of
the constants multiplying two of each quartet's four squares: every node value, every first
partial, and every adjoint agree between them, and only the curvature they carry differs. So the
tangent and adjoint channels have identical work to do in both, while the Hessian of the cancel
variant collapses onto its rank-1 `delta` term. The gap that opens is cancellation between
elementals, which every corner discards.
"""
function cross_elemental(n, ms)
    @printf(
        "%6s %10s %9s %9s %9s %9s %9s\n",
        "m", "variant", "quad", "signed", "fwd-G", "fwd", "cross-elem"
    )
    for m in ms, aligned in (true, false)
        p = cancellation(n; m, aligned, rng = MersenneTwister(11))
        c = corners(p.f, p.x0)
        @printf(
            "%6d %10s %9.4g %9.4g %9.4g %9.4g %9.3f\n",
            m, aligned ? "aligned" : "cancel", c.quad, c.signed, c.fwdg, c.fwd, c.cross.geo
        )
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

# Use the same `softmax` and `layernorm` definitions as the tests.
include(joinpath(@__DIR__, "..", "DerivativeCovers", "test", "composites.jl"))

"Array-op heads, the recording style the other scripts measure."
function array_fixtures(rng, n)
    W1, W2 = randn(rng, 2n, n), randn(rng, n, 2n)
    A = let M = randn(rng, n, n)
        M + M'
    end
    x = 0.5 .* randn(rng, n)
    return [
        ("mlp 2-layer", z -> sum(abs2, W2 * tanh.(W1 * z)), x),
        (
            "logsumexp", z -> let m = maximum(DerivativeCovers.value, z)
                log(sum(exp, z .- m)) + m
            end, x,
        ),
        ("softmax head", z -> sum(abs2, softmax(z)), x),
        ("layernorm head", z -> sum(abs2, tanh.(layernorm(z))), x),
        ("quadratic form", z -> dot(z, A * z) / 2, x),
    ]
end

function main()
    n = 12
    fixtures = vcat(family_fixtures(n), array_fixtures(MersenneTwister(20240607), n))

    println("Looseness is 1/cover_ratio against ForwardDiff's Hessian: 1 is exact, larger is looser.")
    println("`tangent` and `adjoint` are the factor by which keeping that channel's signs shrinks")
    println("the cover, per coordinate; `within-path` and `cross-elem` split the elementwise bound.")
    println()

    println("--- looseness of the four corners ---")
    corner_table(fixtures)

    println("--- what each channel is worth, and what it discards ---")
    decomposition_table(fixtures)

    println("--- reconvergence is what the channels cancel over ---")
    sharing_sweep(n, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))

    println("--- a stationary output is what the adjoint channel cancels over ---")
    adjoint_sweep((1.0, 1.0e-1, 1.0e-2, 1.0e-3))

    println("--- the same condition, reached by a converged inner solve ---")
    inner_solve_sweep(0:4)

    println("--- the loss no corner recovers ---")
    return cross_elemental(8, (1, 4, 16))
end

main()
