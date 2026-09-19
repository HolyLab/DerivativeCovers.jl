# What the support components lose when liveness cannot read values.
#
# `support_components` calls a tape edge live where its first partial is nonzero, which is a
# property of the point, not of the program. A traced backend has no values at trace time, so the
# most it can offer is structural support: every recorded edge live. Structural liveness is a
# superset of value liveness, so its components are never finer — the question is how much coarser
# they are on real programs, and what that costs.
#
# The cost is paid through `gramcover`, which sums `a_k^2` over the outputs sharing a column's
# component. Merging components enlarges every one of those sums, so both factors of the bound
# `abs(G_jk) <= s_j*s_k` inflate. Looseness here is `1/cover_ratio(G, s)` against the exact Gram
# matrix `G = J'*J` from ForwardDiff, on the same scale as the Hessian scripts: 1 is exact.

using DerivativeCovers, ForwardDiff, LinearAlgebra, Random, Printf
using DerivativeCovers: tape_function, nnodes, cover_jacobian, cover_ratio, support_components
using MatrixCovers: SupportComponents, gramcover, ncomponents, colcomponent

"""
    structural_components(t::Tape, outidxs) -> sc::SupportComponents

Counterpart of [`support_components`](@ref) that takes every recorded edge to be live, whatever its
first partial. This is the finest support a pattern propagation can derive from the recorded graph
alone, and it bounds how far value liveness and a traced backend can disagree.
"""
function structural_components(t, outidxs::AbstractVector{<:Integer})
    N, n = nnodes(t), t.ninputs
    dep = falses(N)
    dep[1:n] .= true
    for k in (n + 1):N
        dep[k] = any(p -> dep[p], t.parents[k])
    end
    reach = falses(N)
    for k in outidxs
        reach[k] = true
    end
    for k in N:-1:(n + 1)
        reach[k] || continue
        for p in t.parents[k]
            reach[p] = true
        end
    end
    live = dep .& reach

    parent = collect(1:N)
    function find(p)
        while parent[p] != p
            parent[p] = parent[parent[p]]        # path halving
            p = parent[p]
        end
        return p
    end
    for k in (n + 1):N
        live[k] || continue
        for p in t.parents[k]
            live[p] || continue
            rk, rp = find(k), find(p)
            rk == rp || (parent[rp] = rk)
        end
    end

    ids = zeros(Int, N)
    seen = Dict{Int, Int}()
    ncomp = 0
    for p in 1:N
        live[p] || continue
        r = find(p)
        id = get(seen, r, 0)
        if id == 0
            ncomp += 1
            id = ncomp
            seen[r] = id
        end
        ids[p] = id
    end
    rowcomp = [ids[k] for k in outidxs]
    return SupportComponents(rowcomp, ids[1:n], ncomp, Base.OneTo(length(outidxs)), Base.OneTo(n))
end

"Fraction of recorded parent slots whose first partial is nonzero."
function live_fraction(t)
    total = sum(length, t.ds)
    return count(!iszero, Iterators.flatten(t.ds)) / total
end

"""
    granularity(f, x0) -> (; ...)

Component counts, dead-input counts, and Gram-cover looseness under both liveness rules, for the
same tape and the same asymmetric cover `(a, b_sum)`. Only the components differ between the two
columns, so the looseness ratio isolates their effect.
"""
function granularity(f, x0)
    t, outidxs = tape_function(f, x0)
    scv, scs = support_components(t, outidxs), structural_components(t, outidxs)
    # Uniform output weights: the price below is stated against them.
    r = cover_jacobian(t, outidxs; a = ones(length(outidxs)))
    G = let J = ForwardDiff.jacobian(f, x0)
        J' * J
    end
    loose(sc) = 1 / cover_ratio(G, gramcover(r.a, r.b_sum, sc))
    dead(sc) = count(j -> colcomponent(sc, j) == 0, 1:t.ninputs)
    lv, ls = loose(scv), loose(scs)
    return (;
        nodes = nnodes(t), live = live_fraction(t),
        ncompv = ncomponents(scv), ncomps = ncomponents(scs),
        deadv = dead(scv), deads = dead(scs),
        loosev = lv, looses = ls, ratio = ls / lv,
    )
end

function fixture_table(fixtures)
    @printf(
        "%-24s %6s %6s %11s %9s %11s %11s %7s\n",
        "case", "nodes", "live", "ncomp v/s", "dead v/s", "gram value", "gram struct", "ratio"
    )
    for (name, f, x0) in fixtures
        g = granularity(f, x0)
        @printf(
            "%-24s %6d %5.0f%% %5d /%5d %4d /%4d %11.4g %11.4g %7.3f\n",
            name, g.nodes, 100 * g.live, g.ncompv, g.ncomps, g.deadv, g.deads,
            g.loosev, g.looses, g.ratio
        )
    end
    return println()
end

"""
    block_sweep(rng, n, nblocks)

Looseness against the number of blocks in a two-layer map whose weights are block diagonal. Value
liveness resolves every block; structural liveness resolves none, so the Gram bound picks up the
`a`-mass of all `nblocks` blocks in each factor.
"""
function block_sweep(rng, n, nblocks)
    x0 = 0.5 .* randn(rng, n)
    @printf("%8s %11s %11s %11s %7s\n", "blocks", "ncomp v/s", "gram value", "gram struct", "ratio")
    for nb in nblocks
        W1, W2 = blockdiagonal(rng, 2n, n, nb), blockdiagonal(rng, n, 2n, nb)
        g = granularity(x -> W2 * tanh.(W1 * x), x0)
        @printf(
            "%8d %5d /%5d %11.4g %11.4g %7.3f\n",
            nb, g.ncompv, g.ncomps, g.loosev, g.looses, g.ratio
        )
    end
    return println()
end

"Block-diagonal `r x c` matrix with `nb` equal blocks of standard normal entries."
function blockdiagonal(rng, r, c, nb)
    W = zeros(r, c)
    rs, cs = div(r, nb), div(c, nb)
    for b in 1:nb
        W[((b - 1) * rs + 1):(b * rs), ((b - 1) * cs + 1):(b * cs)] = randn(rng, rs, cs)
    end
    return W
end

"""
    point_dependence()

Two halves of a program joined only by the product `c*s`, where `c` depends on the first half and
`s` on the second. At a point where `c` vanishes the product's edge to `s` carries a zero partial
and the halves separate; anywhere else they do not. The components are therefore a property of the
point, which no pattern representation can reproduce.
"""
function point_dependence()
    w = [1.0, -2.0, 0.5, 1.5]
    a0 = [1.0, 0.5, -2.0, 0.0]
    a0[4] = -dot(w[1:3], a0[1:3]) / w[4]        # dot(w, a0) == 0
    x0 = vcat(a0, [0.3, -0.7, 1.1, 0.2])
    half1(z) = tanh(z[1] * z[2]) + z[3] * z[4]
    half2(z) = tanh(z[3]) * z[1] + z[2] * z[4]
    joined(x) = [half1(x[1:4]), half2(x[5:8]), dot(w, x[1:4]) * sum(x[5:8])]

    println("--- liveness at a point versus liveness of the program ---")
    fixture_table(
        (
            ("joined, c = 0", joined, x0),
            ("joined, c != 0", joined, x0 .+ 0.01),
        )
    )
    return nothing
end

"Banded map: output `i` depends on inputs `i` and `i+1` only."
chain(x) = [tanh(x[i]) * x[i + 1] for i in 1:(length(x) - 1)]

"Three outputs over four inputs, recorded without any array primitive."
group(z) = [tanh(z[1] * z[2]), z[3] * z[4], sum(z)]

function main()
    rng = MersenneTwister(7)
    n = 8
    W1, W2 = randn(rng, 12, n), randn(rng, 6, 12)
    B1, B2 = blockdiagonal(rng, 12, n, 2), blockdiagonal(rng, 6, 12, 2)
    Wdead = copy(W1)
    Wdead[:, 3] .= 0
    xd = 0.5 .* randn(rng, n)
    xc = 0.7 .* randn(rng, n)

    println("Looseness is 1/cover_ratio of the Gram cover against J'*J: 1 is exact, larger is looser.")
    println("`live` is the fraction of recorded parent slots whose first partial is nonzero.")
    println()
    println("--- components under value and structural liveness ---")
    fixture_table(
        (
            ("dense MLP", x -> W2 * tanh.(W1 * x), xd),
            ("block-diagonal MLP", x -> B2 * tanh.(B1 * x), xd),
            ("banded chain", chain, xc),
            ("split in the program", x -> [group(x[1:4]); group(x[5:8])], xd),
            ("MLP, dead input", x -> W2 * tanh.(Wdead * x), xd),
        )
    )

    println("--- blocks merged against looseness paid ---")
    block_sweep(MersenneTwister(11), 16, (1, 2, 4, 8))

    return point_dependence()
end

main()
