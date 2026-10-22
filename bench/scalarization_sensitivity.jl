# How much of the quadrature merge survives array-level recording.
#
# `cover_hybrid` sums cover contributions within a group and combines the group totals in
# quadrature, so what it can recover depends on which contributions the partition is able to
# separate. It distributes cover atoms without regard to how many of them one elemental
# carries: on the programs below the two recordings give the same curve. A node-granular partition
# (`cyclic_partition`) depends on that number. It separates siblings on a scalar tape, which lists them
# consecutively, but an array primitive holds those same siblings inside one node, where no such
# partition reaches them.
#
# Both recordings are scored against the atomwise quadrature merge of
# `cover_hessian(WithDiagnostics())`, which neither granularity affects. Node, curvature-node, and
# atom counts are reported alongside, since those are exact.

using DerivativeCovers, ForwardDiff, LinearAlgebra, Random, Statistics, Printf
using MatrixCovers: max_coverage_ratio
using DerivativeCovers: value, local_covers, adjoints, nnodes, natoms,
    cyclic_partition

"`dot(z, A*z)/2`, whose taped-by-taped inner product records as one node carrying `n` atoms."
quadform_prim(A) = z -> dot(z, A * z) / 2

"Same function as [`quadform_prim`](@ref), with the inner product as an explicit multiply-add chain."
function quadform_scalar(A)
    return z -> begin
        n = length(z)
        w = [sum(A[i, j] * z[j] for j in 1:n) for i in 1:n]
        acc = z[1] * w[1]
        for i in 2:n
            acc = acc + z[i] * w[i]
        end
        return acc / 2
    end
end

# Use the same `softmax` and `layernorm` definitions as the tests.
include(joinpath(@__DIR__, "..", "DerivativeCovers", "test", "composites.jl"))

"Head `dot(c0, softmax(z))`."
softmax_prim(c0) = z -> dot(c0, softmax(z))

"Same function as [`softmax_prim`](@ref), with explicit chains for the sum and the head."
function softmax_scalar(c0)
    return z -> begin
        m = maximum(value.(z))
        n = length(z)
        es = [exp(z[i] - m) for i in 1:n]
        s = es[1]
        for i in 2:n
            s = s + es[i]
        end
        ps = [es[i] / s for i in 1:n]
        acc = c0[1] * ps[1]
        for i in 2:n
            acc = acc + c0[i] * ps[i]
        end
        return acc
    end
end

"Head `dot(c0, tanh.(layernorm(z)))`."
function layernorm_prim(c0)
    return z -> begin
        n = length(z)
        mu = sum(z) / n
        d = z .- mu
        s2 = sum(abs2, d) / n
        return dot(c0, tanh.(d ./ sqrt(s2)))
    end
end

"Same function as [`layernorm_prim`](@ref), with explicit chains for both sums and the head."
function layernorm_scalar(c0)
    return z -> begin
        n = length(z)
        s = z[1]
        for i in 2:n
            s = s + z[i]
        end
        mu = s / n
        d = [z[i] - mu for i in 1:n]
        sq = [abs2(d[i]) for i in 1:n]
        ssum = sq[1]
        for i in 2:n
            ssum = ssum + sq[i]
        end
        denom = sqrt(ssum / n)
        y = [tanh(d[i] / denom) for i in 1:n]
        acc = c0[1] * y[1]
        for i in 2:n
            acc = acc + c0[i] * y[i]
        end
        return acc
    end
end

"""
    structure(f, z) -> (; nodes, curvnodes, atoms)

Tape size, the number of nodes carrying a second-order contribution, and the number of cover atoms
those nodes expose. `atoms > curvnodes` exactly where a node carries more than one atom.
"""
function structure(f, z)
    t, outidx = tape_function(f, z)
    rhos, _ = local_covers(t, adjoints(t, outidx); check = false)
    return (nodes = nnodes(t), curvnodes = count(!isempty, rhos), atoms = natoms(t))
end

"""
    merge_curve(f, z, Gs) -> (; sum_loose, quad_loose, byatom, bynode, recovered, recovered_node)

Looseness `1/max_coverage_ratio` of the sum merge, the atomwise quadrature merge, and `cover_hybrid` at
each group count in `Gs`, under both granularities of partition. `recovered` is the fraction of the
sum-to-quadrature gap that the finest partition reaches, measured in log space, and is `NaN` when
there is no gap to recover.
"""
function merge_curve(f, z, Gs)
    H = ForwardDiff.hessian(f, z)
    t, outidx = tape_function(f, z)
    d = cover_hessian(WithDiagnostics(), t, outidx)
    sum_loose = 1 / max_coverage_ratio(d.h_sum, H)
    quad_loose = 1 / max_coverage_ratio(d.h_quad, H)
    byatom = [1 / max_coverage_ratio(cover_hybrid(t, outidx, G), H) for G in Gs]
    bynode = [1 / max_coverage_ratio(cover_hybrid(t, outidx, cyclic_partition(t, G)), H) for G in Gs]
    span = log(sum_loose) - log(quad_loose)
    frac = h -> span > 1.0e-12 ? (log(sum_loose) - log(minimum(h))) / span : NaN
    return (;
        sum_loose, quad_loose, byatom, bynode,
        recovered = frac(byatom), recovered_node = frac(bynode),
    )
end

function report(name, fprim, fscalar, z, Gs)
    println("--- $name (n = $(length(z))) ---")
    forms = ("array" => fprim, "scalar" => fscalar)
    ms = Dict(label => merge_curve(f, z, Gs) for (label, f) in forms)
    @printf(
        "%8s %7s %10s %7s %9s %9s %11s %11s\n",
        "form", "nodes", "curvnodes", "atoms", "sum", "quad", "rec/atom", "rec/node"
    )
    for (label, f) in forms
        s = structure(f, z)
        m = ms[label]
        @printf(
            "%8s %7d %10d %7d %9.4g %9.4g %11.3f %11.3f\n",
            label, s.nodes, s.curvnodes, s.atoms, m.sum_loose, m.quad_loose,
            m.recovered, m.recovered_node
        )
    end
    @printf("%18s", "G:")
    for G in Gs
        @printf(" %8d", G)
    end
    println()
    for (label, _) in forms, (kind, curve) in ("by atom" => ms[label].byatom, "by node" => ms[label].bynode)
        @printf("%8s %9s", label, kind)
        for h in curve
            @printf(" %8.4g", h)
        end
        println()
    end
    return println()
end

function main()
    n = 12
    rng = MersenneTwister(4)
    z = 0.6 .* randn(rng, n)
    c0 = randn(rng, n)
    A = randn(rng, n, n)
    A = (A + A') ./ 2
    Gs = (1, 2, 4, 8, 16, 32, 64, 128)

    println("Looseness is 1/max_coverage_ratio: 1 is exact, larger is looser.")
    println("`rec/atom` and `rec/node` are the fractions of the sum-to-quadrature gap the finest")
    println("partition of each granularity reaches.")
    println()
    report("dot(z, A*z)/2", quadform_prim(A), quadform_scalar(A), z, Gs)
    report("dot(c0, softmax(z))", softmax_prim(c0), softmax_scalar(c0), z, Gs)
    report("dot(c0, tanh(layernorm(z)))", layernorm_prim(c0), layernorm_scalar(c0), z, Gs)
    return nothing
end

main()
