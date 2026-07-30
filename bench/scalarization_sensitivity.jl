# How much of the quadrature merge survives array-level recording.
#
# `cover_hybrid` sums cover contributions within a group and combines the group totals in
# quadrature, so what it can recover depends on which contributions the partition is able to
# separate. `cyclic_partition` deals nodes round-robin, which separates siblings because a scalar
# tape lists them consecutively. An array primitive puts those same siblings inside one node, and
# groups contain whole nodes, so no partition separates them.
#
# The atomwise quadrature merge of `cover_hessian(WithDiagnostics())` is unaffected: `atom_slots`
# lets one node expose several atoms, and `dotnode` does. The question this script answers is how
# much of that atomwise limit the production merge can still reach on each recording, and it
# reports node and atom counts alongside, since those are exact.

using DerivativeCovers, ForwardDiff, LinearAlgebra, Random, Statistics, Printf
using DerivativeCovers: value, cover_ratio, atom_slots, local_covers, adjoints, nnodes,
    abs_reverse

"""
    hybrid_atomwise(t, outidx, G; rule=ValueSplit()) -> h

`cover_hybrid` with the partition applied to atoms rather than to whole nodes: atoms are numbered
in tape order and dealt round-robin over `G` groups, so a node exposing several atoms can spread
them. Every slot of a node belongs to exactly one of its atoms, so the seeds a node contributes
are the same ones `group_seeds` contributes, only distributed differently.
"""
function hybrid_atomwise(t, outidx, G; rule = ValueSplit())
    T = eltype(t)
    rhos, _ = local_covers(t, adjoints(t, outidx); check = false, rule)
    Theta = zeros(T, G, nnodes(t))
    a = 0
    for k in 1:nnodes(t)
        rho = rhos[k]
        isempty(rho) && continue
        ps = t.parents[k]
        for atom in atom_slots(t.Cs[k], length(ps))
            g = 1 + a % G
            a += 1
            for i in atom
                Theta[g, ps[i]] += rho[i]
            end
        end
    end
    Abar = abs_reverse(t, Theta)
    return [sqrt(sum(abs2, view(Abar, :, j))) for j in 1:t.ninputs]
end

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

function softmax(x::AbstractVector)
    m = maximum(value, x)
    e = exp.(x .- m)
    return e ./ sum(e)
end

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

Tape size, the number of nodes carrying a second-order contribution, and the number of quadrature
atoms those nodes expose. `atoms > curvnodes` exactly where a node carries more than one atom.
"""
function structure(f, z)
    t, outidx = tape_function(f, z)
    rhos, _ = local_covers(t, adjoints(t, outidx); check = false)
    curvnodes = count(!isempty, rhos)
    atoms = 0
    for k in 1:nnodes(t)
        isempty(rhos[k]) && continue
        atoms += length(collect(atom_slots(t.Cs[k], length(t.parents[k]))))
    end
    return (nodes = nnodes(t), curvnodes, atoms)
end

"""
    merge_curve(f, z, Gs) -> (; sum_loose, quad_loose, hybrid, recovered)

Looseness `1/cover_ratio` of the sum merge, the atomwise quadrature merge, and `cover_hybrid` at
each group count in `Gs`. `recovered` is the fraction of the sum-to-quadrature gap that the finest
partition reaches, measured in log space, and is `NaN` when there is no gap to recover.
"""
function merge_curve(f, z, Gs)
    H = ForwardDiff.hessian(f, z)
    t, outidx = tape_function(f, z)
    d = cover_hessian(WithDiagnostics(), t, outidx)
    sum_loose = 1 / cover_ratio(H, d.h_sum)
    quad_loose = 1 / cover_ratio(H, d.h_quad)
    hybrid = [1 / cover_ratio(H, cover_hybrid(t, outidx, G)) for G in Gs]
    atomwise = [1 / cover_ratio(H, hybrid_atomwise(t, outidx, G)) for G in Gs]
    span = log(sum_loose) - log(quad_loose)
    frac = h -> span > 1.0e-12 ? (log(sum_loose) - log(minimum(h))) / span : NaN
    return (; sum_loose, quad_loose, hybrid, atomwise, recovered = frac(hybrid), recovered_atom = frac(atomwise))
end

function report(name, fprim, fscalar, z, Gs)
    println("--- $name (n = $(length(z))) ---")
    forms = ("array" => fprim, "scalar" => fscalar)
    ms = Dict(label => merge_curve(f, z, Gs) for (label, f) in forms)
    @printf(
        "%8s %7s %10s %7s %9s %9s %11s %11s\n",
        "form", "nodes", "curvnodes", "atoms", "sum", "quad", "recovered", "rec/atomwise"
    )
    for (label, f) in forms
        s = structure(f, z)
        m = ms[label]
        @printf(
            "%8s %7d %10d %7d %9.4g %9.4g %11.3f %11.3f\n",
            label, s.nodes, s.curvnodes, s.atoms, m.sum_loose, m.quad_loose,
            m.recovered, m.recovered_atom
        )
    end
    @printf("%18s", "G:")
    for G in Gs
        @printf(" %8d", G)
    end
    println()
    for (label, _) in forms, (kind, curve) in ("by node" => ms[label].hybrid, "by atom" => ms[label].atomwise)
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

    println("Looseness is 1/cover_ratio: 1 is exact, larger is looser.")
    println("`recovered` is the fraction of the sum-to-quadrature gap the finest partition reaches.")
    println()
    report("dot(z, A*z)/2", quadform_prim(A), quadform_scalar(A), z, Gs)
    report("dot(c0, softmax(z))", softmax_prim(c0), softmax_scalar(c0), z, Gs)
    report("dot(c0, tanh(layernorm(z)))", layernorm_prim(c0), layernorm_scalar(c0), z, Gs)
    return nothing
end

main()
