# Merges of the per-node covers that interpolate between the two extremes of `cover_hessian`.
#
# Writing s_k for node k's cover vector, the sum merge takes sum_k s_k and the quadrature merge
# takes sqrt(sum_k s_k.^2). Partition the nodes into G groups, sum within a group and combine
# groups in quadrature, and the two extremes become the endpoints G = 1 and one-node-per-group.
# Every group's partial sum comes out of one channel of a vector-mode absolute-value reverse
# sweep, so the cost is O(G*L) rather than the O(n*L) of the exact quadrature merge.

"""
    cyclic_partition(t::Tape, ngroups::Integer) -> groups

Assign node `k` of `t` to group `1 + (k-1) % ngroups`, the default partition of
[`cover_hybrid`](@ref).

Looseness comes from summing nodes whose covers overlap, and overlapping nodes are typically
*siblings* — the many product terms of a quadratic form, the fan-out of a reduction — which a
tape lists consecutively. Dealing them round-robin therefore spreads them across groups, where
quadrature combines them. Binning by depth instead (see [`level_partition`](@ref)) puts every
sibling in the *same* group, so on such programs it recovers little of the quadrature merge.
"""
function cyclic_partition(t::Tape, ngroups::Integer)
    ngroups >= 1 || throw(ArgumentError("ngroups must be positive, got $ngroups"))
    return [1 + (k - 1) % ngroups for k in 1:nnodes(t)]
end

"""
    node_levels(t::Tape) -> levels

Depth of each node: `0` for the inputs, and `1 + maximum(levels[parents])` otherwise. Parents
always precede their children on the tape, so one forward pass suffices.
"""
function node_levels(t::Tape)
    N = nnodes(t)
    levels = zeros(Int, N)
    for k in (t.ninputs + 1):N
        lk = 0
        for p in t.parents[k]
            lk = max(lk, levels[p] + 1)
        end
        levels[k] = lk
    end
    return levels
end

"""
    level_partition(t::Tape, ngroups::Integer) -> groups

Assign each node of `t` to one of `ngroups` groups by binning [`node_levels`](@ref) into
contiguous depth bands. Nodes at the same depth always land in the same group, and when the tape
is shallower than `ngroups` each depth gets a group of its own (leaving the surplus groups
empty). Inputs carry no second-order contribution, so their group is irrelevant; they are
assigned group 1.

This separates nodes along a chain but never separates siblings, which limits how much of the
quadrature merge it can recover; [`cyclic_partition`](@ref) is the default for that reason.
"""
function level_partition(t::Tape, ngroups::Integer)
    ngroups >= 1 || throw(ArgumentError("ngroups must be positive, got $ngroups"))
    levels = node_levels(t)
    lmax = maximum(levels; init=0)
    groups = ones(Int, nnodes(t))
    lmax == 0 && return groups
    for k in (t.ninputs + 1):nnodes(t)
        groups[k] = clamp(1 + ((levels[k] - 1) * ngroups) ÷ lmax, 1, ngroups)
    end
    return groups
end

"""
    group_seeds(t::Tape, rhos, groups) -> Theta

Seed weights for the vector-mode reverse sweep, `Theta[g, j] = sum over nodes k in group g of the
weight node k places on its parent j`. Channels index the rows so that each node's weights are
contiguous. `groups[k]` is the group of node k; `maximum(groups)` sets the channel count.

With `rhos` from [`local_covers`](@ref), column `j` collects exactly the seeds that
[`abs_reverse`](@ref) propagates into the group partial sums.
"""
function group_seeds(t::Tape, rhos::AbstractVector, groups::AbstractVector{<:Integer})
    N = nnodes(t)
    length(groups) == N ||
        throw(DimensionMismatch("groups has length $(length(groups)), expected $N (one per node)"))
    all(>=(1), groups) || throw(ArgumentError("group indices must be positive"))
    G = maximum(groups)
    Theta = zeros(G, N)
    for k in (t.ninputs + 1):N
        rho = rhos[k]
        isempty(rho) && continue
        g = groups[k]
        for (i, p) in enumerate(t.parents[k])
            Theta[g, p] += rho[i]
        end
    end
    return Theta
end

"""
    abs_reverse(t::Tape, Theta::AbstractMatrix) -> Abar

Vector-mode form of [`abs_reverse`](@ref): each row of `Theta` is an independent seed vector over
the nodes (columns), and all rows are propagated in one traversal. `Abar[:, a]` for an input `a`
holds that input's value for every channel.
"""
function abs_reverse(t::Tape, Theta::AbstractMatrix)
    size(Theta, 2) == nnodes(t) ||
        throw(DimensionMismatch("Theta has $(size(Theta, 2)) columns, expected $(nnodes(t)) (one per node)"))
    Abar = collect(float.(Theta))
    G = size(Abar, 1)
    for k in nnodes(t):-1:1
        for (i, p) in enumerate(t.parents[k])
            a = abs(t.ds[k][i])
            a == 0 && continue
            for g in 1:G
                Abar[g, p] += a * Abar[g, k]
            end
        end
    end
    return Abar
end

"""
    cover_hybrid(t::Tape, outidx::Int, groups::AbstractVector{<:Integer}; check::Bool=true, rule=ValueSplit()) -> h
    cover_hybrid(t::Tape, outidx::Int, ngroups::Integer; check::Bool=true, rule=ValueSplit()) -> h
    cover_hybrid(f, x0::AbstractVector, spec; check::Bool=true, rule=ValueSplit()) -> h

Cover of the Hessian obtained by summing the per-node covers within each group of `groups` and
combining the groups in quadrature. An integer `spec` selects a [`cyclic_partition`](@ref) with
that many groups. `rule` selects the [`SplitRule`](@ref) behind the per-node covers.

The result is a valid cover for *any* partition: the group partial sums `S_g = sum_{k in g} s_k`
are nonnegative, so `sum_g S_g S_g' >= sum_k s_k s_k'` entrywise, and Cauchy–Schwarz makes
`h[i] = sqrt(sum_g S_g[i]^2)` a cover of the left-hand side. Refining a partition can only
tighten `h`, since `sqrt((a+b)^2) >= sqrt(a^2 + b^2)` for nonnegative `a`, `b`; one group
reproduces the sum merge and one node per group reproduces the quadrature merge.

Costs one adjoint sweep and one `maximum(groups)`-channel absolute-value reverse sweep, i.e.
`O(G*L)` time and memory — independent of the input dimension, unlike the exact quadrature merge.

A partition assigns whole *nodes* to groups, so an inner-product node's atoms (see
[`atom_slots`](@ref)) always travel together within one channel. The finest-partition endpoint
on a tape with inner-product nodes is therefore the per-node quadrature merge, bounded below by
`h_quad` from [`cover_hessian`](@ref) rather than equal to it; validity for every partition, and
the interpolation between the sum and quadrature extremes, are unaffected.
"""
function cover_hybrid(t::Tape, outidx::Int, groups::AbstractVector{<:Integer}; check::Bool=true,
                      rule::SplitRule=ValueSplit())
    vbar = adjoints(t, outidx)
    rhos, _ = local_covers(t, vbar; check, rule)
    Abar = abs_reverse(t, group_seeds(t, rhos, groups))
    return [sqrt(sum(abs2, view(Abar, :, a))) for a in 1:t.ninputs]
end

function cover_hybrid(t::Tape, outidx::Int, ngroups::Integer; check::Bool=true,
                      rule::SplitRule=ValueSplit())
    return cover_hybrid(t, outidx, cyclic_partition(t, ngroups); check, rule)
end

function cover_hybrid(f, x0::AbstractVector, spec; check::Bool=true, rule::SplitRule=ValueSplit())
    t, outidx = tape_function(f, x0)
    return cover_hybrid(t, outidx, spec; check, rule)
end
