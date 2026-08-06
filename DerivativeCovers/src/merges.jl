# Grouped merges interpolate between sum and quadrature covers.
#
# Sum cover atoms within each of G groups, then combine the group totals in quadrature. One group
# gives the sum merge; one group per atom gives the quadrature merge. A vector-mode reverse sweep
# computes all group totals in O(G L).

"""
    cyclic_partition(t::Tape, ngroups::Integer) -> groups

Assign node `k` of `t` to group `1 + (k-1) % ngroups`, a node-granular partition for
[`cover_hybrid`](@ref).

Looseness comes from summing contributions whose covers overlap, and overlapping contributions are
typically *siblings* — the many product terms of a quadratic form, the fan-out of a reduction —
which a tape lists consecutively. Dealing them round-robin therefore spreads them across groups,
where quadrature combines them. Binning by depth instead (see [`level_partition`](@ref)) puts every
sibling in the *same* group, so on such programs it recovers little of the quadrature merge.

Siblings absorbed into a single array primitive are siblings this partition cannot separate, since
every atom of a node lands in the node's group. Dealing atoms instead — `cover_hybrid`'s integer
form, see [`group_seeds`](@ref) — spreads them whether the program packs them into one primitive
or spells them out.
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
    lmax = maximum(levels; init = 0)
    groups = ones(Int, nnodes(t))
    lmax == 0 && return groups
    for k in (t.ninputs + 1):nnodes(t)
        groups[k] = clamp(1 + ((levels[k] - 1) * ngroups) ÷ lmax, 1, ngroups)
    end
    return groups
end

"""
    group_seeds(t::Tape, rhos, groups::AbstractVector{<:Integer}) -> Theta
    group_seeds(t::Tape, rhos, ngroups::Integer) -> Theta

Seed weights for the vector-mode reverse sweep, `Theta[g, j] = sum over the contributions in group
g of the weight each places on node j`. Channels index the rows so that the weights of one
contribution are contiguous.

The two forms differ in what a group holds. `groups[k]` assigns tape node `k`, keeping all of its
atoms together, and `maximum(groups)` sets the channel count. An integer `ngroups` assigns *atoms*
(see [`atom_slots`](@ref)): they are numbered in tape order and dealt round-robin, so a node
exposing several atoms spreads them over several groups. The channel count is `ngroups`, capped at
[`natoms`](@ref)`(t)` since further channels would stay empty.

Since a node's atoms partition its slots, both forms deposit the same total weight and differ only
in how they distribute it over channels. Dealing atoms makes the distribution independent of how
many atoms one elemental carries, so an array primitive that absorbs several atoms into one node
still has them spread across channels. Which atoms share a channel remains a matter of the order
they appear in.

With `rhos` from [`local_covers`](@ref), column `j` collects exactly the seeds that
[`abs_reverse`](@ref) propagates into the group partial sums.
"""
function group_seeds(t::Tape{T}, rhos::AbstractVector, groups::AbstractVector{<:Integer}) where {T}
    N = nnodes(t)
    length(groups) == N ||
        throw(DimensionMismatch("groups has length $(length(groups)), expected $N (one per node)"))
    all(>=(1), groups) || throw(ArgumentError("group indices must be positive"))
    G = maximum(groups)
    Theta = zeros(T, G, N)
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

function group_seeds(t::Tape{T}, rhos::AbstractVector, ngroups::Integer) where {T}
    ngroups >= 1 || throw(ArgumentError("ngroups must be positive, got $ngroups"))
    N = nnodes(t)
    # Channels beyond the atom count would stay empty, so the sweep need not carry them.
    G = max(1, min(Int(ngroups), natoms(t)))
    Theta = zeros(T, G, N)
    a = 0
    for k in (t.ninputs + 1):N
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
    cover_hybrid(t::Tape, outidx::Integer, ngroups::Integer; check::Bool=true, scales=nothing, ratio=nothing) -> h
    cover_hybrid(t::Tape, outidx::Integer, groups::AbstractVector{<:Integer}; check::Bool=true, scales=nothing, ratio=nothing) -> h
    cover_hybrid(f, x0::AbstractVector, spec; check::Bool=true, scales=nothing, ratio=nothing) -> h

Compute a Hessian cover by grouping cover contributions: contributions are summed within each
group, then the group totals are combined in quadrature.

An integer `ngroups` groups the tape's cover atoms (see [`atom_slots`](@ref)), dealing them
round-robin. `ngroups=1` is the sum merge, and any `ngroups` at or above [`natoms`](@ref)`(t)`
gives every atom a group of its own, which is exactly the quadrature merge `h_quad` of diagnostic
[`cover_hessian`](@ref).
Because atoms rather than nodes are dealt, the whole range between the two merges stays reachable
however the program was written: an elemental that carries several atoms — an inner product, a
matrix-vector product — spreads them across groups like the scalar chain it stands for.

Alternatively, `groups[k]` assigns tape node `k` to a group, holding all of that node's atoms
together; [`cyclic_partition`](@ref) and [`level_partition`](@ref) build such partitions.

Every grouping produces a valid cover, and one group is always the sum merge. The computation
costs `O(G * L)` time and memory for `G` groups. `scales` and `ratio` govern the split, as in
[`cover_hessian`](@ref), and `check=true` verifies local covers.

# Examples

The two endpoints, on a tape carrying three atoms:

```jldoctest
julia> f(x) = exp(x[1]) + x[1] * x[2]^2;

julia> round.(cover_hybrid(f, [0.5, 2.0], 1); digits=4)   # one group: the sum merge
2-element Vector{Float64}:
 4.1125
 2.4142

julia> round.(cover_hybrid(f, [0.5, 2.0], 3); digits=4)   # a group per atom: the quadrature merge
2-element Vector{Float64}:
 3.1062
 1.7321
```
"""
function cover_hybrid(
        t::Tape, outidx::Integer, spec; check::Bool = true,
        scales = nothing, ratio = nothing
    )
    return _cover_hybrid(t, adjoints(t, outidx), spec; check, scales, ratio)
end

# Accept precomputed adjoints so callers such as cover_nls can reuse their gradient sweep.
function _cover_hybrid(
        t::Tape, vbar::AbstractVector, spec;
        check::Bool = true, scales = nothing, ratio = nothing
    )
    rhos, _ = local_covers(t, vbar; check, scales, ratio)
    Abar = abs_reverse(t, group_seeds(t, rhos, spec))
    return [sqrt(sum(abs2, view(Abar, :, a))) for a in 1:t.ninputs]
end

function cover_hybrid(f, x0::AbstractVector, spec; check::Bool = true, scales = nothing, ratio = nothing)
    t, outidx = tape_function(f, x0)
    return cover_hybrid(t, outidx, spec; check, scales, ratio)
end
