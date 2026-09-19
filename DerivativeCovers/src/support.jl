# Connected components of the tape's output-input support graph.
#
# Outputs and inputs are vertices; a nonzero surrogate entry is an edge. The cover gauge acts
# independently on each component, so scaling conventions and Gram-cover accumulation must also
# be componentwise.
#
# This O(L) tape traversal avoids materializing Ũ. MatrixCovers handles the resulting components.

struct UnionFind
    parent::Vector{Int}
end

UnionFind(n::Integer) = UnionFind(collect(1:n))

function _find(uf::UnionFind, p::Int)
    parent = uf.parent
    while parent[p] != p
        parent[p] = parent[parent[p]]        # path halving
        p = parent[p]
    end
    return p
end

function _union!(uf::UnionFind, p::Int, q::Int)
    rp, rq = _find(uf, p), _find(uf, q)
    rp == rq && return rp
    uf.parent[rq] = rp
    return rp
end

# Number nonempty components by first encounter.
function _label_components(uf::UnionFind, touched::AbstractVector{Bool})
    ids = zeros(Int, length(touched))
    seen = Dict{Int, Int}()
    ncomp = 0
    for p in eachindex(touched)
        touched[p] || continue
        r = _find(uf, p)
        id = get(seen, r, 0)
        if id == 0
            ncomp += 1
            id = ncomp
            seen[r] = id
        end
        ids[p] = id
    end
    return ids, ncomp
end

"""
    support_components(t::Tape, outidxs) -> sc::SupportComponents

Connected components of the bipartite support graph of the Jacobian of the outputs
`outidxs` with respect to the tape's inputs: outputs and inputs are the vertices,
and each nonzero `Ũ_kj` is an edge joining output `k` to input `j`. The result is
a `MatrixCovers.SupportComponents` whose row axis runs over the outputs in the
order of `outidxs` (`1:length(outidxs)`) and whose column axis runs over the
inputs (`1:t.ninputs`); query it with `rowcomponent`, `colcomponent`, and
`ncomponents`.

This is the tape counterpart of `MatrixCovers.support_components(Ũ)`: it costs
`O(L)` in the tape length and never forms `Ũ`. A tape edge transmits sensitivity
exactly where its first partial is nonzero, so the surrogate's support is carried
by those edges alone, and a node is *live* when it both depends on an input and
reaches one of `outidxs` along them. Any undirected path between live tape nodes
splits into ascending and descending runs whose nodes each reach some output,
which forces the corresponding bipartite vertices into one component; conversely a
support edge is a directed path of live nodes. The undirected components of the
live tape graph therefore restrict to the bipartite components exactly.

Passing the result to [`gramcover`](@ref) covers a Gram matrix of the Jacobian
without ever forming `Ũ`.
"""
function support_components(t::Tape, outidxs::AbstractVector{<:Integer})
    N, n = nnodes(t), t.ninputs
    live = _live_nodes(t, outidxs)

    uf = UnionFind(N)
    for k in (n + 1):N
        live[k] || continue
        ps, ds = t.parents[k], t.ds[k]
        for i in eachindex(ps)
            (live[ps[i]] && !iszero(ds[i])) || continue
            _union!(uf, k, ps[i])
        end
    end
    ids, ncomp = _label_components(uf, live)
    m = length(outidxs)
    # Component rows follow output position, not the axes of outidxs.
    rowcomp = Vector{Int}(undef, m)
    for (p, k) in enumerate(outidxs)
        rowcomp[p] = ids[k]
    end
    colcomp = ids[1:n]
    return SupportComponents(rowcomp, colcomp, ncomp, Base.OneTo(m), Base.OneTo(n))
end

# Nodes lying on some input-to-output path whose local derivatives are all nonzero: one forward
# reachability pass from the inputs and one reverse pass from the output nodes, O(L) together.
# Only the slots between two live nodes carry sensitivity, so these are the nodes the surrogate's
# support is built from.
function _live_nodes(t::Tape, outidxs::AbstractVector{<:Integer})
    N, n = nnodes(t), t.ninputs
    for k in outidxs
        1 <= k <= N || throw(BoundsError(1:N, k))
    end

    # Forward reachability from inputs.
    dep = falses(N)
    dep[1:n] .= true
    for k in (n + 1):N
        ps, ds = t.parents[k], t.ds[k]
        for i in eachindex(ps)
            if dep[ps[i]] && !iszero(ds[i])
                dep[k] = true
                break
            end
        end
    end
    # Reverse reachability from outputs.
    reach = falses(N)
    for k in outidxs
        reach[k] = true
    end
    for k in N:-1:(n + 1)
        reach[k] || continue
        ps, ds = t.parents[k], t.ds[k]
        for i in eachindex(ps)
            iszero(ds[i]) || (reach[ps[i]] = true)
        end
    end
    return dep .& reach
end
