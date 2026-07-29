# Connected components of a program's bipartite output/input support graph, read
# straight off the tape.
#
# Outputs and inputs are the vertices, a nonzero surrogate entry Ũ_kj an edge. No
# product a_k*b_j of an asymmetric cover spans two components, so the gauge
# (a, b) -> (γa, γ⁻¹b) acts independently on each: a convention that pins the
# split is well posed only when imposed componentwise, and a single global
# constraint leaves ncomp-1 directions to the alternation's internals. This is why
# [`gramcover`](@ref) accumulates per component rather than globally.
#
# MatrixCovers computes the same components from a materialized Ũ in O(mn); the
# tape route below never forms Ũ and costs O(L), and hands its result to
# MatrixCovers as a `SupportComponents` so the two share one downstream path.

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

# Number the touched vertices 1:ncomp by first-encounter order of their roots.
function _label_components(uf::UnionFind, touched::AbstractVector{Bool})
    ids = zeros(Int, length(touched))
    seen = Dict{Int,Int}()
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
    for k in outidxs
        1 <= k <= N || throw(BoundsError(1:N, k))
    end

    # Depends on an input: forward sweep, parents precede children in tape order.
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
    # Reaches an output: reverse sweep, every child of k precedes k in this order.
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
    live = dep .& reach

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
    # Rows are numbered by output *position*, so the labels are collected 1:m however `outidxs`
    # itself is indexed.
    rowcomp = Vector{Int}(undef, m)
    for (p, k) in enumerate(outidxs)
        rowcomp[p] = ids[k]
    end
    colcomp = ids[1:n]
    return SupportComponents(rowcomp, colcomp, ncomp, Base.OneTo(m), Base.OneTo(n))
end
