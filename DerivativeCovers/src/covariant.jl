# Covariant O(L) output weighting for Jacobian covers: row and column targets read off the live
# slot graph, which are realizable on any sparsity pattern, and log-scale potentials on the tape
# graph, which give `balance_cover` a start that is covariant at every finite round count rather
# than only in the limit.

# Structural liveness, computed once per public call and threaded through everything that needs
# it. A node is live when it lies on some input-to-output path whose local derivatives are all
# nonzero, and a slot is live when it joins two live nodes with a nonzero derivative. `indeg`
# counts a live node's live parent slots, plus one virtual source slot at each live input; `outdeg`
# counts its live child slots, plus one virtual sink slot for every output position it fills.
# Those virtual slots are what normalize the two flow weightings: each carries the mass that enters
# at an input or leaves at an output.
#
# Underflow is not liveness: a product that falls below the smallest representable number leaves a
# structurally live entry numerically zero, and every caller here treats that as an error rather
# than as a dead row.
struct LiveInfo
    live::BitVector
    liverow::BitVector      # by output position
    livecol::BitVector      # by input
    indeg::Vector{Int}
    outdeg::Vector{Int}
    nrow::Int               # live output positions
    ncol::Int               # live inputs
end

function _live_info(t::Tape, outidxs)
    N, nin = nnodes(t), t.ninputs
    m = length(outidxs)
    live = _live_nodes(t, outidxs)
    indeg, outdeg = zeros(Int, N), zeros(Int, N)
    for k in (nin + 1):N
        live[k] || continue
        ps, ds = t.parents[k], t.ds[k]
        for i in eachindex(ps)
            (live[ps[i]] && !iszero(magnitude(ds[i]))) || continue
            indeg[k] += 1
            outdeg[ps[i]] += 1
        end
    end
    livecol = falses(nin)
    for j in 1:nin
        live[j] || continue
        livecol[j] = true
        indeg[j] += 1
    end
    # Liveness follows output position, not the axes of outidxs.
    liverow = falses(m)
    for (k, o) in enumerate(outidxs)
        live[o] || continue
        liverow[k] = true
        outdeg[o] += 1
    end
    return LiveInfo(live, liverow, livecol, indeg, outdeg, count(liverow), count(livecol))
end

# Slot weight `1/indeg(child)`: a node divides its incoming mass equally over its live parent
# slots, so every live node carries mass 1.
struct ForwardFlow
    li::LiveInfo
end
(f::ForwardFlow)(d, k, i, p) =
    (f.li.live[k] && f.li.live[p] && !iszero(magnitude(d))) ? 1 / f.li.indeg[k] : 0.0

# Slot weight `1/outdeg(parent)`: a node divides its mass equally over its live child slots and its
# virtual sink slots, so every live input emits exactly 1.
struct ReverseFlow
    li::LiveInfo
end
(f::ReverseFlow)(d, k, i, p) =
    (f.li.live[k] && f.li.live[p] && !iszero(magnitude(d))) ? 1 / f.li.outdeg[p] : 0.0

"""
    _flow_targets(t::Tape, outidxs, li::LiveInfo) -> (r, c)

Row and column targets from the two flow weightings of the live slot graph, mixed so that neither
side dominates: `r = ρ_fwd/nrow + ρ_rev/ncol` and `c = γ_fwd/nrow + γ_rev/ncol`, with `nrow` the
live output positions and `ncol` the live inputs. Marginals are linear in the weight matrix, so the
mixture is itself the marginal pair of a nonnegative matrix with `Ũ`'s pattern, which is what makes
the targets realizable.

Two of the four marginals are constant by construction — forward flow reaches every live node with
mass 1, and reverse flow leaves every live input with mass 1 — so only two sweeps run. `O(L)`,
forming no matrix; every weight is at most 1, so nothing can overflow. The two totals are `2`
each, to roundoff.
"""
function _flow_targets(t::Tape, outidxs, li::LiveInfo)
    N, nin, m = nnodes(t), t.ninputs, length(outidxs)
    (li.nrow > 0 && li.ncol > 0) || return zeros(Float64, m), zeros(Float64, nin)

    # Forward flow: ρ is 1 at every live output position, and γ comes from the transposed sweep
    # seeded there.
    rseed = zeros(Float64, N)
    for (k, o) in enumerate(outidxs)
        li.liverow[k] && (rseed[o] += 1.0)
    end
    gf = _reverse_sweep(ForwardFlow(li), t, rseed)

    # Reverse flow: γ is 1 at every live input, and ρ comes from the forward sweep seeded there,
    # read through the sink slot's own 1/outdeg factor.
    fseed = zeros(Float64, N)
    for j in 1:nin
        li.livecol[j] && (fseed[j] = 1.0)
    end
    fr = _forward_sweep(ReverseFlow(li), t, fseed)

    r, c = zeros(Float64, m), zeros(Float64, nin)
    for (k, o) in enumerate(outidxs)
        li.liverow[k] || continue
        r[k] = 1 / li.nrow + (fr[o] / li.outdeg[o]) / li.ncol
    end
    for j in 1:nin
        li.livecol[j] || continue
        c[j] = gf[j] / li.nrow + 1 / li.ncol
    end
    return r, c
end

"""
    _support_counts(t::Tape, outidxs, li::LiveInfo) -> (r, c)

Exact reachability counts: `r[k]` live inputs reaching output position `k`, `c[j]` live output
positions reached by input `j` — the row and column nonzero counts of `Ũ`, obtained without forming
it. Word-parallel bit propagation over the live slot graph: `cld(n, 64)` forward passes and
`cld(m, 64)` reverse passes, each `O(L)` integer work.
"""
function _support_counts(t::Tape, outidxs, li::LiveInfo)
    nin, m = t.ninputs, length(outidxs)
    r, c = zeros(Float64, m), zeros(Float64, nin)
    _foreach_input_block(t, outidxs, li) do _, mask, outs
        for k in 1:m
            li.liverow[k] && (r[k] += count_ones(mask[outs[k]]))
        end
    end
    _foreach_output_block(t, outidxs, li) do _, mask
        for j in 1:nin
            li.livecol[j] && (c[j] += count_ones(mask[j]))
        end
    end
    return r, c
end

"""
    _structural_pattern(t::Tape, outidxs, li::LiveInfo) -> P::Matrix{UInt64}

The structural sparsity pattern of `Ũ` packed 64 inputs to a word: bit `(j - 1) % 64` of
`P[k, cld(j, 64)]` is set exactly when output position `k` reaches input `j` along a path of
nonzero local derivatives. `cld(n, 64)` integer passes over the tape and `m * cld(n, 64)` words,
both small beside the `m * n` surrogate this is compared against.
"""
function _structural_pattern(t::Tape, outidxs, li::LiveInfo)
    m, nblk = length(outidxs), cld(t.ninputs, 64)
    P = zeros(UInt64, m, nblk)
    _foreach_input_block(t, outidxs, li) do blk, mask, outs
        for k in 1:m
            li.liverow[k] && (P[k, blk] = mask[outs[k]])
        end
    end
    return P
end

# Output nodes by position 1:m, independent of the axes of outidxs.
function _output_nodes(outidxs)
    outs = Vector{Int}(undef, length(outidxs))
    for (k, o) in enumerate(outidxs)
        outs[k] = o
    end
    return outs
end

# One OR-propagation pass per block of 64 inputs, forward through the live slots.
function _foreach_input_block(f, t::Tape, outidxs, li::LiveInfo)
    N, nin = nnodes(t), t.ninputs
    outs = _output_nodes(outidxs)
    mask = Vector{UInt64}(undef, N)
    for blk in 1:cld(nin, 64)
        fill!(mask, UInt64(0))
        for j in ((blk - 1) * 64 + 1):min(blk * 64, nin)
            li.livecol[j] && (mask[j] = UInt64(1) << ((j - 1) % 64))
        end
        for k in (nin + 1):N
            li.live[k] || continue
            ps, ds = t.parents[k], t.ds[k]
            acc = UInt64(0)
            for i in eachindex(ps)
                (li.live[ps[i]] && !iszero(magnitude(ds[i]))) || continue
                acc |= mask[ps[i]]
            end
            mask[k] = acc
        end
        f(blk, mask, outs)
    end
    return nothing
end

# One OR-propagation pass per block of 64 output positions, backward through the live slots.
function _foreach_output_block(f, t::Tape, outidxs, li::LiveInfo)
    N, nin, m = nnodes(t), t.ninputs, length(outidxs)
    outs = _output_nodes(outidxs)
    mask = Vector{UInt64}(undef, N)
    for blk in 1:cld(m, 64)
        fill!(mask, UInt64(0))
        for k in ((blk - 1) * 64 + 1):min(blk * 64, m)
            li.liverow[k] && (mask[outs[k]] |= UInt64(1) << ((k - 1) % 64))
        end
        for k in N:-1:(nin + 1)
            li.live[k] || continue
            mk = mask[k]
            mk == 0 && continue
            ps, ds = t.parents[k], t.ds[k]
            for i in eachindex(ps)
                (li.live[ps[i]] && !iszero(magnitude(ds[i]))) || continue
                mask[ps[i]] |= mk
            end
        end
        f(blk, mask)
    end
    return nothing
end

# The undirected live tape graph in compressed form. Node `v`'s half-edges occupy
# `ptr[v]:ptr[v+1]-1`; each carries the node at the other end of a live parent slot and the value
# `θ_v - θ_u` that slot asks for: `+log|d|` when `v` is the slot's child, `-log|d|` when `v` is its
# parent. Parallel slots between the same two nodes are separate half-edges.
struct LiveGraph{T}
    live::BitVector
    ptr::Vector{Int}
    nbr::Vector{Int}
    val::Vector{T}
end

_degree(g::LiveGraph, v::Integer) = g.ptr[v + 1] - g.ptr[v]

function _live_graph(t::Tape{T}, outidxs::AbstractVector{<:Integer}, live) where {T}
    TL = magtype(T)
    N = nnodes(t)
    ptr = zeros(Int, N + 1)
    for k in (t.ninputs + 1):N
        live[k] || continue
        ps, ds = t.parents[k], t.ds[k]
        for i in eachindex(ps)
            (live[ps[i]] && !iszero(magnitude(ds[i]))) || continue
            ptr[k + 1] += 1
            ptr[ps[i] + 1] += 1
        end
    end
    ptr[1] = 1
    for v in 1:N
        ptr[v + 1] += ptr[v]
    end
    nbr = Vector{Int}(undef, ptr[N + 1] - 1)
    val = Vector{TL}(undef, length(nbr))
    pos = ptr[1:N]
    for k in (t.ninputs + 1):N
        live[k] || continue
        ps, ds = t.parents[k], t.ds[k]
        for i in eachindex(ps)
            p = ps[i]
            (live[p] && !iszero(magnitude(ds[i]))) || continue
            l = log(TL(magnitude(ds[i])))
            nbr[pos[k]] = p
            val[pos[k]] = l
            pos[k] += 1
            nbr[pos[p]] = k
            val[pos[p]] = -l
            pos[p] += 1
        end
    end
    return LiveGraph(live, ptr, nbr, val)
end

# Live nodes ordered by decreasing degree, ties by increasing node index: a counting sort, so the
# traversal that consumes it stays `O(L)`.
function _degree_order(g::LiveGraph)
    N = length(g.live)
    dmax = 0
    nlive = 0
    for v in 1:N
        g.live[v] || continue
        nlive += 1
        dmax = max(dmax, _degree(g, v))
    end
    cnt = zeros(Int, dmax + 2)
    for v in 1:N
        g.live[v] || continue
        cnt[dmax - _degree(g, v) + 2] += 1
    end
    cnt[1] = 1
    cumsum!(cnt, cnt)
    order = Vector{Int}(undef, nlive)
    for v in 1:N
        g.live[v] || continue
        k = dmax - _degree(g, v) + 1
        order[cnt[k]] = v
        cnt[k] += 1
    end
    return order
end

# Label the components of the live graph and pick each one's default root: its node of maximum
# undirected degree, lowest node index on ties. `nties` counts the nodes that share that degree.
function _live_components(g::LiveGraph)
    N = length(g.live)
    comp = zeros(Int, N)
    roots = Int[]
    queue = Vector{Int}(undef, count(g.live))
    for root in _degree_order(g)
        comp[root] == 0 || continue
        push!(roots, root)
        ncomp = length(roots)
        comp[root] = ncomp
        queue[1] = root
        head, tail = 1, 1
        while head <= tail
            u = queue[head]
            head += 1
            for e in g.ptr[u]:(g.ptr[u + 1] - 1)
                w = g.nbr[e]
                comp[w] == 0 || continue
                comp[w] = ncomp
                tail += 1
                queue[tail] = w
            end
        end
    end
    nties = zeros(Int, length(roots))
    for v in 1:N
        g.live[v] || continue
        c = comp[v]
        _degree(g, v) == _degree(g, roots[c]) && (nties[c] += 1)
    end
    return comp, roots, nties
end

# A supplied root replaces the degree rule for its component, so no tie was broken there and the
# component's tie count becomes 1.
function _resolve_roots(root, g::LiveGraph, comp, defroots, nties)
    root === nothing && return defroots, nties
    given = root isa Integer ? (Int(root),) : root
    roots = copy(defroots)
    counts = copy(nties)
    seen = falses(length(roots))
    for v in given
        1 <= v <= length(g.live) && g.live[v] ||
            throw(ArgumentError("root $v is not a live node of the tape"))
        c = comp[v]
        seen[c] &&
            throw(ArgumentError("two roots were given for live component $c; supply at most one per component"))
        seen[c] = true
        roots[c] = v
        counts[c] = 1
    end
    return roots, counts
end

# Breadth-first layers from each root, averaging over the slots that join a node to the preceding
# layer. Parallel slots are separate terms.
function _layer_average!(theta, g::LiveGraph, roots)
    N = length(g.live)
    dist = fill(-1, N)
    queue = Vector{Int}(undef, count(g.live))
    for rt in roots
        dist[rt] = 0
        theta[rt] = 0
        queue[1] = rt
        head, tail = 1, 1
        while head <= tail
            u = queue[head]
            head += 1
            for e in g.ptr[u]:(g.ptr[u + 1] - 1)
                w = g.nbr[e]
                dist[w] < 0 || continue
                dist[w] = dist[u] + 1
                tail += 1
                queue[tail] = w
            end
        end
        # Breadth-first order, so a node's preceding layer is already valued when it is reached.
        for q in 2:tail
            v = queue[q]
            tot = zero(eltype(theta))
            cnt = 0
            for e in g.ptr[v]:(g.ptr[v + 1] - 1)
                u = g.nbr[e]
                dist[u] == dist[v] - 1 || continue
                tot += theta[u] + g.val[e]
                cnt += 1
            end
            cnt > 0 || error("live node $v has no neighbor in the preceding breadth-first layer")
            theta[v] = tot / cnt
        end
    end
    return theta
end

# y = L*x for the graph Laplacian of the live graph, in O(L).
function _laplacian!(y, g::LiveGraph, x)
    for v in eachindex(y)
        acc = zero(eltype(y))
        for e in g.ptr[v]:(g.ptr[v + 1] - 1)
            acc += x[g.nbr[e]]
        end
        y[v] = _degree(g, v) * x[v] - acc
    end
    return y
end

# Euclidean norm of the slot residuals `θ_child - θ_parent - log|d|`. Each slot contributes two
# half-edges carrying the same magnitude.
function _slot_residual_norm(theta, g::LiveGraph)
    acc = zero(eltype(theta))
    for v in eachindex(theta)
        for e in g.ptr[v]:(g.ptr[v + 1] - 1)
            acc += (theta[v] - theta[g.nbr[e]] - g.val[e])^2
        end
    end
    return sqrt(acc / 2)
end

# Conjugate gradients on the normal equations of the slot least-squares problem
# `θ_child - θ_parent = log|d|`. The Laplacian is singular along one constant per component and the
# right-hand side is orthogonal to that constant, so CG converges on the complement.
#
# Two things stop the iteration, both determined by the graph and the invariant residuals alone,
# so the stopping point is invariant under rescaling. The residual floor catches the ordinary case.
# The Krylov bound `nlive - ncomp`, the rank of the Laplacian, catches the case where the start is
# already a solution and the floor is therefore at roundoff: past exact termination the iteration
# divides one roundoff-level quantity by another and the potentials drift along the nullspace,
# losing the covariance they started with.
function _refine_potentials!(theta, g::LiveGraph, cgiters::Integer, ncomp::Integer)
    TL = eltype(theta)
    N = length(g.live)
    cgiters = min(cgiters, max(0, count(g.live) - ncomp))
    x = TL[g.live[v] ? theta[v] : zero(TL) for v in 1:N]
    rhs = zeros(TL, N)
    for v in 1:N
        for e in g.ptr[v]:(g.ptr[v + 1] - 1)
            rhs[v] += g.val[e]
        end
    end
    y = similar(x)
    res = rhs .- _laplacian!(y, g, x)
    rnorm0 = norm(res)
    rfloor = eps(TL)^(3 // 4) * max(rnorm0, _slot_residual_norm(theta, g))
    rnorm0 > rfloor || return theta
    p = copy(res)
    rs = dot(res, res)
    for _ in 1:cgiters
        _laplacian!(y, g, p)
        pAp = dot(p, y)
        pAp > 0 || break                       # the search direction carries no curvature left
        alpha = rs / pAp
        @. x += alpha * p
        @. res -= alpha * y
        rsnew = dot(res, res)
        sqrt(rsnew) <= rfloor && break
        @. p = res + (rsnew / rs) * p
        rs = rsnew
    end
    for v in 1:N
        g.live[v] && (theta[v] = x[v])
    end
    return theta
end

"""
    tape_potentials(t::Tape, outidxs::AbstractVector{<:Integer}; root=nothing, cgiters=0)
        -> (; theta, live, roots, nties)

Log-scale potentials on the live tape graph: one number per node, least-squares-fitting the slot
equations `θ_child - θ_parent = log|d|`. `exp(θ)` at the output nodes is the start
[`balance_cover`](@ref) uses for `start = :covariant` on a tape, because rescaling a node's value
by `s` shifts its potential by `log s` and leaves every difference within a component covariant.

A node is *live* when it lies on some input-to-output path whose local derivatives are all nonzero;
`live` is that mask over the tape's nodes, and `theta` is `NaN` outside it. The live nodes and slots
form an undirected graph (parallel slots counted separately) whose components are rooted at their
node of maximum undirected degree, lowest node index on ties. `roots` lists those node indices and
`nties[c]` how many nodes of component `c` share the root's degree, so a caller can see when the
root came down to the index rather than the graph. `root` overrides the rule with a node index or a
vector of them, at most one per component and each live; components with no supplied root keep the
default. A supplied root breaks no tie, so its component reports `nties[c] = 1`.

`theta` is the breadth-first layer average from each root: `θ = 0` at the root, and a node in layer
`d` takes the mean over the slots joining it to layer `d-1` of `θ_u ± log|d|`, `+` when it is the
slot's child. `cgiters > 0` refines that with up to this many conjugate-gradient iterations on the
normal equations, stopping earlier at a residual floor or at the Laplacian's rank, both of which
are invariant under rescaling; past those the iteration only amplifies roundoff along the
nullspace. The exact least-squares potentials are independent of the root up to one additive
constant per component.

Costs `O(L)` for `cgiters = 0` and `O(L)` per refinement iteration, with one logarithm per live
slot and no derivative evaluations. The fit weights tape slots equally, so like the surrogate
itself it depends on how the program is written.

# Examples

```jldoctest
julia> r(x) = [2 * x[1] * x[2], 4 * x[2] * x[3]];

julia> pot = DerivativeCovers.tape_potentials(tape_function(r, [1.0, 1.0, 1.0])...);

julia> pot.nties, all(isnan, pot.theta[.!pot.live])
([5], true)
```
"""
function tape_potentials(
        t::Tape{T}, outidxs::AbstractVector{<:Integer}; root = nothing, cgiters::Integer = 0
    ) where {T}
    return _tape_potentials(t, outidxs; root, cgiters)[1]
end

function tape_potentials(r, x0::AbstractVector; kwargs...)
    t, outidxs = tape_function(r, x0)
    outidxs isa AbstractVector ||
        throw(ArgumentError("tape_potentials needs a vector-valued program; got a scalar output"))
    return tape_potentials(t, outidxs; kwargs...)
end

# `balance_cover`'s covariant start needs the components for its per-component gauge shift, but
# they are not part of the result, so they are handed back separately.
function _tape_potentials(
        t::Tape{T}, outidxs::AbstractVector{<:Integer}, li::LiveInfo = _live_info(t, outidxs);
        root = nothing, cgiters::Integer = 0
    ) where {T}
    cgiters >= 0 || throw(ArgumentError("cgiters must be nonnegative, got $cgiters"))
    TL = magtype(T)
    g = _live_graph(t, outidxs, li.live)
    comp, defroots, defties = _live_components(g)
    roots, nties = _resolve_roots(root, g, comp, defroots, defties)
    theta = fill(TL(NaN), nnodes(t))
    _layer_average!(theta, g, roots)
    cgiters > 0 && _refine_potentials!(theta, g, cgiters, length(roots))
    return (; theta, live = g.live, roots, nties), comp
end
