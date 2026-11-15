# Declared scales derived from the cover's own curvature.

"""
    CurvatureScales(; rounds=1)
    CurvatureScales(; maxiter=100, rtol=eps^(2/3))

A value for the `scales` keyword of the tape-based cover functions ([`cover_hessian`](@ref),
[`cover_nls`](@ref), and those built on them) that derives the declared input scales from the
program instead of taking them from the caller.

The declared scales matter only at blocks where the parent values carry no scale (see
[`split_block`](@ref)), and there the split depends only on the *ratio* of the two parents'
propagated scales. Covers have units: under `x -> D*x` and `f -> c*f` a cover transforms as
`h -> sqrt(c) * inv(D) * h`, so `sigma = 1 ./ h` transforms as the inputs do, up to the factor
`1/sqrt(c)` common to every input, which cancels from every ratio. Scales derived this way
therefore make the cover covariant at the guarded blocks as well.

Each cover derives its scales from covers of its own kind: the sum cover from sum covers, and the
sign-averaged cover (`merge = SignAveragedMerge()`) from sign-averaged covers, which transform in
the same way. [`cover_nls`](@ref) derives them from the Hessian cover of that kind of its loss
tape. The two covers of one program can therefore split a guarded block differently.

The derivation starts from a *drop pass*: a sweep in which the guarded blocks contribute only their
diagonal curvature, so that its result `h_drop` depends on no declared scale. An input with
`h_drop[i] > 0` is *anchored* and receives `sigma[i] = 1 / h_drop[i]`. An input with
`h_drop[i] == 0` receives curvature only through guarded blocks, so the program says nothing about
its scale relative to the anchored inputs. Such an input is harmless where its scale cancels from
every guarded split, as when both parents of each block it reaches depend on it alone; otherwise
the derivation throws, naming the inputs, and the caller must declare scales explicitly with
`scales = w`.

The scales from the drop pass define round 1. Each later round re-derives the anchored scales from
the cover computed with the previous round's scales; the scales of unanchored inputs
stay fixed, since they would otherwise feed on the guarded blocks that are their only curvature.

  - `rounds = 1`: run exactly this many rounds, with no convergence test.
  - `maxiter = 100`, `rtol = eps^(2/3)`: naming either, or passing `rounds = nothing`, runs rounds
    until the anchored entries of successive covers change by at most `rtol` relative, and throws
    if `maxiter` rounds do not reach that. `eps` is that of [`magtype`](@ref) of the tape's element
    type. Passing `maxiter` or `rtol` together with a `rounds` count is an error.

Cost: the drop pass is the entry point's own cover sweep, so a program with no guarded block costs
exactly what it costs without `CurvatureScales`, and its result has `nguarded == 0`. Otherwise
only the guarded blocks read the scales, so each round, the one that returns the cover included,
splits those blocks alone and adds the change to the drop pass's cover in one reverse sweep (see
[`GuardedRedo`](@ref)), after the forward sweep that propagates the scales. Checking unanchored
inputs, when there are any, takes one forward sweep plus one per such input, which together also
give the first round's propagated scales. `nguarded` counts the blocks whose split read the
derived scales.

[`forward_cover`](@ref) records no tape to sweep and does not accept `CurvatureScales`.
"""
struct CurvatureScales{R}
    rounds::Union{Nothing, Int}
    maxiter::Union{Nothing, Int}
    rtol::R
end

function CurvatureScales(; rounds = missing, maxiter = nothing, rtol = nothing)
    rounds === missing && (rounds = maxiter === nothing && rtol === nothing ? 1 : nothing)
    if rounds === nothing
        m = something(maxiter, 100)
        m >= 2 || throw(
            ArgumentError(
                "maxiter must be at least 2: convergence compares the covers of successive " *
                    "rounds (got maxiter=$m; for a single round pass rounds=1)"
            )
        )
        rtol === nothing || (isfinite(rtol) && rtol > 0) ||
            throw(ArgumentError("rtol must be positive and finite, got $rtol"))
        return CurvatureScales(nothing, Int(m), rtol)
    end
    for (name, v) in (("maxiter", maxiter), ("rtol", rtol))
        v === nothing || throw(
            ArgumentError(
                "$name tunes a run to convergence and rounds is a fixed budget, so they cannot " *
                    "be combined; got $name=$v with rounds=$rounds (pass rounds=nothing to converge)"
            )
        )
    end
    rounds >= 1 || throw(
        ArgumentError(
            "rounds must be at least 1: the drop pass that precedes round 1 leaves the guarded " *
                "blocks uncovered (got rounds=$rounds)"
        )
    )
    return CurvatureScales(Int(rounds), nothing, nothing)
end

check_input_scales(::CurvatureScales, ninputs::Integer) = throw(
    ArgumentError(
        "CurvatureScales derives the declared scales from reverse sweeps over a recorded tape, " *
            "which this computation does not have; pass explicit scales instead"
    )
)

"""
    resolve_scales(scales, t::Tape, vbar_or_outidx, ratio, check, merge=SumMerge()) -> w

The declared input scales to run the split with: `scales` itself, or for a
[`CurvatureScales`](@ref) the scales [`derived_scales`](@ref) finds for the cover `merge` selects
after a drop pass over the adjoints `vbar` (computed from `outidx` when an output index is
passed), or `nothing` when that pass guards no block. `ratio` is the resolved threshold of
[`split_ratio`](@ref).

The sweep entry points do not call this: they run the drop pass as their own sweep and keep its
result when it guards no block, which leaves the cost of such a program unchanged (see
`cover_only_channels!` and [`local_covers`](@ref)). This serves the computations that
need the scales before a sweep of their own.
"""
resolve_scales(w, t::Tape, vbar_or_outidx, ratio, check::Bool, merge::SweepMerge = SumMerge()) = w
resolve_scales(
    cs::CurvatureScales, t::Tape, outidx::Integer, ratio, check::Bool, merge::SweepMerge = SumMerge()
) = resolve_scales(cs, t, adjoints(t, outidx), ratio, check, merge)
function resolve_scales(
        cs::CurvatureScales, t::Tape{T}, vbar::AbstractVector, ratio, check::Bool,
        merge::SweepMerge = SumMerge()
    ) where {T}
    drop = DropGuarded{magtype(T)}()
    sc = split_scales(drop, t)
    C = cover_only_sweep!(Val(false), merge, vbar, t, ratio, sc, check)
    isempty(drop.blocks) && return nothing
    redo = GuardedRedo(merge, t, vbar, ratio, merged_channels(merge, C), drop, sc.store.lives, check)
    return first(derived_scales(cs, redo))
end

# The drop pass is a sweep of the entry point's own kind, and where it guards no block it read no
# declared scale, so its result is the cover under any scales: a program without guarded blocks
# pays for one sweep, as it would without `CurvatureScales`. The scales derive from the cover of
# all channels merged, the Hessian cover of the program, whatever the channels of `layout`.
function cover_only_channels!(
        p::Val, merge::SweepMerge, vbar, t::Tape{T}, r, cs::CurvatureScales, check::Bool,
        layout = OneChannel()
    ) where {T}
    drop = DropGuarded{magtype(T)}()
    sc = split_scales(drop, t)
    C = cover_only_sweep!(p, merge, vbar, t, r, sc, check, layout)
    isempty(drop.blocks) && return C, sc.store.nguarded
    # A propagating sweep has finished the adjoints in `vbar`, so the rest only reads them.
    redo = GuardedRedo(merge, t, vbar, r, merged_channels(merge, C), drop, sc.store.lives, check)
    sigma, S = derived_scales(cs, redo)
    etas, nguarded = guarded_covers(redo, sigma, S)
    return raise_channels!(C, redo, etas, layout), nguarded
end

# The cover of all the channels `C` of a sweep merged into one, at every node. Both covers are
# linear in their seeds, the sum cover in the seeds and the sign-averaged cover in their squares,
# so this is the cover of a single-channel sweep with the same seeds.
merged_channels(::SumMerge, C::AbstractMatrix) = [sum(view(C, :, k)) for k in axes(C, 2)]
merged_channels(::SignAveragedMerge, C::AbstractMatrix) = [foldl(hypot, view(C, :, k)) for k in axes(C, 2)]

function local_covers(
        t::Tape{T}, vbar::AbstractVector, r, cs::CurvatureScales, check::Bool,
        merge::MergeRule = SumMerge()
    ) where {T}
    check_sweep_merge(merge)
    drop = DropGuarded{magtype(T)}()
    sc = split_scales(drop, t)
    etas, theta, nguarded = local_covers!(t, vbar, r, sc, check)
    isempty(drop.blocks) && return etas, theta, nguarded
    redo = GuardedRedo(merge, t, vbar, r, drop_cover(merge, t, etas, theta), drop, sc.store.lives, check)
    sigma, S = derived_scales(cs, redo)
    guarded, nguarded = guarded_covers(redo, sigma, S)
    for (j, k) in enumerate(redo.nodes)
        etas[k] = guarded[j]
    end
    return etas, seeds_from(t, etas), nguarded
end

# The drop pass's cover of the kind `merge` selects, merged from its local covers.
drop_cover(::SumMerge, t::Tape, etas, theta) = abs_reverse(t, theta)
drop_cover(::SignAveragedMerge, t::Tape, etas, theta) =
    _reverse_sweep(hypot, _absslot, t, quadrature_seeds(t, etas))

"""
    GuardedRedo(merge, t, vbar, ratio, hdrop, drop::DropGuarded, lives, check)

What a cover under derived scales shares with its drop pass: the drop pass's cover `hdrop` of the
kind `merge` selects, over all nodes, and the `nodes` whose blocks it dropped, in tape order, with
the slot covers `etas` it gave them, and the parent nodes `pairs` of each dropped block (all from
`drop`). Only those nodes read the declared scales, so [`guarded_covers`](@ref) recomputes theirs
alone, and [`raised_cover`](@ref) adds the change to `hdrop` in one reverse sweep. `lives` is the
drop pass's [`live_nodes`](@ref), which does not depend on the scales. `cone` is the
[`ancestor_cone`](@ref) of `pairs`, the only nodes whose declared scales a split reads.
"""
struct GuardedRedo{R <: SweepMerge, TT <: Tape, V, Q, H, M}
    merge::R
    tape::TT
    vbar::V
    ratio::Q
    hdrop::H
    nodes::Vector{Int}
    etas::Vector{Vector{M}}
    pairs::Vector{Tuple{Int, Int}}
    lives::Vector{Bool}
    cone::Vector{Bool}
    check::Bool
end
function GuardedRedo(merge, t, vbar, ratio, hdrop, drop::DropGuarded{M}, lives, check) where {M}
    nodes, etas = Int[], Vector{M}[]
    # The blocks of one node, such as the pairs of an inner product, are recorded consecutively.
    for b in drop.blocks
        isempty(nodes) || last(nodes) != b.node || continue
        b.eta === nothing && error("the drop pass recorded the block with parents $(b.parents) but not its node")
        push!(nodes, b.node)
        push!(etas, b.eta)
    end
    pairs = [b.parents for b in drop.blocks]
    cone = ancestor_cone(t, (p for pr in pairs for p in pr))
    return GuardedRedo(merge, t, vbar, ratio, hdrop, nodes, etas, pairs, lives, cone, check)
end

"""
    guarded_covers(redo::GuardedRedo, sigma, S) -> (etas, nguarded)

The slot covers of the nodes of `redo` under the declared input scales `sigma`, and the number of
blocks that read them. `S`, when not `nothing`, is the [`sensitivity_scales`](@ref) of `sigma`,
already computed.
"""
function guarded_covers(redo::GuardedRedo, sigma, S)
    t = redo.tape
    sc = split_scales(sigma, t)
    isempty(redo.lives) || (sc.store.lives = redo.lives)
    sc.store.cone = redo.cone
    S === nothing || (sc.store.sigmas = S)
    etas = map(redo.nodes) do k
        eta = node_cover(redo.ratio, t, k, redo.vbar[k], sc, redo.check)
        eta === nothing && error("node $k dropped a block in the drop pass but has no cover under declared scales")
        eta
    end
    return etas, sc.store.nguarded
end

"""
    raised_cover(redo::GuardedRedo, etas) -> h

The cover at the inputs when the nodes of `redo` take the slot covers `etas` in place of the drop
pass's. Each cover is linear in its seeds, the sum cover in the seeds themselves and the
sign-averaged cover in their squares, so the change propagates from the recorded nodes' parents
alone and adds to `hdrop`. A split weight is never below the drop pass's weight for the same slot
(see [`balanced_split`](@ref)), so every change is nonnegative.
"""
function raised_cover(redo::GuardedRedo, etas)
    delta = cover_increase(redo, etas, eachindex(redo.nodes))
    return [combine_channels(redo.merge, redo.hdrop[i], delta[i]) for i in 1:redo.tape.ninputs]
end

# The channels `C` of the drop pass, raised in place at the inputs to the slot covers `etas` of
# the nodes of `redo`. Each node raises the channel of `layout` that it seeds, in one reverse sweep
# per channel that any of them seeds.
function raise_channels!(C::AbstractMatrix, redo::GuardedRedo, etas, layout)
    gs = [channel_of(layout, k) for k in redo.nodes]
    for g in unique(gs)
        delta = cover_increase(redo, etas, findall(==(g), gs))
        for i in 1:redo.tape.ninputs
            C[g, i] = combine_channels(redo.merge, C[g, i], delta[i])
        end
    end
    return C
end

# The change, at every node, of the cover seeded by the nodes `redo.nodes[js]` when they take the
# slot covers `etas[js]` in place of the drop pass's.
function cover_increase(redo::GuardedRedo, etas, js)
    t = redo.tape
    T = eltype(t)
    merge = redo.merge
    rho = zeros(channeltype(merge, T), nnodes(t))
    for j in js
        k = redo.nodes[j]
        e0, e1 = redo.etas[j], etas[j]
        for (i, p) in enumerate(t.parents[k])
            rho[p] = raise_seed(merge, T, rho[p], e0[i], e1[i], k, redo.check)
        end
    end
    return _reverse_sweep(channel_accumulator(merge), _absslot, t, rho)
end

# A slot's seed increase from `e0` to `e1`. Rounding can leave a split weight a hair below the
# diagonal weight it bounds; such a slot keeps its drop-pass weight, which covers it. `check`
# verifies that no weight falls further.
function raise_seed(merge::SweepMerge, ::Type{T}, acc, e0, e1, k, check::Bool) where {T}
    check && (e1 >= e0 * (1 - sqrt(eps(typeof(e0)))) ||
        error("node $k: a split weight $e1 under declared scales is below the drop pass's weight $e0"))
    e1 > e0 || return acc
    return raise_seed(merge, T, acc, e0, e1)
end
raise_seed(::SumMerge, ::Type{T}, acc, e0, e1) where {T} = acc + boundsub(T, e1, e0)
# The root of `e1^2 - e0^2`, as the product of two roots, so no square is formed.
raise_seed(::SignAveragedMerge, ::Type{T}, acc, e0, e1) where {T} = hypot(acc, sqrt(e1 - e0) * sqrt(e1 + e0))

channel_accumulator(::SumMerge) = +
channel_accumulator(::SignAveragedMerge) = hypot
combine_channels(::SumMerge, h, d) = h + d
combine_channels(::SignAveragedMerge, h, d) = hypot(h, d)

# One round of the derivation: the cover at the inputs under the declared scales `sigma`.
(redo::GuardedRedo)(sigma, S) =
    tomag(eltype(redo.tape), raised_cover(redo, first(guarded_covers(redo, sigma, S))))

"""
    derived_scales(cs::CurvatureScales, redo::GuardedRedo) -> (sigma, S)

The declared input scales [`CurvatureScales`](@ref) describes, in
[`ExpScaled`](@ref)`{magtype(T)}`, from a completed drop pass over the program `redo.tape`:
`redo.hdrop[1:ninputs]` is its cover and `redo.pairs` the parents of the blocks it dropped (see
[`DropGuarded`](@ref)), of which there must be at least one. `redo(sigma, S)` returns the cover
at the inputs under the declared scales `sigma`, of the same kind as `redo.hdrop`, and is called
once per round after the first. `S` is the [`sensitivity_scales`](@ref) of the returned `sigma` on
`redo.cone` where the check of unanchored inputs has already formed them, and `nothing`
otherwise.
"""
function derived_scales(cs::CurvatureScales, redo::GuardedRedo)
    t, hdrop = redo.tape, redo.hdrop
    T = eltype(t)
    n = t.ninputs
    M = magtype(T)
    E = ExpScaled{M}
    anchored = [plainvalue(hdrop[i]) > 0 for i in 1:n]
    sigma = E[anchored[i] ? inv(E(plainvalue(hdrop[i]))) : one(E) for i in 1:n]
    # Rederiving changes `sigma`, after which these no longer belong to it.
    S = all(anchored) ? nothing : check_unanchored(t, sigma, anchored, redo.pairs, redo.cone)
    rederive!(sigma, h) = for i in 1:n
        anchored[i] && (sigma[i] = inv(E(h[i])))
    end
    if cs.rounds !== nothing
        for _ in 2:cs.rounds
            rederive!(sigma, redo(sigma, S))
            S = nothing
        end
        return sigma, S
    end
    rtol = cs.rtol === nothing ? eps(M)^(2 // 3) : cs.rtol
    hprev = redo(sigma, S)
    rederive!(sigma, hprev)
    delta = M(Inf)
    for _ in 2:cs.maxiter
        h = redo(sigma, nothing)
        rederive!(sigma, h)
        delta = maximum(i -> anchored[i] ? abs(h[i] - hprev[i]) / h[i] : zero(M), 1:n)
        delta <= rtol && return sigma, nothing
        hprev = h
    end
    error(
        "CurvatureScales did not converge in $(cs.maxiter) rounds: the anchored cover entries " *
            "last changed by $delta relative, above rtol = $rtol"
    )
end

"""
    check_unanchored(t::Tape, sigma, anchored, pairs, cone) -> S

Throw unless every guarded block, given by its parent nodes in `pairs`, splits the same way
whatever scales the unanchored inputs (`anchored[i] == false`) are given. Otherwise return
`S = a + sum_j b_j`, the [`sensitivity_scales`](@ref) of `sigma`, whose unanchored entries are one.
The sweeps visit only the nodes of `cone`, the [`ancestor_cone`](@ref) of `pairs`.

A parent's propagated scale is linear in the input scales, `a_p + sum_j b_pj * sigma_j` over the
unanchored inputs `j`, with `a_p` from the anchored scales in `sigma`. The split reads the ratio of
the two parents' scales, which is independent of every `sigma_j` exactly when the coefficient
vectors `(a_p, b_p)` of the two parents are proportional. The test allows a relative discrepancy
of `sqrt(eps)`, as the covering checks do.
"""
function check_unanchored(t::Tape{T}, sigma::AbstractVector{E}, anchored, pairs, cone) where {T, E}
    n, N = t.ninputs, nnodes(t)
    M = magtype(T)
    theta = zeros(E, N)
    for i in 1:n
        anchored[i] && (theta[i] = sigma[i])
    end
    a = abs_forward(t, theta, cone)
    U = findall(!, anchored)
    b = map(U) do j
        fill!(theta, zero(E))
        theta[j] = one(E)
        abs_forward(t, theta, cone)
    end
    tol = sqrt(eps(M))
    coefficients(p) = [a[p]; [bj[p] for bj in b]]
    bad = Set{Tuple{Int, Int}}()
    for (p1, p2) in pairs
        proportional(coefficients(p1), coefficients(p2), tol) || push!(bad, (p1, p2))
    end
    isempty(bad) && return foldl((x, y) -> x .+ y, b; init = a)
    inputs = sort!([j for (k, j) in enumerate(U) if any(((p1, p2),) -> b[k][p1] > 0 || b[k][p2] > 0, bad)])
    nodes = sort!(unique!([p for pr in bad for p in pr]))
    throw(
        ArgumentError(
            "CurvatureScales: input(s) $inputs receive curvature only through blocks whose " *
                "split falls back on declared scales (parent nodes $nodes), so the program does " *
                "not determine their scales relative to the other inputs. Pass declared scales " *
                "explicitly, `scales = w`, for this computation."
        )
    )
end

# Whether `y` is a positive multiple of `x`, up to a relative discrepancy `tol` per entry.
function proportional(x::AbstractVector, y::AbstractVector, tol)
    i = argmax(x)
    iszero(x[i]) && return all(iszero, y)
    rho = y[i] / x[i]
    return all(eachindex(x, y)) do k
        u, v = rho * x[k], y[k]
        abs(u - v) <= tol * (u + v)
    end
end
