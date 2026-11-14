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

The derivation starts from a *drop pass*: a sweep in which the guarded blocks contribute only their
diagonal curvature, so that its result `h_drop` depends on no declared scale. An input with
`h_drop[i] > 0` is *anchored* and receives `sigma[i] = 1 / h_drop[i]`. An input with
`h_drop[i] == 0` receives curvature only through guarded blocks, so the program says nothing about
its scale relative to the anchored inputs. Such an input is harmless where its scale cancels from
every guarded split, as when both parents of each block it reaches depend on it alone; otherwise
the derivation throws, naming the inputs, and the caller must declare scales explicitly with
`scales = w`.

The scales from the drop pass define round 1. Each later round re-derives the anchored scales from
the sum-merge cover computed with the previous round's scales; the scales of unanchored inputs
stay fixed, since they would otherwise feed on the guarded blocks that are their only curvature.

  - `rounds = 1`: run exactly this many rounds, with no convergence test.
  - `maxiter = 100`, `rtol = eps^(2/3)`: naming either, or passing `rounds = nothing`, runs rounds
    until the anchored entries of successive covers change by at most `rtol` relative, and throws
    if `maxiter` rounds do not reach that. `eps` is that of [`magtype`](@ref) of the tape's element
    type. Passing `maxiter` or `rtol` together with a `rounds` count is an error.

Cost: the drop pass is the entry point's own cover sweep, so a program with no guarded block costs
exactly what it costs without `CurvatureScales`, and its result has `nguarded == 0`. On a program
with guarded blocks, `rounds = 1` adds one cover sweep, about doubling the cost of the cover; each
further round adds another. Checking unanchored inputs, when there are any, takes one forward
sweep plus one per such input. Otherwise `nguarded` counts the blocks whose split read the derived
scales.

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
    resolve_scales(scales, t::Tape, vbar_or_outidx, ratio, check) -> w

The declared input scales to run the split with: `scales` itself, or for a
[`CurvatureScales`](@ref) the scales [`derived_scales`](@ref) finds after a drop pass over the
adjoints `vbar` (computed from `outidx` when an output index is passed), or `nothing` when that
pass guards no block. `ratio` is the resolved threshold of [`split_ratio`](@ref).

The sweep entry points do not call this: they run the drop pass as their own sweep and keep its
result when it guards no block, which leaves the cost of such a program unchanged (see
`cover_only_channels!` and [`local_covers`](@ref)). This serves the computations that
need the scales before a sweep of their own.
"""
resolve_scales(w, t::Tape, vbar_or_outidx, ratio, check::Bool) = w
resolve_scales(cs::CurvatureScales, t::Tape, outidx::Integer, ratio, check::Bool) =
    resolve_scales(cs, t, adjoints(t, outidx), ratio, check)
function resolve_scales(cs::CurvatureScales, t::Tape, vbar::AbstractVector, ratio, check::Bool)
    drop = DropGuarded()
    abar, _ = cover_only_sweep!(Val(false), vbar, t, ratio, split_scales(drop, t), check)
    isempty(drop.pairs) && return nothing
    return derived_scales(cs, t, vbar, abar, drop.pairs, ratio, check)
end

# The drop pass is a sweep of the entry point's own kind, and where it guards no block it read no
# declared scale, so its result is the cover under any scales: a program without guarded blocks
# pays for one sweep, as it would without `CurvatureScales`.
function cover_only_channels!(p::Val, vbar, t::Tape, r, cs::CurvatureScales, check::Bool)
    drop = DropGuarded()
    abar, rbar, nguarded = cover_only_channels!(p, vbar, t, r, drop, check)
    isempty(drop.pairs) && return abar, rbar, nguarded
    # A propagating sweep has finished the adjoints in `vbar`, so the rest only reads them.
    w = derived_scales(cs, t, vbar, abar, drop.pairs, r, check)
    return cover_only_channels!(Val(false), vbar, t, r, w, check)
end

function local_covers(t::Tape, vbar::AbstractVector, r, cs::CurvatureScales, check::Bool)
    drop = DropGuarded()
    res = local_covers(t, vbar, r, drop, check)
    isempty(drop.pairs) && return res
    hdrop = abs_reverse(t, res[2])
    return local_covers(t, vbar, r, derived_scales(cs, t, vbar, hdrop, drop.pairs, r, check), check)
end

"""
    derived_scales(cs::CurvatureScales, t::Tape, vbar, hdrop, pairs, ratio, check) -> Vector

The declared input scales [`CurvatureScales`](@ref) describes, in
[`ExpScaled`](@ref)`{magtype(T)}`, from a completed drop pass over the program `t` with node
adjoints `vbar`: `hdrop[1:t.ninputs]` is its sum-merge cover and `pairs` the parents of the blocks
it dropped (see [`DropGuarded`](@ref)), of which there must be at least one.
"""
function derived_scales(
        cs::CurvatureScales, t::Tape{T}, vbar::AbstractVector, hdrop::AbstractVector, pairs,
        ratio, check::Bool
    ) where {T}
    n = t.ninputs
    M = magtype(T)
    E = ExpScaled{M}
    anchored = [plainvalue(hdrop[i]) > 0 for i in 1:n]
    sigma = E[anchored[i] ? inv(E(plainvalue(hdrop[i]))) : one(E) for i in 1:n]
    all(anchored) || check_unanchored(t, sigma, anchored, pairs)
    sweep(sigma) = tomag(T, cover_only_sweep!(Val(false), vbar, t, ratio, split_scales(sigma, t), check)[1][1:n])
    rederive!(sigma, h) = for i in 1:n
        anchored[i] && (sigma[i] = inv(E(h[i])))
    end
    if cs.rounds !== nothing
        for _ in 2:cs.rounds
            rederive!(sigma, sweep(sigma))
        end
        return sigma
    end
    rtol = cs.rtol === nothing ? eps(M)^(2 // 3) : cs.rtol
    hprev = sweep(sigma)
    rederive!(sigma, hprev)
    delta = M(Inf)
    for _ in 2:cs.maxiter
        h = sweep(sigma)
        rederive!(sigma, h)
        delta = maximum(i -> anchored[i] ? abs(h[i] - hprev[i]) / h[i] : zero(M), 1:n)
        delta <= rtol && return sigma
        hprev = h
    end
    error(
        "CurvatureScales did not converge in $(cs.maxiter) rounds: the anchored cover entries " *
            "last changed by $delta relative, above rtol = $rtol"
    )
end

"""
    check_unanchored(t::Tape, sigma, anchored, pairs)

Throw unless every guarded block, given by its parent nodes in `pairs`, splits the same way
whatever scales the unanchored inputs (`anchored[i] == false`) are given.

A parent's propagated scale is linear in the input scales, `a_p + sum_j b_pj * sigma_j` over the
unanchored inputs `j`, with `a_p` from the anchored scales in `sigma`. The split reads the ratio of
the two parents' scales, which is independent of every `sigma_j` exactly when the coefficient
vectors `(a_p, b_p)` of the two parents are proportional. The test allows a relative discrepancy
of `sqrt(eps)`, as the covering checks do.
"""
function check_unanchored(t::Tape{T}, sigma::AbstractVector{E}, anchored, pairs) where {T, E}
    n, N = t.ninputs, nnodes(t)
    M = magtype(T)
    theta = zeros(E, N)
    for i in 1:n
        anchored[i] && (theta[i] = sigma[i])
    end
    a = abs_forward(t, theta)
    U = findall(!, anchored)
    b = map(U) do j
        fill!(theta, zero(E))
        theta[j] = one(E)
        abs_forward(t, theta)
    end
    tol = sqrt(eps(M))
    coefficients(p) = [a[p]; [bj[p] for bj in b]]
    bad = Set{Tuple{Int, Int}}()
    for (p1, p2) in pairs
        proportional(coefficients(p1), coefficients(p2), tol) || push!(bad, (p1, p2))
    end
    isempty(bad) && return nothing
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
