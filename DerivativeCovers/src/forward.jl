# A Hessian cover propagated forward, in one pass and with no tape.

"""
    ForwardCover{T}

The result of [`forward_cover`](@ref), with `T` as the propagation's element type.

- `grad`: the gradient, represented by the exact signed tangent at the output.
- `h_quad`: a certified Hessian cover formed by merging the cover atoms in quadrature.
"""
struct ForwardCover{T}
    grad::Vector{T}
    h_quad::Vector{T}
end

"""
    ForwardContext{T, R}

State shared by every variable in a forward propagation: the [`SplitRule`](@ref) followed by local
covers, the input count used to size channels, whether each local cover is verified, and a scratch
vector reused during atom assembly.
"""
struct ForwardContext{T, R <: SplitRule}
    rule::R
    ninputs::Int
    check::Bool
    scratch::Vector{T}
end

"""
    ForwardChannels{T, R}

The two channels an [`FVar`](@ref) carries beyond its value: the signed tangent `g = ∇v`, and the
squared cover `G` maintaining `sqrt.(G) * sqrt.(G)' >= abs.(∇²v)` entrywise.

`G` is squared rather than rooted because the recursion that maintains it is *linear* in the
parents' `G`, which is what keeps a per-node re-merge from compounding with depth.
"""
struct ForwardChannels{T, R}
    ctx::ForwardContext{T, R}
    g::Vector{T}
    G::Vector{T}
end

"""
    FVar{T, R}

A scalar of type `T` that propagates a Hessian cover forward. This is the tape-free
[`CoverVar`](@ref) backend.

`R` is the active [`SplitRule`](@ref) type, which a constant must carry along with `T`
because generic numerical code builds typed literals from `eltype`.

For each variable, the propagation holds the value, the signed tangent `∇v`, and a squared cover of
`∇²v`; see [`ForwardChannels`](@ref). Working memory is `O(n)` per *live* variable rather than per
recorded node, so a program whose intermediates go out of scope costs `O(n × live width)` and
never materializes a tape.

Build one with [`forward_cover`](@ref) rather than directly.
"""
struct FVar{T, R} <: CoverVar{T}
    channels::Union{ForwardChannels{T, R}, Nothing}
    val::T

    # `R` reaches the fields only through the channels, so a constant -- which has none -- leaves
    # the generated constructor unable to infer it. Both parameters are named at every call site
    # instead, and the field declarations coerce.
    FVar{T, R}(channels, val) where {T, R} = new{T, R}(channels, val)
end

isconstant(x::FVar) = x.channels === nothing
value(x::FVar) = x.val
constantvar(::Type{FVar{T, R}}, c::Real) where {T, R} = FVar{T, R}(nothing, c)

"""
    channels(x::FVar) -> ForwardChannels

The tangent and cover channels behind `x`. A constant has neither: every elemental folds a
constant operand into one that propagates, so reaching here with a constant means a caller
bypassed that folding.
"""
function channels(x::FVar)
    ch = x.channels
    ch === nothing &&
        throw(ArgumentError("a constant FVar (value $(value(x))) carries no tangent or cover channel; it must be folded into an operand that does"))
    return ch
end

"""
    ElementalSite(val, operands)

The [`CoverSite`](@ref) of an elemental in a forward propagation. Without a tape to number the
elemental, only its value and operand values identify it, so those are what the site reports.

`operands` is held as given and read only when an error is raised, so naming a site costs nothing
on the path that succeeds.
"""
struct ElementalSite{T, O} <: CoverSite
    val::T
    operands::O
end

Base.eltype(::Type{<:ElementalSite{T}}) where {T} = T
site_label(::ElementalSite) = "this elemental"
site_values(s::ElementalSite) = "value $(s.val), operand values $(Tuple(s.operands))"

"""
    forward_scale(rule::SplitRule, x::FVar)

The scale reference that `rule` reads from `x` when splitting a local cover—the forward analogue of
[`split_scales`](@ref), which reads a whole tape at once.

A rule is usable here only if a forward propagation carries what it references.
[`ValueSplit`](@ref) needs the value and [`VectorNormSplit`](@ref) the tangent, both of which are
channels of an [`FVar`](@ref). [`SensitivitySplit`](@ref) needs the absolute sensitivities, which
are a third channel this propagation does not carry, so it is rejected rather than approximated by
something else.
"""
forward_scale(::ValueSplit, x::FVar) = abs(value(x))
forward_scale(::VectorNormSplit{Nothing}, x::FVar) = norm(channels(x).g)

function forward_scale(rule::VectorNormSplit, x::FVar)
    g = channels(x).g
    w = rule.w
    Base.require_one_based_indexing(w)
    length(w) == length(g) ||
        throw(DimensionMismatch("VectorNormSplit weights have length $(length(w)), expected $(length(g)) (one per input)"))
    return sqrt(sum(w[a] * g[a]^2 for a in eachindex(g)))
end

forward_scale(::SensitivitySplit, ::FVar) = throw(
    ArgumentError(
        "SensitivitySplit references the absolute sensitivities, which a forward propagation " *
            "does not carry: it propagates the signed tangents and the cover, not the " *
            "absolute-value twin of the tangents. Use ValueSplit or VectorNormSplit"
    )
)

forward_scale(rule::SplitRule, ::FVar) = throw(
    ArgumentError(
        "$(nameof(typeof(rule))) supplies its scale references through `split_scales`, which " *
            "reads a tape; a forward propagation has none. A rule usable here defines " *
            "`forward_scale`"
    )
)

# The cover recursion of one elemental, in the squared channel:
#
#     G_k = sum_i abs(d_i) * G_i + sum over atoms of (sum_{i in atom} rho_i * abs.(g_i))^2
#
# The first term is folded in by the callers, which have the partials; this adds the second. The
# atoms are assembled exactly as `merge_atoms` assembles them on a tape, with `av = 1` in place of
# the adjoint, which is the one factor a forward pass cannot know.
function add_atoms!(G, C::Curvature, ctx::ForwardContext, site, sl, gs)
    T = eltype(G)
    rho = cover_slots(C, ctx.rule, site, one(T), sl; check = ctx.check)
    rho === nothing && return G
    s = ctx.scratch
    for atom in atom_slots(C, length(gs))
        fill!(s, 0)
        for i in atom
            r = rho[i]
            r == 0 && continue
            gi = gs[i]
            for q in eachindex(s, gi)
                s[q] += r * abs(gi[q])
            end
        end
        for q in eachindex(G, s)
            G[q] += s[q]^2
        end
    end
    return G
end

function unarynode(x::FVar{T, R}, val, d, c) where {T, R}
    ch = channels(x)
    ctx = ch.ctx
    dT = convert(T, d)
    g = dT .* ch.g
    G = abs(dT) .* ch.G
    if !iszero(c)
        site = ElementalSite(convert(T, val), (value(x),))
        add_atoms!(G, UnaryCurvature(convert(T, c)), ctx, site, (), (ch.g,))
    end
    return FVar{T, R}(ForwardChannels(ctx, g, G), val)
end

function binarynode(x::FVar{T, R}, y::FVar{T, R}, val, d1, d2, c11, c12, c22) where {T, R}
    cx, cy = channels(x), channels(y)
    ctx = cx.ctx
    ctx === cy.ctx || throw(ArgumentError("cannot combine FVars from different propagations"))
    a, b = convert(T, d1), convert(T, d2)
    g = a .* cx.g .+ b .* cy.g
    G = abs(a) .* cx.G .+ abs(b) .* cy.G
    if !(iszero(c11) && iszero(c12) && iszero(c22))
        C = BinaryCurvature(convert(T, c11), convert(T, c12), convert(T, c22))
        site = ElementalSite(convert(T, val), (value(x), value(y)))
        sl = (forward_scale(ctx.rule, x), forward_scale(ctx.rule, y))
        add_atoms!(G, C, ctx, site, sl, (cx.g, cy.g))
    end
    return FVar{T, R}(ForwardChannels(ctx, g, G), val)
end

function linearnode(coeffs::AbstractVector{<:Real}, xs::AbstractVector{<:FVar{T, R}}) where {T, R}
    isempty(xs) && throw(ArgumentError("cannot propagate an empty linear combination"))
    idxs = eachindex(coeffs, xs)
    ctx = channels(xs[first(idxs)]).ctx
    val = zero(T)
    g = zeros(T, ctx.ninputs)
    G = zeros(T, ctx.ninputs)
    for j in idxs
        ch = channels(xs[j])
        ch.ctx === ctx || throw(ArgumentError("cannot combine FVars from different propagations"))
        cj = convert(T, coeffs[j])
        val += cj * value(xs[j])
        acj = abs(cj)
        for q in eachindex(g, ch.g)
            g[q] += cj * ch.g[q]
            G[q] += acj * ch.G[q]
        end
    end
    return FVar{T, R}(ForwardChannels(ctx, g, G), val)
end

function dotnode(ws::AbstractVector{<:FVar{T, R}}, xs::AbstractVector{<:FVar{T, R}}) where {T, R}
    isempty(ws) && throw(ArgumentError("cannot propagate an empty inner product"))
    idxs = eachindex(ws, xs)
    q = length(idxs)
    ctx = channels(ws[first(idxs)]).ctx
    val = zero(T)
    g = zeros(T, ctx.ninputs)
    G = zeros(T, ctx.ninputs)
    # Slot order is `(w_1..w_q, x_1..x_q)`, matching `PairedCurvature`'s pairing of `j` with `q+j`.
    gs = Vector{Vector{T}}(undef, 2q)
    ops = Vector{T}(undef, 2q)
    for (n, j) in enumerate(idxs)
        cw, cx = channels(ws[j]), channels(xs[j])
        (cw.ctx === ctx && cx.ctx === ctx) ||
            throw(ArgumentError("cannot combine FVars from different propagations"))
        vw, vx = value(ws[j]), value(xs[j])
        val += vw * vx
        # The partial with respect to one side of a pair is the value of the other.
        for r in eachindex(g, cw.g, cx.g)
            g[r] += vx * cw.g[r] + vw * cx.g[r]
            G[r] += abs(vx) * cw.G[r] + abs(vw) * cx.G[r]
        end
        gs[n], gs[q + n] = cw.g, cx.g
        ops[n], ops[q + n] = vw, vx
    end
    site = ElementalSite(val, ops)
    sl = [forward_scale(ctx.rule, v) for v in Iterators.flatten((ws, xs))]
    add_atoms!(G, PairedCurvature(q), ctx, site, sl, gs)
    return FVar{T, R}(ForwardChannels(ctx, g, G), val)
end

# A declared sub-map is not among the elementals a propagation handles: `declare_submap` builds the
# declaration by sweeping a tape of the sub-map, and a propagation has no tape to sweep. Naming
# that beats the bare MethodError the missing recorder would otherwise raise.
submapnode(
    ::AbstractVector{<:FVar}, ::AbstractVector{<:Real}, ::AbstractMatrix{<:Real},
    ::Union{Nothing, AbstractMatrix{<:Real}} = nothing
) = throw(
    ArgumentError(
        "a declared sub-map is available while recording, not while propagating a cover " *
            "forward: `declare_submap` builds the declaration by sweeping a tape of the sub-map. " *
            "Call the sub-map's own operations instead, which the propagation follows directly"
    )
)

"""
    forward_cover(f, x0::AbstractVector; rule=VectorNormSplit(), check=true) -> ForwardCover
    forward_cover(x::FVar) -> ForwardCover

Compute a certified rank-1 cover of a scalar function's Hessian by propagating it forward, in one
pass and with no tape.

`f` must accept a vector of [`FVar`](@ref)s and return one, using operations `FVar` supports. Per
variable the propagation carries the value, the signed tangent `∇v`, and a squared cover of `∇²v`;
the recursions are

    ∇v_k = sum_i d_i * ∇v_i
    G_k  = sum_i abs(d_i) * G_i + sum over atoms of (sum_{i in atom} rho_i * abs.(∇v_i))^2

with `rho` the same local cover a reverse sweep builds, except that the adjoint factor is `1`: a
forward pass cannot know how the output depends on the node it is at. The cover is `sqrt.(G)`.

The second form returns the cover of any variable reached along the way, the propagation having
maintained the invariant at every one of them, not only at the output.

# Computational cost and tightness

`O(n*L)` time for `n` inputs and `L` operations, and `O(n)` memory per *live* variable — no tape,
and nothing proportional to the length of the program. Its main advantage is therefore the ability
to process programs that are too long to record. Currently, the tape-based
[`cover_hessian`](@ref) with [`SignedTangents`](@ref) is tighter, since it merges
the same atoms weighted by exact adjoints rather than by the unsigned path sums a forward pass has
to use.

The cover this reaches is exactly the signed-tangent, absolute-adjoint corner of
[`cover_lattice`](@ref) — `h_quad_forward` — which is therefore what it is tested against. That
corner is *incomparable* with the default `h_quad`: it keeps every tangent sign and discards every
adjoint sign, so which is tighter is a property of the program.

`rule` selects the [`SplitRule`](@ref), and only rules that reference something a forward pass
carries are usable; see [`forward_scale`](@ref). The default [`VectorNormSplit`](@ref) is the one
whose scale reference — the tangent norm — the propagation already has, and it is the per-node
minimizer of the quadrature merge this construction performs. `check=true` verifies every local
cover.

That default differs from the tape entry points, which default to [`ValueSplit`](@ref) because a
tape cannot afford a tangent norm. Two covers are comparable only under the same rule, so pass
`rule` on both sides when comparing this against [`cover_hessian`](@ref) or
[`cover_lattice`](@ref).

# Examples

```jldoctest
julia> f(x) = exp(x[1]) + x[1] * x[2]^2;

julia> c = forward_cover(f, [0.5, 2.0]);

julia> round.(c.grad; digits=4)
2-element Vector{Float64}:
 5.6487
 2.0

julia> H = [exp(0.5) 4.0; 4.0 1.0];   # the Hessian of f at [0.5, 2.0]

julia> all(abs.(H) .<= c.h_quad * c.h_quad')
true
```
"""
function forward_cover(
        f, x0::AbstractVector; rule::SplitRule = VectorNormSplit(), check::Bool = true
    )
    Base.require_one_based_indexing(x0)
    T = float(eltype(x0))
    R = typeof(rule)
    n = length(x0)
    n == 0 && throw(ArgumentError("cannot propagate a cover through a program with no inputs"))
    ctx = ForwardContext{T, R}(rule, n, check, zeros(T, n))
    xs = [
        FVar{T, R}(
                ForwardChannels(ctx, T[i == a for i in 1:n], zeros(T, n)), convert(T, x0[a])
            ) for a in 1:n
    ]
    out = f(xs)
    out isa FVar{T, R} ||
        throw(ArgumentError("program output is not an FVar of this propagation (got a $(typeof(out))); forward_cover covers a scalar-valued program"))
    isconstant(out) &&
        throw(ArgumentError("program output is a constant (value $(value(out))); it does not depend on the inputs"))
    channels(out).ctx === ctx ||
        throw(ArgumentError("program output came from a different propagation than its inputs"))
    return forward_cover(out)
end

function forward_cover(x::FVar)
    ch = channels(x)
    return ForwardCover(copy(ch.g), sqrt.(ch.G))
end
