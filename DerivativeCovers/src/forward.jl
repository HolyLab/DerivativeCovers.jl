# A Hessian cover propagated forward, in one pass and with no tape.

"""
    ForwardCover{T, M, VG, VH}

The result of [`forward_cover`](@ref), with `T` as the propagation's element type,
`M =` [`magtype`](@ref)`(T)` the type of the cover, and `VG`, `VH` the channel storage the
propagation ran in (see [`ForwardContext`](@ref)). `ForwardCover{T}` stands for
`ForwardCover{T, magtype(T)}`.

- `grad::VG`: the exact gradient of the output, in `T`.
- `h_quad::VH`: a certified Hessian cover formed by merging the cover atoms in quadrature, in `M`.
"""
struct ForwardCover{T, M, VG <: AbstractVector{T}, VH <: AbstractVector{M}}
    grad::VG
    h_quad::VH

    ForwardCover{T, M, VG, VH}(grad, h_quad) where {T, M, VG, VH} =
        new{T, M, VG, VH}(grad, h_quad)
end

ForwardCover(grad::AbstractVector{T}, h_quad) where {T} = ForwardCover{T}(grad, h_quad)

ForwardCover{T}(grad, h_quad) where {T} = ForwardCover{T, magtype(T)}(grad, h_quad)

function ForwardCover{T, M}(grad, h_quad) where {T, M}
    g = convert(AbstractArray{T}, grad)::AbstractArray{T}
    h = convert(AbstractArray{M}, h_quad)::AbstractArray{M}
    return ForwardCover{T, M, typeof(g), typeof(h)}(g, h)
end

"""
    ForwardContext{T, S, M, SM}

State shared by every variable in a forward propagation: the degeneracy threshold local covers
split at, whether each local cover is verified, and the scratch a node writes through while
assembling its cover.

`S` is the storage of the intermediate-gradient channel, one entry of `T` per input, and `SM` that
of the cover channel, one entry of `M =` [`boundtype`](@ref)`(T)` per input. [`forward_cover`](@ref)
derives both from the point it is given, so a `Vector` input propagates in `Vector`s and a
`StaticVector` input in `StaticVector`s. `zerogradient` and `zerochannel` are zeros of the two
storages, which fresh channels are copied from; the input variables share `zerochannel` as their
cover channel. `scratch` is the atom accumulator, written in place when `SM` is mutable and left
alone when it is not; and `rho` holds the slot weights of the elemental being covered.

It is a reference type so that every variable of the propagation holds a pointer to the one
context: storing an immutable into a variable's field would copy it, at one allocation per node.
Identity is then also what `===` compares, which is what distinguishes two propagations.
"""
mutable struct ForwardContext{T, S <: AbstractVector{T}, M, SM <: AbstractVector{M}}
    const ratio::M
    const check::Bool
    const zerogradient::S
    const zerochannel::SM
    const scratch::SM
    const rho::Vector{M}

    ForwardContext{T, S, M, SM}(ratio, check, zerogradient, zerochannel, scratch, rho) where {T, S, M, SM} =
        new{T, S, M, SM}(ratio, check, zerogradient, zerochannel, scratch, rho)
end

function ForwardContext{T}(ratio, check, zerogradient::AbstractVector{T}, zerochannel::AbstractVector) where {T}
    M = boundtype(T)
    return ForwardContext{T, typeof(zerogradient), M, typeof(zerochannel)}(
        ratio, check, zerogradient, zerochannel, zero(zerochannel), M[]
    )
end

"""
    ForwardChannels{T, S, M, SM}

The two vector channels an [`FVar`](@ref) carries beyond its value and its declared scale: the
intermediate gradient `g = ∇v` (storage `S`, entries `T`), and the squared cover `G` (storage `SM`,
entries `M =` [`boundtype`](@ref)`(T)`) maintaining `sqrt.(G) * sqrt.(G)' >= magnitude.(∇²v)`
entrywise.

`G` is squared rather than rooted because the recursion that maintains it is *linear* in the
parents' `G`, which is what keeps a per-node re-merge from compounding with depth.

The channels live in the [`FVar`](@ref)'s own fields; [`channels`](@ref) assembles this view of
them on demand, and it costs nothing once inlined.
"""
struct ForwardChannels{T, S, M, SM}
    ctx::ForwardContext{T, S, M, SM}
    g::S
    G::SM

    ForwardChannels{T, S, M, SM}(ctx, g, G) where {T, S, M, SM} = new{T, S, M, SM}(ctx, g, G)
end

"""
    FVar{T, S, M, SM}

A scalar of type `T` that propagates a Hessian cover forward. This is the tape-free
[`CoverVar`](@ref) backend.

`S` and `SM` are the storage of the intermediate-gradient and cover channels, and `M =`
[`boundtype`](@ref)`(T)` the type the cover and the declared scale are propagated in (see
[`ForwardContext`](@ref)). `M` and `SM` follow from `T` and `S`, so `FVar{T, S}` names the variable
type as well. A constant carries the storage too, because generic numerical code builds typed
literals from `eltype`, which is all it has to reach the backend by.

For each variable, the propagation holds the value, the intermediate gradient `∇v`, a squared cover
of `∇²v` (see [`ForwardChannels`](@ref)), the scalar declared scale `sigma` the split falls back on,
and whether `∇v` is identically zero (see [`ForwardScale`](@ref)). Working memory is `O(n)` per
*live* variable rather than per recorded node, so a program whose intermediates go out of scope
costs `O(n × live width)` and never materializes a tape.

Build one with [`forward_cover`](@ref) rather than directly.
"""
struct FVar{T, S, M, SM} <: CoverVar{T}
    val::T
    sigma::M
    live::Bool
    ctx::Union{ForwardContext{T, S, M, SM}, Nothing}
    g::S
    G::SM

    # The channels are held field by field rather than as a nested struct: a struct is immutable
    # and so would be copied into the field, at one allocation per node. Their type must stay
    # concrete for the same reason -- an immutable `S` is then stored inline, where a
    # `Union{S,Nothing}` would be copied to the heap at every node -- so a constant, which has no
    # channels, gets `constantchannel` in the fields and is recognized by a null context instead.
    FVar{T, S, M, SM}(val, sigma, live, ctx, g, G) where {T, S, M, SM} =
        new{T, S, M, SM}(val, sigma, live, ctx, g, G)
end

# A constant. Its storage is named rather than defaulted: a constant of the wrong storage combines
# with nothing in the propagation that produced it.
function FVar{T, S}(val) where {T, S}
    M = boundtype(T)
    return FVar{T, S, M, channeltype(S, M)}(val)
end

FVar{T, S, M, SM}(val) where {T, S, M, SM} =
    FVar{T, S, M, SM}(val, zero(M), false, nothing, constantchannel(S), constantchannel(SM))

isconstant(x::FVar) = x.ctx === nothing
value(x::FVar) = x.val
constantvar(::Type{FVar{T, S, M, SM}}, c::Real) where {T, S, M, SM} = FVar{T, S, M, SM}(c)

@noinline _constant_channels(x::FVar) = throw(
    ArgumentError(
        "a constant FVar (value $(value(x))) carries no gradient or cover channel; it must be " *
            "folded into an operand that does"
    )
)

"""
    channels(x::FVar) -> ForwardChannels

The intermediate-gradient and cover channels behind `x`. A constant has neither: every elemental
folds a constant operand into one that propagates, so reaching here with a constant means a caller
bypassed that folding.
"""
function channels(x::FVar{T, S, M, SM}) where {T, S, M, SM}
    ctx = x.ctx
    ctx === nothing && _constant_channels(x)
    return ForwardChannels{T, S, M, SM}(ctx, x.g, x.G)
end

# --- channel storage ---------------------------------------------------------------------------
# The storage a propagation runs in is whatever the point it is given is made of, so that a caller
# asking for static arrays gets a propagation that allocates nothing per node. Two shapes of
# accumulation follow from that: writable storage is filled in place, and immutable storage is
# rebuilt at every step. Which one applies is a property of the type, so the branch below is
# resolved at compile time and only one of the two ever appears in the generated code.

struct MutableChannels end
struct ImmutableChannels end

channelkind(::Type{S}) where {S} = Base.ismutabletype(S) ? MutableChannels() : ImmutableChannels()
channelkind(g::AbstractVector) = channelkind(typeof(g))

# A zero channel matching the point `x0`, and the intermediate gradient of its `a`th coordinate.
channel_zero(x0::AbstractVector, ::Type{T}) where {T} = zeros(T, length(x0))
channel_zero(x0::StaticVector, ::Type{T}) where {T} = zeros(similar_type(x0, T))

function channel_basis(x0::AbstractVector, ::Type{T}, a::Integer) where {T}
    g = channel_zero(x0, T)
    g[a] = one(T)
    return g
end

channel_basis(x0::StaticVector{N}, ::Type{T}, a::Integer) where {N, T} =
    similar_type(x0, T)(ntuple(i -> T(i == a), Val(N)))

# The storage `channel_zero` gives entries of type `M` when it gives `S` for the point's own.
channeltype(::Type{<:Vector}, ::Type{M}) where {M} = Vector{M}
channeltype(::Type{S}, ::Type{M}) where {S <: StaticVector, M} = similar_type(S, M)

# The channels of a constant, which has none. Nothing reads them; they exist so that the fields
# they fill can stay concretely typed.
constantchannel(::Type{S}) where {S} = zero(S)
constantchannel(::Type{S}) where {S <: Array} = S(undef, 0)

# Channel accumulation. Each returns the accumulator, which the caller must use: only writable
# storage is updated in place, and immutable storage comes back as a new value.
#
# Nodes read but never write their parents' channels, allowing input variables to share
# `zerochannel`. Accumulation targets are fresh channels, broadcast results, or `scratch`.
# `add_scaled_abs!` reads an intermediate gradient into a cover channel through `magnitude`.
add_scaled!(acc, c, v) = add_scaled!(channelkind(acc), acc, c, v)
add_scaled!(::ImmutableChannels, acc, c, v) = acc .+ c .* v
function add_scaled!(::MutableChannels, acc, c, v)
    for q in eachindex(acc, v)
        acc[q] += c * v[q]
    end
    return acc
end

add_scaled_abs!(acc, c, v) = add_scaled_abs!(channelkind(acc), acc, c, v)
add_scaled_abs!(::ImmutableChannels, acc, c, v) = acc .+ c .* magnitude.(v)
function add_scaled_abs!(::MutableChannels, acc, c, v)
    for q in eachindex(acc, v)
        acc[q] += c * magnitude(v[q])
    end
    return acc
end

add_squares!(acc, v) = add_squares!(channelkind(acc), acc, v)
add_squares!(::ImmutableChannels, acc, v) = acc .+ v .^ 2
function add_squares!(::MutableChannels, acc, v)
    for q in eachindex(acc, v)
        acc[q] += v[q]^2
    end
    return acc
end

# The variables a propagation starts from. The container is a `Vector` whatever the point is: a
# program reaches its inputs at indices it computes, and a static container answers such an index
# only by spilling the whole thing to the stack. The channels inside are static or not according
# to the point, which is where the cost of the propagation actually lies.
function input_vars(::Type{FVar{T, S, M, SM}}, ctx, x0, w) where {T, S, M, SM}
    sigma(a) = w === nothing ? one(M) : convert(M, w[a])
    # Parent channels are read-only, so inputs can share `ctx.zerochannel`.
    var(a) = FVar{T, S, M, SM}(x0[a], sigma(a), true, ctx, channel_basis(x0, T, a), ctx.zerochannel)
    # `Base.OneTo`, not `eachindex(x0)`: a static point indexes over `SOneTo`, and a comprehension
    # over that rebuilds the static container this is here to avoid. The caller has already
    # declared one-based indexing.
    return [var(a) for a in Base.OneTo(length(x0))]
end

# A zero channel to accumulate into: a fresh intermediate-gradient or cover channel, and the
# context's reusable cover channel.
freshgradient(ctx::ForwardContext) = zero(ctx.zerogradient)
freshchannel(ctx::ForwardContext) = zero(ctx.zerochannel)
zeroscratch(ctx::ForwardContext) = zeroscratch(channelkind(ctx.scratch), ctx)
zeroscratch(::MutableChannels, ctx::ForwardContext) = fill!(ctx.scratch, 0)
zeroscratch(::ImmutableChannels, ctx::ForwardContext) = ctx.scratch

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
    ForwardScale(sigma, live)

The declared channel of one slot of a forward propagation: the scale `sigma` that stands in for a
parent value carrying none, and whether the slot's intermediate gradient is identically zero.

Liveness travels separately from `sigma` because `sigma` cannot report it. It is a sum of products
of first partials, so it underflows on a long chain of small factors while the intermediate gradient
it bounds is still there, and a slot dropped on that evidence would leave a real Hessian entry
uncovered. The flag is carried by the same recursion [`live_nodes`](@ref) runs over a tape — a slot
is live when some parent is live and reaches it through a nonzero partial — so both backends drop
exactly the same entries.
"""
struct ForwardScale{T}
    sigma::T
    live::Bool
end

declared_scale(f::ForwardScale) = f.sigma
dead_slot(f::ForwardScale) = !f.live

"""
    forward_scale(x::FVar) -> SplitScale

The scale references [`split_block`](@ref) reads at `x` — the forward analogue of
[`split_scales`](@ref), which reads a whole tape at once.

Both channels are already on the variable: the value, and the [`ForwardScale`](@ref) behind it. So
a propagation splits its local covers by exactly the rule a tape does, and the two backends are
comparable coordinate by coordinate.

Nothing is deferred here, unlike on a tape. `sigma` obeys the same recursion as the tape's
[`sensitivity_scales`](@ref) sweep, `sigma_k = sum_i magnitude(d_i) * sigma_{p_i}`, but it is a
single scalar per variable rather than a pass over recorded nodes — so carrying it costs `O(1)` per
elemental whether or not any block asks for it, against the `O(L)` sweep a tape would have to run.
"""
forward_scale(x::FVar) = SplitScale(magnitude(value(x)), ForwardScale(x.sigma, x.live))

# The cover recursion of one elemental, in the squared channel:
#
#     G_k = sum_i magnitude(d_i) * G_i + sum over atoms of (sum_{i in atom} rho_i * magnitude.(g_i))^2
#
# The first term is folded in by the callers, which have the partials; this adds the second. The
# atoms are assembled exactly as `merge_atoms` assembles them on a tape, with `av = 1` in place of
# the adjoint, which is the one factor a forward pass cannot know. `G` comes back rather than
# being filled: immutable storage cannot be filled.
function add_atoms(G, C::Curvature, ctx::ForwardContext, site, sl, gs)
    rho = ctx.rho
    cover_slots!(rho, C, ctx.ratio, site, one(eltype(G)), sl; check = ctx.check) || return G
    for atom in atom_slots(C, length(gs))
        s = zeroscratch(ctx)
        for i in atom
            r = rho[i]
            r == 0 && continue
            s = add_scaled_abs!(s, r, gs[i])
        end
        G = add_squares!(G, s)
    end
    return G
end

function unarynode(x::FVar{T, S, M, SM}, val, d, c) where {T, S, M, SM}
    ch = channels(x)
    ctx = ch.ctx
    dT = convert(T, d)
    site = ElementalSite(convert(T, val), (value(x),))
    check_values(site, site.val, (dT,), (c,))
    ad = magnitude(dT)
    g = dT .* ch.g
    G = ad .* ch.G
    sigma = ad * x.sigma
    if !iszero(magnitude(c))
        G = add_atoms(G, UnaryCurvature(convert(T, c)), ctx, site, (), (ch.g,))
    end
    return FVar{T, S, M, SM}(val, sigma, x.live && !iszero(ad), ctx, g, G)
end

function binarynode(
        x::FVar{T, S, M, SM}, y::FVar{T, S, M, SM}, val, d1, d2, c11, c12, c22
    ) where {T, S, M, SM}
    cx, cy = channels(x), channels(y)
    ctx = cx.ctx
    ctx === cy.ctx || throw(ArgumentError("cannot combine FVars from different propagations"))
    a, b = convert(T, d1), convert(T, d2)
    site = ElementalSite(convert(T, val), (value(x), value(y)))
    check_values(site, site.val, (a, b), (c11, c12, c22))
    aa, ab = magnitude(a), magnitude(b)
    g = a .* cx.g .+ b .* cy.g
    G = aa .* cx.G .+ ab .* cy.G
    sigma = aa * x.sigma + ab * y.sigma
    if !(iszero(magnitude(c11)) && iszero(magnitude(c12)) && iszero(magnitude(c22)))
        C = BinaryCurvature(convert(T, c11), convert(T, c12), convert(T, c22))
        sl = (forward_scale(x), forward_scale(y))
        G = add_atoms(G, C, ctx, site, sl, (cx.g, cy.g))
    end
    live = (x.live && !iszero(aa)) || (y.live && !iszero(ab))
    return FVar{T, S, M, SM}(val, sigma, live, ctx, g, G)
end

function linearnode(
        coeffs::AbstractVector{<:Real}, xs::AbstractVector{<:FVar{T, S, M, SM}}
    ) where {T, S, M, SM}
    isempty(xs) && throw(ArgumentError("cannot propagate an empty linear combination"))
    idxs = eachindex(coeffs, xs)
    ctx = channels(xs[first(idxs)]).ctx
    val = zero(T)
    sigma = zero(M)
    live = false
    g = freshgradient(ctx)
    G = freshchannel(ctx)
    for j in idxs
        ch = channels(xs[j])
        ch.ctx === ctx || throw(ArgumentError("cannot combine FVars from different propagations"))
        cj = convert(T, coeffs[j])
        acj = magnitude(cj)
        val += cj * value(xs[j])
        sigma += acj * xs[j].sigma
        live |= xs[j].live && !iszero(acj)
        g = add_scaled!(g, cj, ch.g)
        G = add_scaled!(G, acj, ch.G)
    end
    return FVar{T, S, M, SM}(val, sigma, live, ctx, g, G)
end

function dotnode(
        ws::AbstractVector{<:FVar{T, S, M, SM}}, xs::AbstractVector{<:FVar{T, S, M, SM}}
    ) where {T, S, M, SM}
    isempty(ws) && throw(ArgumentError("cannot propagate an empty inner product"))
    idxs = eachindex(ws, xs)
    q = length(idxs)
    ctx = channels(ws[first(idxs)]).ctx
    val = zero(T)
    sigma = zero(M)
    live = false
    g = freshgradient(ctx)
    G = freshchannel(ctx)
    # Slot order is `(w_1..w_q, x_1..x_q)`, matching `PairedCurvature`'s pairing of `j` with `q+j`.
    gs = Vector{S}(undef, 2q)
    ops = Vector{T}(undef, 2q)
    for (n, j) in enumerate(idxs)
        cw, cx = channels(ws[j]), channels(xs[j])
        (cw.ctx === ctx && cx.ctx === ctx) ||
            throw(ArgumentError("cannot combine FVars from different propagations"))
        vw, vx = value(ws[j]), value(xs[j])
        val += vw * vx
        # The partial with respect to one side of a pair is the value of the other.
        avw, avx = magnitude(vw), magnitude(vx)
        sigma += avx * ws[j].sigma + avw * xs[j].sigma
        live |= (ws[j].live && !iszero(avx)) || (xs[j].live && !iszero(avw))
        g = add_scaled!(add_scaled!(g, vx, cw.g), vw, cx.g)
        G = add_scaled!(add_scaled!(G, avx, cw.G), avw, cx.G)
        gs[n], gs[q + n] = cw.g, cx.g
        ops[n], ops[q + n] = vw, vx
    end
    site = ElementalSite(val, ops)
    sl = [forward_scale(v) for v in Iterators.flatten((ws, xs))]
    G = add_atoms(G, PairedCurvature(q), ctx, site, sl, gs)
    return FVar{T, S, M, SM}(val, sigma, live, ctx, g, G)
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
    forward_cover(f, x0::AbstractVector; scales=nothing, ratio=nothing, check=true) -> ForwardCover
    forward_cover(x::FVar) -> ForwardCover

Compute a certified rank-1 cover of a scalar function's Hessian by propagating it forward, in one
pass and with no tape.

`f` must accept a vector of [`FVar`](@ref)s and return one, using operations `FVar` supports. Per
variable the propagation carries the value, the intermediate gradient `∇v`, a squared cover of
`∇²v`, and the scalar declared scale `sigma`; the recursions are

    ∇v_k    = sum_i d_i * ∇v_i
    sigma_k = sum_i magnitude(d_i) * sigma_i
    G_k     = sum_i magnitude(d_i) * G_i + sum over atoms of (sum_{i in atom} rho_i * magnitude.(∇v_i))^2

with `rho` the same local cover a reverse sweep builds, except that the adjoint factor is `1`: a
forward pass cannot know how the output depends on the node it is at. The cover is `sqrt.(G)`.

The gradient is in the element type `T = float(eltype(x0))` and the cover in
[`magtype`](@ref)`(T)`. As for [`cover_hessian`](@ref), `x0` may be a box of intervals: the
gradient then encloses the gradient over the box and the cover holds at every point of it, the
propagation rounding its bounds upward (see [`boundtype`](@ref)). An
elemental evaluated outside its domain somewhere on the box is an error (see
[`check_values`](@ref)).

The second form returns the cover of any variable reached along the way, the propagation having
maintained the invariant at every one of them, not only at the output.

# Computational cost and tightness

`O(n*L)` time for `n` inputs and `L` operations, and `O(n)` memory per *live* variable — no tape,
and nothing proportional to the length of the program. Its main advantage is therefore the ability
to process programs that are too long to record.

The channels are stored in whatever `x0` is made of, so a `StaticVector` point propagates in
static arrays and no node allocates at all. That is worth doing for a handful of inputs and not
for many: the propagation is then compiled for the exact input count, which for large `n` costs
more in compilation than it saves at run time. A `Vector` point propagates in `Vector`s and
compiles once for every size.

The cover keeps the signs of the intermediate gradients `∇v`, which a forward pass carries
exactly, and discards the signs of the adjoints, which it cannot see: each atom is weighted by the
unsigned sum over paths from its node to the output rather than by the node's adjoint. That makes
it *incomparable* with the quadrature merge `h_quad` of [`HessianDiagnostics`](@ref), which
discards the signs of the intermediate gradients and keeps those of the adjoints; which is tighter
is a property of the program.

`scales` and `ratio` govern the split exactly as in [`cover_hessian`](@ref): the propagation
carries both of the references [`split_block`](@ref) reads, so it splits by the same rule a tape
does and its cover is comparable with the tape's coordinate by coordinate. See
[`forward_scale`](@ref). `check=true` verifies every local cover.

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
        f, x0::AbstractVector; scales = nothing, ratio = nothing, check::Bool = true
    )
    Base.require_one_based_indexing(x0)
    T = float(eltype(x0))
    length(x0) == 0 &&
        throw(ArgumentError("cannot propagate a cover through a program with no inputs"))
    w = check_input_scales(scales, length(x0))
    M = boundtype(T)
    z = channel_zero(x0, T)
    zm = channel_zero(x0, M)
    S, SM = typeof(z), typeof(zm)
    ctx = ForwardContext{T}(split_ratio(ratio, T), check, z, zm)
    xs = input_vars(FVar{T, S, M, SM}, ctx, x0, w)
    out = f(xs)
    out isa FVar{T, S, M, SM} ||
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
