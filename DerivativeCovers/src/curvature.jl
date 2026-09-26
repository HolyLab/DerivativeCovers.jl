# The second-partial forms a tape node can carry, and the operations every cover sweep performs
# on them.

"""
    Curvature

The second partials `∂²φ_k/∂v_{p_i}∂v_{p_j}` of one tape node, in whatever form that node's
elemental can supply them. [`Tape`](@ref) stores one per node.

Cover constructions never need the entries themselves, only bounds on their magnitudes, so a
curvature may declare a bound instead of recording a matrix ([`DeclaredCurvature`](@ref)). The
sweeps reach every form through the methods below, which is what lets a new elemental introduce
a new form without touching them.

A new subtype must define:

  - [`cover_slots`](@ref): the node's rank-1 cover, slot-aligned with its parents. It reads its
    scale references and its error context through a [`CoverSite`](@ref) rather than off a tape,
    so one definition serves every backend.
  - [`bound_contribution!`](@ref): the node's term of the elementwise bound `B`.
  - [`atom_slots`](@ref): the parent-slot groups the quadrature merge combines separately. The
    fallback treats every slot as one atom, which is correct for any curvature whose cover is a
    single rank-1 term.

A subtype whose entries are recorded exactly may also define:

  - [`curvature_apply!`](@ref): the second-order term of a Hessian-vector product.
  - [`has_exact_partials`](@ref): `false` suppresses [`hvp`](@ref) on any tape containing the
    node. The fallback is `true`, so a subtype that declares a bound must override it.

`getindex` and `size` are introspection, not part of the contract: a curvature that records no
entries has none to report.
"""
abstract type Curvature end

"""
    CoverSite

Where a local cover is being built, carried by [`cover_slots`](@ref) for error reporting.

The split needs the element type to work in and enough context to say which elemental it failed
on, and nothing else about the backend. Naming that little through a site is what lets one split
serve a tape sweep and a tapeless forward pass alike.

A subtype defines `Base.eltype`, [`site_label`](@ref) and [`site_values`](@ref).
"""
abstract type CoverSite end

"""
    site_label(site::CoverSite) -> AbstractString

How to name the elemental in an error message, such as `"node 7"`.
"""
function site_label end

"""
    site_values(site::CoverSite) -> AbstractString

The values surrounding the elemental, for the error that reports where a program stopped being
twice-differentiable: `"node value 1.5, parent values (0.3, 2.0)"` and the like.
"""
function site_values end

"""
    NoCurvature() <: Curvature

The curvature of a node with no second-order term: an input node, a linear n-ary node such as
[`linearnode`](@ref), or an elemental whose curvature at the recorded point is zero. It
contributes to no cover and to no bound, so the sweeps skip it by dispatch rather than by
inspecting a matrix of zeros.
"""
struct NoCurvature <: Curvature end

"""
    UnaryCurvature(c) <: Curvature

The second partial `φ''(x)` of a unary elemental, held inline as the single number it is.

A tape stores its curvatures in a container with abstract element type, so every form is reached
through a reference; holding the entry inline rather than in a `1×1` matrix saves the second
reference on the way to it, once per node per sweep, and an allocation per node while recording.
"""
struct UnaryCurvature{T} <: Curvature
    c::T
end

"""
    BinaryCurvature(c11, c12, c22) <: Curvature

The second partials `[c11 c12; c12 c22]` of a binary elemental, slot-aligned with the node's two
parents and held inline. Symmetry is structural: one entry stands for both off-diagonal slots.

Inline for the reason [`UnaryCurvature`](@ref) is.
"""
struct BinaryCurvature{T} <: Curvature
    c11::T
    c12::T
    c22::T
end

"""
    DenseCurvature(C::AbstractMatrix) <: Curvature

Recorded second partials as an explicit symmetric matrix, slot-aligned with the node's parents.

This is the general form, for a curvature with no structure to exploit and no fixed arity. The
elementals of `overloads.jl` do not use it: a unary or binary block is [`UnaryCurvature`](@ref) or
[`BinaryCurvature`](@ref), which hold their entries inline.

The rank-1 cover of a block larger than `2×2` is not a single split, so [`cover_slots`](@ref)
rejects one; the elementwise bound and the Hessian-vector product read any arity. An elemental
with more parents whose curvature *does* have structure supplies a form that carries it, as
[`PairedCurvature`](@ref) and [`DeclaredCurvature`](@ref) do.
"""
struct DenseCurvature{T, M <: AbstractMatrix{T}} <: Curvature
    C::M
end

"""
    PairedCurvature(q::Int) <: Curvature

Curvature of an inner-product node with `q` pairs of parent slots ordered
`(w_1, ..., w_q, x_1, ..., x_q)` (see [`dotnode`](@ref)). Entry `(i, j)` is `1` when `i` and `j`
are paired (`abs(i - j) == q`), the sole nonzero second partials `∂²y/∂w_j∂x_j` of
`y = sum_j w_j*x_j`, and `0` otherwise.

The `2q × 2q` matrix is never materialized: it has `2q` nonzeros, and its block-diagonal
structure lets every operation run over the `q` pairs directly.
"""
struct PairedCurvature <: Curvature
    q::Int
end

"""
    DeclaredCurvature(s::AbstractVector) <: Curvature

Declared bound on the second partials of a sub-map output (see [`submapnode`](@ref)):
`abs(C[i, j]) <= s[i]*s[j]`, with the true `C` not recorded. `s` is nonnegative and slot-aligned
with the node's parents. On a [`Tape{T}`](@ref Tape), `s` is a vector of [`magtype`](@ref)`(T)`.

Cover constructions consume second partials only through bounds on `abs.(C)`, so the declaration
substitutes for a recorded matrix everywhere except [`hvp`](@ref), which needs the signed entries
and fails fast — hence [`has_exact_partials`](@ref) is `false`. Covers built over such a node are
valid conditional on the declaration, which is the caller's assertion: nothing on the tape can
verify it.
"""
struct DeclaredCurvature{T, V <: AbstractVector{T}} <: Curvature
    s::V
end

"""
    dispatchcurvature(f, C::Curvature, ::Type{T})

Call `f(C)`, branching on the built-in [`Curvature`](@ref) forms over the element type `T` first.

[`Tape`](@ref) stores its curvatures in a container with abstract element type, so an interface
method invoked on one is a dynamic dispatch — once per node per sweep, which is a large fraction
of the sweep's cost. Testing the built-in forms by `isa` first lets the compiler resolve those
calls statically. A form defined elsewhere falls through to the same call, dispatched the
ordinary way: this is an optimization for the common cases, not a restriction on which forms are
allowed.

`T` is the element type of the tape the curvature came from, which is the one the elementals of
`overloads.jl` record in; a [`DeclaredCurvature`](@ref) holds magnitudes, so its form is resolved
over [`magtype`](@ref)`(T)`. A curvature over some other element or matrix type is correct here,
just not statically resolved.

`C` is `@nospecialize`d because specializing on it would defeat the point: a caller that knows
the concrete form would compile an instance in which every branch below but one is dead, having
already paid the dispatch this ladder exists to avoid. One instance per call site and element
type, branching at run time, is the intended shape.
"""
@inline function dispatchcurvature(f::F, @nospecialize(C::Curvature), ::Type{T}) where {F, T}
    C isa NoCurvature && return f(C)
    C isa UnaryCurvature{T} && return f(C)
    C isa BinaryCurvature{T} && return f(C)
    C isa PairedCurvature && return f(C)
    M = magtype(T)
    C isa DeclaredCurvature{M, Vector{M}} && return f(C)
    return f(C)
end

# Introspection. A curvature that records entries reports them; DeclaredCurvature has none.
Base.size(::UnaryCurvature) = (1, 1)
Base.size(::BinaryCurvature) = (2, 2)
Base.size(C::DenseCurvature) = size(C.C)
Base.size(C::PairedCurvature) = (2C.q, 2C.q)
function Base.getindex(C::UnaryCurvature, i::Int, j::Int)
    (i == 1 && j == 1) || throw(BoundsError(C, (i, j)))
    return C.c
end
function Base.getindex(C::BinaryCurvature, i::Int, j::Int)
    (1 <= i <= 2 && 1 <= j <= 2) || throw(BoundsError(C, (i, j)))
    return i == j == 1 ? C.c11 : i == j == 2 ? C.c22 : C.c12
end
Base.getindex(C::DenseCurvature, i::Int, j::Int) = C.C[i, j]
Base.getindex(C::PairedCurvature, i::Int, j::Int) = abs(i - j) == C.q ? 1.0 : 0.0

"""
    has_exact_partials(C::Curvature) -> Bool

Whether `C` records its second partials exactly, so an exact second-order product can be formed
through it. [`hvp`](@ref) refuses to run on a tape carrying any node that answers `false`.
"""
has_exact_partials(::Curvature) = true
has_exact_partials(::DeclaredCurvature) = false

"""
    atom_slots(C::Curvature, nslots) -> atoms

The groups of a node's parent slots that the quadrature merge of [`cover_hessian`](@ref)
combines separately. One atom spanning every slot is correct whenever the node's cover is a
single rank-1 term. A [`PairedCurvature`](@ref)`(q)` instead forms `q` atoms, one per pair
`(j, q+j)`, matching the block-diagonal structure of its second partials: the atoms it
contributes are independent rank-1 terms, not one combined term for the whole node. A
[`NoCurvature`](@ref) forms none.

The atoms must *partition* the slots carrying cover weight: a slot left out of every atom drops
its contribution from the quadrature merge and from [`cover_hybrid`](@ref), and a slot named twice
counts it twice.
"""
atom_slots(::Curvature, nslots) = (1:nslots,)
atom_slots(::NoCurvature, nslots) = ()
atom_slots(C::PairedCurvature, nslots) = ((j, C.q + j) for j in 1:C.q)

"""
    cover_slots(C::Curvature, ratio, site, av, sl; check=true) -> Union{Nothing,Vector{M}}
    cover_slots!(rho::Vector, C::Curvature, ratio, site, av, sl; check=true) -> Bool

The rank-1 cover of one elemental, slot-aligned with its operands: nonnegative `rho` with
`rho[i]*rho[j] >= av*magnitude(C[i,j])`, where `av` is the weight the elemental's second-order
term carries (the absolute adjoint `magnitude(vbar[k])` in a reverse sweep, `1` in a forward one).
Returns `nothing` when the elemental contributes no second-order term.

Entries involving a slot whose tangent vanishes are exempt: they are multiplied by zero in the
Hessian, so covering them would only import the other slot's scale into a term that is not there.
[`dead_slot`](@ref) is the test, and it reads the slot's entry of `sl`.

`rho`, `av` and the entries of `sl` are magnitudes, in `M = magtype(eltype(site))`. `rho` must be a
`Vector{M}`: [`local_cover_slots`](@ref) asserts that type so that the sweeps, which call it once
per node, need not box the result.

The mutating form resizes `rho` to the elemental's slot count and fills it, returning whether
there is a second-order term at all and leaving `rho` untouched when there is not. Every
construction lives there, so that a sweep can hand it one buffer and let no node allocate for its
slot weights; `cover_slots` allocates a fresh `rho` and forwards.

[`split_block`](@ref) breaks the tie at off-diagonal-active blocks; `ratio` is its degeneracy
threshold and `sl` its scale references, one per slot and in slot order — not indexed by node, so
that a backend holding no nodes can supply them. `check` verifies the covering invariant against
the recorded entries; a curvature that declares a bound rather than recording entries has nothing
to check against.

`site` is a [`CoverSite`](@ref), used for the element type and for error reporting. This is the
per-elemental kernel shared by [`local_covers`](@ref), the [`CoverOnly`](@ref) sweep, and the
forward propagation; see [`local_cover_slots`](@ref).
"""
cover_slots(::NoCurvature, ratio, site, av, sl; check::Bool = true) = nothing

function cover_slots(C::Curvature, ratio, site, av, sl; check::Bool = true)
    rho = Vector{magtype(eltype(site))}(undef, cover_nslots(C, sl))
    cover_slots!(rho, C, ratio, site, av, sl; check) || return nothing
    return rho
end

# How large `cover_slots` allocates. A form that does not say is measured by its scale references,
# which the contract supplies one per slot; `cover_slots!` resizes to the exact count either way,
# so this only decides whether that resize has to reallocate.
cover_nslots(::Curvature, sl) = length(sl)
cover_nslots(::UnaryCurvature, sl) = 1
cover_nslots(::BinaryCurvature, sl) = 2
cover_nslots(C::PairedCurvature, sl) = 2 * C.q
cover_nslots(C::DeclaredCurvature, sl) = length(C.s)

@doc (@doc cover_slots)
cover_slots!(rho::Vector, ::NoCurvature, ratio, site, av, sl; check::Bool = true) = false

function cover_slots!(rho::Vector, C::DeclaredCurvature, ratio, site, av, sl; check::Bool = true)
    av == 0 && return false
    resize!(rho, length(C.s))
    r = sqrt(av)
    for (i, s) in enumerate(C.s)
        rho[i] = r * s
    end
    return true
end

function cover_slots!(rho::Vector, C::PairedCurvature, ratio, site, av, sl; check::Bool = true)
    c = av
    c == 0 && return false
    q = C.q
    resize!(rho, 2q)
    fill!(rho, 0)
    for j in 1:q
        s1 = sl[j]
        s2 = sl[q + j]
        # A pair is a pure cross term, so one dead slot empties it: the pair's Hessian
        # contribution is exactly zero and neither slot carries weight.
        dead = dead_slot(s1) || dead_slot(s2)
        a, b = dead ? (zero(c), zero(c)) : split_block(ratio, zero(c), c, zero(c), s1, s2)
        if check
            check_finite_partials(site, av, (c,))
            # Allow for roundoff accumulated across the program, not just this comparison.
            rtol = sqrt(eps(magtype(eltype(site))))
            dead || a * b >= c * (1 - rtol) || error(
                "local cover invariant violated at $(site_label(site)), pair $j: rho=($a,$b), c=$c"
            )
        end
        rho[j] = a
        rho[q + j] = b
    end
    return true
end

function cover_slots!(rho::Vector, C::UnaryCurvature, ratio, site, av, sl; check::Bool = true)
    c11 = av * magnitude(C.c)
    c11 == 0 && return false
    check && check_finite_partials(site, av, (c11,))
    resize!(rho, 1)
    rho[1] = sqrt(c11)
    return true
end

function cover_slots!(rho::Vector, C::BinaryCurvature, ratio, site, av, sl; check::Bool = true)
    M = magtype(eltype(site))
    c11 = av * magnitude(C.c11)
    c12 = av * magnitude(C.c12)
    c22 = av * magnitude(C.c22)
    (c11 == 0 && c12 == 0 && c22 == 0) && return false
    s1, s2 = sl[1], sl[2]
    resize!(rho, 2)
    # A dead slot empties every entry it appears in, leaving only the live slot's own curvature —
    # which the diagonal cover already supplies. The test comes second so that a block the
    # diagonal cover serves anyway never asks for it.
    offdiag = offdiag_active(c11, c12, c22)
    dead = offdiag && (dead_slot(s1) || dead_slot(s2))
    if offdiag && !dead
        rho1, rho2 = split_block(ratio, c11, c12, c22, s1, s2)
    else
        rho1, rho2 = sqrt(c11), sqrt(c22)
    end
    if check
        check_finite_partials(site, av, (c11, c12, c22))
        rtol = sqrt(eps(M))
        (
            rho1 >= sqrt(c11) * (1 - rtol) && rho2 >= sqrt(c22) * (1 - rtol) &&
                (dead || rho1 * rho2 >= c12 * (1 - rtol))
        ) ||
            error(
            "local cover invariant violated at $(site_label(site)): " *
                "rho=($rho1,$rho2), c=($c11,$c12,$c22)"
        )
    end
    rho[1] = rho1
    rho[2] = rho2
    return true
end

# The general form carries no fixed arity, and the rank-1 cover of a block larger than 2x2 is not
# a single split.
cover_slots!(rho::Vector, C::DenseCurvature, ratio, site, av, sl; check::Bool = true) =
    error(
    "$(site_label(site)) carries a $(join(size(C), "x")) dense curvature; the rank-1 cover is " *
        "defined for the unary and binary forms, not for a general dense block"
)

"""
    bound_contribution!(B, C::Curvature, av, ps, utilde)

Accumulate node `k`'s term of the elementwise bound into `B`:
`av * sum_{i,j} magnitude(C[i,j]) * utilde[ps[i]] * utilde[ps[j]]'`, where `av` is the absolute
adjoint and `utilde` the forward absolute sensitivities. See [`hessian_bound`](@ref).

A [`DeclaredCurvature`](@ref) contributes its declared bound `s[i]*s[j] >= abs(C[i,j])`, which
factorizes, so its term is the single rank-1 outer product `w*w'` with
`w = sum_i s[i]*utilde[ps[i]]`.
"""
bound_contribution!(B, ::NoCurvature, av, ps, utilde) = B

function bound_contribution!(B, C::DeclaredCurvature, av, ps, utilde)
    n = size(B, 1)
    w = zeros(eltype(B), n)
    for (i, p) in enumerate(ps)
        si = C.s[i]
        si == 0 && continue
        up = utilde[p]
        for q in 1:n
            w[q] += si * up[q]
        end
    end
    for q in 1:n, r in 1:n
        B[q, r] += av * w[q] * w[r]
    end
    return B
end

function bound_contribution!(B, C::PairedCurvature, av, ps, utilde)
    n = size(B, 1)
    qh = C.q
    for j in 1:qh
        ui, uj = utilde[ps[j]], utilde[ps[qh + j]]
        for q in 1:n, r in 1:n
            B[q, r] += av * (ui[q] * uj[r] + uj[q] * ui[r])
        end
    end
    return B
end

function bound_contribution!(
        B, C::Union{UnaryCurvature, BinaryCurvature, DenseCurvature}, av, ps, utilde
    )
    n = size(B, 1)
    for i in eachindex(ps), j in eachindex(ps)
        cij = av * magnitude(C[i, j])
        cij == 0 && continue
        ui, uj = utilde[ps[i]], utilde[ps[j]]
        for q in 1:n, r in 1:n
            B[q, r] += cij * ui[q] * uj[r]
        end
    end
    return B
end

"""
    curvature_apply!(vbardot, C::Curvature, ps, bk, dv)

Accumulate the second-order term of a Hessian-vector product into `vbardot`:
`vbardot[ps[i]] += bk * sum_j C[i,j] * dv[ps[j]]`, where `bk` is the node's adjoint and `dv` the
forward tangents. See [`hvp`](@ref).

Defined only for a curvature that records its entries exactly; [`has_exact_partials`](@ref)
reports which do.
"""
curvature_apply!(vbardot, ::NoCurvature, ps, bk, dv) = vbardot

function curvature_apply!(vbardot, C::PairedCurvature, ps, bk, dv)
    # One partner per slot; the structural zeros are never visited.
    qh = C.q
    for j in 1:qh
        pw, px = ps[j], ps[qh + j]
        vbardot[pw] += bk * dv[px]
        vbardot[px] += bk * dv[pw]
    end
    return vbardot
end

function curvature_apply!(vbardot, C::UnaryCurvature, ps, bk, dv)
    p = ps[1]
    vbardot[p] += bk * C.c * dv[p]
    return vbardot
end

function curvature_apply!(vbardot, C::BinaryCurvature, ps, bk, dv)
    p1, p2 = ps[1], ps[2]
    d1, d2 = dv[p1], dv[p2]
    vbardot[p1] += bk * (C.c11 * d1 + C.c12 * d2)
    vbardot[p2] += bk * (C.c12 * d1 + C.c22 * d2)
    return vbardot
end

function curvature_apply!(vbardot, C::DenseCurvature, ps, bk, dv)
    for (i, p) in enumerate(ps)
        curv = zero(eltype(vbardot))
        for (j, q) in enumerate(ps)
            curv += C[i, j] * dv[q]
        end
        vbardot[p] += bk * curv
    end
    return vbardot
end
