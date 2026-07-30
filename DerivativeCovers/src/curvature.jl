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

  - [`cover_slots`](@ref): the node's rank-1 cover, slot-aligned with its parents.
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
    NoCurvature() <: Curvature

The curvature of a node with no second-order term: an input node, a linear n-ary node such as
[`linearnode`](@ref), or an elemental whose curvature at the recorded point is zero. It
contributes to no cover and to no bound, so the sweeps skip it by dispatch rather than by
inspecting a matrix of zeros.
"""
struct NoCurvature <: Curvature end

"""
    DenseCurvature(C::AbstractMatrix) <: Curvature

Recorded second partials of a unary or binary elemental, a symmetric `1×1` or `2×2` matrix
slot-aligned with the node's parents.

The rank-1 cover of a block larger than `2×2` is not a single split, so [`cover_slots`](@ref)
accepts only these two arities; an elemental with more parents supplies its curvature in a form
that carries its own structure, as [`PairedCurvature`](@ref) and [`DeclaredCurvature`](@ref) do.
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
with the node's parents.

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
    dispatchcurvature(f, C::Curvature)

Call `f(C)`, branching on the built-in [`Curvature`](@ref) forms first.

[`Tape`](@ref) stores its curvatures in a container with abstract element type, so an interface
method invoked on one is a dynamic dispatch — once per node per sweep, which is a large fraction
of the sweep's cost. Testing the built-in forms by `isa` first lets the compiler resolve those
calls statically. A form defined elsewhere falls through to the same call, dispatched the
ordinary way: this is an optimization for the common cases, not a restriction on which forms are
allowed.

The concrete parameters listed are those the elementals of `overloads.jl` record. A curvature of
the same form over some other element or matrix type is correct here, just not statically
resolved.

`C` is `@nospecialize`d because specializing on it would defeat the point: a caller that knows
the concrete form would compile an instance in which every branch below but one is dead, having
already paid the dispatch this ladder exists to avoid. One instance per call site, branching at
run time, is the intended shape.
"""
@inline function dispatchcurvature(f::F, @nospecialize(C::Curvature)) where {F}
    C isa NoCurvature && return f(C)
    C isa PairedCurvature && return f(C)
    C isa DenseCurvature{Float64, Matrix{Float64}} && return f(C)
    C isa DeclaredCurvature{Float64, Vector{Float64}} && return f(C)
    return f(C)
end

# Introspection. A curvature that records entries reports them; DeclaredCurvature has none.
Base.size(C::DenseCurvature) = size(C.C)
Base.size(C::PairedCurvature) = (2C.q, 2C.q)
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
contributes are independent rank-1 terms, not one combined term for the whole node.
"""
atom_slots(::Curvature, nslots) = (1:nslots,)
atom_slots(C::PairedCurvature, nslots) = ((j, C.q + j) for j in 1:C.q)

"""
    cover_slots(C::Curvature, rule, t, k, av, scales; check=true) -> Union{Nothing,Vector{Float64}}

The rank-1 cover of node `k` of `t`, slot-aligned with its parents: nonnegative `rho` with
`rho[i]*rho[j] >= av*abs(C[i,j])`, where `av` is the absolute adjoint `abs(vbar[k])`. Returns
`nothing` when the node contributes no second-order term.

`rule` is the [`SplitRule`](@ref) that breaks the tie at off-diagonal-active blocks and `scales`
its per-node scale references, as [`split_scales`](@ref) supplies them. `check` verifies the
covering invariant against the recorded entries; a curvature that declares a bound rather than
recording entries has nothing to check against.

`t` and `k` locate the node for error reporting only. This is the per-node kernel shared by
[`local_covers`](@ref) and the [`CoverOnly`](@ref) sweep; see [`local_cover_slots`](@ref).
"""
cover_slots(::NoCurvature, rule, t, k, av, scales; check::Bool = true) = nothing

function cover_slots(C::DeclaredCurvature, rule, t, k, av, scales; check::Bool = true)
    av == 0 && return nothing
    return sqrt(av) .* C.s
end

function cover_slots(C::PairedCurvature, rule, t, k, av, scales; check::Bool = true)
    c = av
    c == 0 && return nothing
    q = C.q
    ps = t.parents[k]
    rho = zeros(2q)
    for j in 1:q
        s1 = scales[ps[j]]
        s2 = scales[ps[q + j]]
        if certifies_dead(rule) && (s1 == 0 || s2 == 0)
            continue   # dead pair: both slots stay zero; see `local_cover_slots`
        end
        a, b = balanced_split(0.0, c, 0.0, s1, s2)
        if check
            check_finite_partials(t, k, av, (c,))
            # Allow for roundoff accumulated across the tape, not just this comparison.
            rtol = sqrt(eps(float(real(eltype(t.vals)))))
            a * b >= c * (1 - rtol) ||
                error("local cover invariant violated at node $k, pair $j: rho=($a,$b), c=$c")
        end
        rho[j] = a
        rho[q + j] = b
    end
    return rho
end

function cover_slots(C::DenseCurvature, rule, t, k, av, scales; check::Bool = true)
    ps = t.parents[k]
    if length(ps) == 1
        c11 = av * abs(C[1, 1])
        c11 == 0 && return nothing
        return [sqrt(c11)]
    elseif length(ps) == 2
        c11 = av * abs(C[1, 1])
        c12 = av * abs(C[1, 2])
        c22 = av * abs(C[2, 2])
        (c11 == 0 && c12 == 0 && c22 == 0) && return nothing
        s1, s2 = scales[ps[1]], scales[ps[2]]
        if certifies_dead(rule) && (s1 == 0 || s2 == 0)
            # A dead slot contributes no diagonal or cross term; only its live partner needs
            # cover weight.
            return [s1 == 0 ? 0.0 : sqrt(c11), s2 == 0 ? 0.0 : sqrt(c22)]
        end
        if c12^2 <= c11 * c22
            rho1, rho2 = sqrt(c11), sqrt(c22)
        else
            rho1, rho2 = balanced_split(c11, c12, c22, s1, s2)
        end
        if check
            check_finite_partials(t, k, av, (c11, c12, c22))
            rtol = sqrt(eps(float(real(eltype(t.vals)))))
            (
                rho1 >= sqrt(c11) * (1 - rtol) && rho2 >= sqrt(c22) * (1 - rtol) &&
                    rho1 * rho2 >= c12 * (1 - rtol)
            ) ||
                error("local cover invariant violated at node $k: rho=($rho1,$rho2), c=($c11,$c12,$c22)")
        end
        return [rho1, rho2]
    end
    return error("node $k has $(length(ps)) parents; only unary and binary elementals are supported")
end

"""
    bound_contribution!(B, C::Curvature, av, ps, utilde)

Accumulate node `k`'s term of the elementwise bound into `B`:
`av * sum_{i,j} abs(C[i,j]) * utilde[ps[i]] * utilde[ps[j]]'`, where `av` is the absolute adjoint
and `utilde` the forward absolute sensitivities. See [`hessian_bound`](@ref).

A [`DeclaredCurvature`](@ref) contributes its declared bound `s[i]*s[j] >= abs(C[i,j])`, which
factorizes, so its term is the single rank-1 outer product `w*w'` with
`w = sum_i s[i]*utilde[ps[i]]`.
"""
bound_contribution!(B, ::NoCurvature, av, ps, utilde) = B

function bound_contribution!(B, C::DeclaredCurvature, av, ps, utilde)
    n = size(B, 1)
    w = zeros(n)
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

function bound_contribution!(B, C::DenseCurvature, av, ps, utilde)
    n = size(B, 1)
    for i in eachindex(ps), j in eachindex(ps)
        cij = av * abs(C[i, j])
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

function curvature_apply!(vbardot, C::DenseCurvature, ps, bk, dv)
    for (i, p) in enumerate(ps)
        curv = 0.0
        for (j, q) in enumerate(ps)
            curv += C[i, j] * dv[q]
        end
        vbardot[p] += bk * curv
    end
    return vbardot
end
