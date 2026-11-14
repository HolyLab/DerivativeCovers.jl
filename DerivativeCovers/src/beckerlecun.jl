# The Becker–LeCun estimate of a Hessian's diagonal, and a structural test of where it is exact.
#
# The edge-pushing sweep of Gower and Mello (2012) computes the Hessian of a scalar program by
# updating a symmetric matrix V indexed by the tape nodes. When node ℓ is reached, the entries of
# V are the second derivatives of the output regarded as a function of the values of ℓ and the
# nodes before it, each treated as independent. Eliminating ℓ updates the diagonal entry of each
# parent p as
#
#     V_pp += (∂v_ℓ/∂v_p)² V_ℓℓ + vbar_ℓ ∂²v_ℓ/∂v_p² + 2 (∂v_ℓ/∂v_p) V_ℓp,
#
# where the derivatives of v_ℓ hold its other parents fixed: ∂v_ℓ/∂v_p sums the first partials of
# the slots `p` occupies, and ∂²v_ℓ/∂v_p² the second partials between those slots. The
# Becker–LeCun estimate keeps one number per node in place of V_ℓℓ and drops the last term, the
# only route by which off-diagonal entries of V reach its diagonal. It is therefore exact at every
# input from which no node p with V_ℓp nonzero, for some child ℓ of p, can be reached.

"""
    BeckerLeCunEstimate{T}

The result of [`becker_lecun`](@ref), for a tape with element type `T`.

- `grad::Vector{T}`: the gradient of the scalar function.
- `diag::Vector{T}`: the Becker–LeCun estimate of the diagonal of the Hessian, entry `i`
  approximating `H[i,i]`. It may have either sign, and in general it bounds nothing; see
  [`becker_lecun`](@ref) for when it gives the minimal cover.
- `exact::Union{BitVector, Nothing}`: `exact[i]` is `true` where the estimate is certified to
  equal `H[i,i]` in exact arithmetic at the recorded point, and `false` where it could not be
  certified. `nothing` unless the check was requested with `exactness=true`.

The test behind `exact` is structural and conservative: `false` means that terms the estimate
drops might reach `diag[i]`, not that they do or that they fail to cancel. A first or second
partial recorded as exactly zero counts as a zero, as it does throughout the package.

An exact `diag[i]` is still only the diagonal entry: it can be zero, or small compared with the
couplings of variable `i` to the others, so `sqrt(abs(diag[i]))` is a usable scale only where
the diagonal represents that variable's curvature.
"""
struct BeckerLeCunEstimate{T}
    grad::Vector{T}
    diag::Vector{T}
    exact::Union{BitVector, Nothing}
end

"""
    becker_lecun(t::Tape, outidx::Integer; exactness=false) -> BeckerLeCunEstimate
    becker_lecun(f, x0::AbstractVector; exactness=false) -> BeckerLeCunEstimate

The Becker–LeCun estimate of the diagonal of the Hessian of a scalar program, together with its
gradient, and, with `exactness=true`, a per-input certificate of where the estimate is exact.
See [`BeckerLeCunEstimate`](@ref). If all are exact and the Hessian positive semidefinite,
then `h * h'` with `h = sqrt.(estimate.diag)` is the minimal cover of the Hessian.

The estimate is one reverse sweep beside the adjoints, carrying one extra number `Ď[ℓ]` per node.
For each node `ℓ` and each of its distinct parents `p`, it applies

    Ď[p] += (Σ_r d[ℓ,r])^2 * Ď[ℓ] + vbar[ℓ] * Σ_{r,s} C[ℓ][r,s]

where `r` and `s` range over the slots of `ℓ` that `p` occupies, `d[ℓ,r]` are the first partials
and `C[ℓ]` the second partials of the node. It replaces the intermediate Hessians of the exact
backward recursion with their diagonals (Becker and LeCun, 1988), so it omits the second-order
terms that couple two children of the same node.

The exactness check simulates the sparsity pattern of the edge-pushing sweep (Gower and Mello,
2012), whose diagonal update differs from the estimate's by the term `2 (∂v_ℓ/∂v_p) V[ℓ,p]`. An
input is certified when no node `p` that it reaches through nonzero partials, itself included,
has a child `ℓ` with `V[ℓ,p]` structurally nonzero when `ℓ` is reached. That holds, for
instance, when the children of each node reach the output through separate terms of a sum, as in
a sum of squared residuals, and whenever no node has more than one child. Its cost grows with the
number of structurally nonzero entries of `V`, which is at least the number of nonzeros of the
Hessian, so it runs only when requested.

A tape carrying a node without recorded second partials, such as a declared sub-map (see
[`has_exact_partials`](@ref)), is refused.

On a tape whose values stand for sets, such as one recorded on `Interval` inputs (see
[Covers over a box](@ref)), each entry of `diag` encloses the estimate at every point of the box
and `exact[i]` certifies it at every point, so a certified `diag[i]` encloses `H[i,i]` over the
box.

# Examples

```jldoctest
julia> r = becker_lecun(x -> (x[1] - 1)^2 + (x[1] * x[2])^2, [2.0, 3.0]; exactness=true);

julia> r.diag    # H = [2+2y² 4xy; 4xy 2x²]
2-element Vector{Float64}:
 20.0
  8.0

julia> r.exact
2-element BitVector:
 1
 1

julia> r = becker_lecun(x -> (x[1] + x[1]^2)^2, [0.4]; exactness=true);

julia> r.diag, r.exact    # f'' = 8.72: the two children of x meet before the square
([5.5200000000000005], Bool[0])
```
"""
function becker_lecun(t::Tape{T}, outidx::Integer; exactness::Bool = false) where {T}
    n, N = t.ninputs, nnodes(t)
    vbar = zeros(T, N)
    vbar[outidx] = 1
    D = zeros(T, N)
    for k in N:-1:(n + 1)
        C = t.Cs[k]
        has_exact_partials(C) || throw(
            ArgumentError(
                "node $k carries a $(nameof(typeof(C))), which records no signed second " *
                    "partials; the Becker–LeCun estimate needs them"
            )
        )
        vk, Dk = vbar[k], D[k]
        # A node with a zero adjoint can still carry curvature from its consumers, as `s` does in
        # `s^2` at `s = 0`.
        vsends = !iszero(magnitude(vk))
        (vsends || !iszero(magnitude(Dk))) || continue
        ps, ds = t.parents[k], t.ds[k]
        if vsends
            for i in eachindex(ps, ds)
                vbar[ps[i]] += ds[i] * vk
            end
        end
        dispatchcurvature(C, T) do C
            becker_lecun_node!(D, C, ps, ds, vk, Dk)
        end
    end
    exact = exactness ? becker_lecun_exact(t, outidx) : nothing
    return BeckerLeCunEstimate{T}(vbar[1:n], D[1:n], exact)
end

becker_lecun(f, x0::AbstractVector; kwargs...) = becker_lecun(tape_function(f, x0)...; kwargs...)

# One node's contribution to the estimate at its parents, given its adjoint `vk` and its own
# estimate `Dk`.
function becker_lecun_node!(D, C::Curvature, ps, ds, vk, Dk)
    # A zero adjoint weights the second partials by an exact zero, even an infinite one.
    curved = !(C isa NoCurvature) && !iszero(magnitude(vk))
    (curved || !iszero(magnitude(Dk))) || return D
    if allunique(ps)
        for i in eachindex(ps, ds)
            d = ds[i]
            D[ps[i]] += curved ? d * d * Dk + vk * curvature_entry(C, i, i) : d * d * Dk
        end
    else
        # A parent in several slots, as in `x * x`: every ordered pair of its slots contributes,
        # which squares the sum of their first partials and sums their second partials.
        for i in eachindex(ps, ds), j in eachindex(ps, ds)
            ps[j] == ps[i] || continue
            D[ps[i]] += curved ? ds[i] * ds[j] * Dk + vk * curvature_entry(C, i, j) :
                ds[i] * ds[j] * Dk
        end
    end
    return D
end

# Entry `(i, j)` of a node's signed second partials, slot-aligned with its parents.
curvature_entry(C::UnaryCurvature, i, j) = C.c
curvature_entry(C::BinaryCurvature, i, j) = i == j ? (i == 1 ? C.c11 : C.c22) : C.c12
curvature_entry(C::PairedCurvature, i, j) = abs(i - j) == C.q ? 1 : 0
curvature_entry(C::DenseCurvature, i, j) = C.C[i, j]
curvature_entry(C::Curvature, i, j) = C[i, j]

# Call `f(r, s)` for each slot pair `r <= s` whose second partial is not exactly zero.
foreach_curved_pair(f, ::NoCurvature) = nothing
function foreach_curved_pair(f, C::UnaryCurvature)
    iszero(magnitude(C.c)) || f(1, 1)
    return nothing
end
function foreach_curved_pair(f, C::BinaryCurvature)
    iszero(magnitude(C.c11)) || f(1, 1)
    iszero(magnitude(C.c12)) || f(1, 2)
    iszero(magnitude(C.c22)) || f(2, 2)
    return nothing
end
function foreach_curved_pair(f, C::PairedCurvature)
    for j in 1:C.q
        f(j, C.q + j)
    end
    return nothing
end
function foreach_curved_pair(f, C::Curvature)
    m = size(C, 1)
    size(C) == (m, m) ||
        throw(DimensionMismatch("second partials must be square; got size $(size(C))"))
    for s in 1:m, r in 1:s
        iszero(magnitude(curvature_entry(C, r, s))) || f(r, s)
    end
    return nothing
end

"""
    becker_lecun_exact(t::Tape, outidx::Integer) -> exact::BitVector

The certificate of [`BeckerLeCunEstimate`](@ref): `exact[i]` is `true` where the Becker–LeCun
estimate at input `i` is structurally guaranteed to equal `H[i,i]`.

Runs the sparsity pattern of the edge-pushing sweep in reverse order. Processing node `ℓ` first
marks each parent `p` with `V[ℓ,p]` nonzero, then moves `ℓ`'s row of `V` to its parents, expands
`V[ℓ,ℓ]` over pairs of parents, and adds the entries its own second partials create where `ℓ`
reaches the output. A slot whose first partial is exactly zero transmits nothing. A marked node
`p` receives a term the estimate drops, and that error reaches the diagonal entries of every
node from which `p` is reachable through nonzero partials, so those nodes are flagged too.

Memory and time grow with the number of structurally nonzero entries of `V` and, at a node with
a nonzero `V[ℓ,ℓ]`, with the square of its number of parents.
"""
function becker_lecun_exact(t::Tape, outidx::Integer)
    n, N = t.ninputs, nnodes(t)
    transmits(d) = !iszero(magnitude(d))
    # reaches[k]: some path of nonzero first partials leads from node k to the output, so its
    # adjoint is not structurally zero.
    reaches = falses(N)
    reaches[outidx] = true
    for k in outidx:-1:(n + 1)
        reaches[k] || continue
        ds = t.ds[k]
        for (i, p) in enumerate(t.parents[k])
            transmits(ds[i]) && (reaches[p] = true)
        end
    end
    # The structurally nonzero off-diagonal entries of V, as neighbor sets of the nodes not yet
    # eliminated, and its nonzero diagonal entries.
    nbrs = Vector{Union{Nothing, Set{Int}}}(nothing, N)
    hasdiag = falses(N)
    function neighbors!(a)
        s = nbrs[a]
        s === nothing || return s
        return nbrs[a] = Set{Int}()
    end
    function couple!(a, b)
        if a == b
            hasdiag[a] = true
        else
            push!(neighbors!(a), b)
            push!(neighbors!(b), a)
        end
        return nothing
    end
    # dropped[p]: some child ℓ of p had V[ℓ,p] nonzero when it was eliminated. inexact[p]: the
    # estimate at p may differ from V[p,p].
    dropped = falses(N)
    inexact = falses(N)
    for k in N:-1:(n + 1)
        ps, ds = t.parents[k], t.ds[k]
        C = t.Cs[k]
        has_exact_partials(C) || throw(
            ArgumentError(
                "node $k carries a $(nameof(typeof(C))), which records no signed second " *
                    "partials; the Becker–LeCun estimate needs them"
            )
        )
        row = nbrs[k]
        if row !== nothing
            for (i, p) in enumerate(ps)
                transmits(ds[i]) && p in row && (dropped[p] = true)
            end
            for j in row
                delete!(nbrs[j], k)
                for (i, p) in enumerate(ps)
                    transmits(ds[i]) && couple!(p, j)
                end
            end
        end
        if hasdiag[k]
            for (i, p) in enumerate(ps), (j, q) in enumerate(ps)
                j <= i || continue
                transmits(ds[i]) && transmits(ds[j]) && couple!(p, q)
            end
        end
        if reaches[k]
            dispatchcurvature(C, eltype(t)) do C
                foreach_curved_pair((r, s) -> couple!(ps[r], ps[s]), C)
            end
        end
        if dropped[k] || inexact[k]
            for (i, p) in enumerate(ps)
                transmits(ds[i]) && (inexact[p] = true)
            end
        end
        # The row of an eliminated node is never read again.
        nbrs[k] = nothing
    end
    return BitVector(.!(dropped[1:n] .| inexact[1:n]))
end
