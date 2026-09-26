# Covers for nonlinear least-squares objectives.
#
# For a residual map r: Rⁿ → Rᵐ the least-squares loss f = ½∑ₖ rₖ² has Hessian
#
#     H = JᵀJ + ∑ₖ rₖ ∇²rₖ,     J = ∂r/∂x,
#
# The first term is Gauss–Newton; the second is residual-weighted curvature. On the loss tape the
# square node rₖ² carries adjoint ½ and second partial 2, so its local cover is the unit atom
# seeded at the residual node — the Gauss–Newton atom itself. Channel g of the grouped reverse
# sweep therefore accumulates
#
#     S_gj = ∑_{k in group g} ũ_{rₖ,j}
#
# at input j, and combining the residual channels in quadrature gives
#
#     h_gn[j] = √(∑_g S_gj²) ≥ √(diag(JᵀJ))[j],
#
# a symmetric cover of JᵀJ: h_gn[i]*h_gn[j] ≥ ∑_g S_gi*S_gj ≥ ∑ₖ ũ_{rₖ,i}*ũ_{rₖ,j} ≥ abs(JᵀJ)[i,j]
# by Cauchy–Schwarz. One residual per channel makes the first inequality an equality over the
# residuals; coarser channels only inflate it.
#
# All residual-curvature nodes share one further channel, so the same sweep covers the full
# Hessian: h² = h_gn² + (that channel)², which is also why h_gn ≤ h entry by entry.

"""
    NLSCover{T, M}

Result of [`cover_nls`](@ref) for the least-squares loss `f = ½∑ₖ rₖ²`. `T` is the element type of
the [`Tape`](@ref) it was computed from, and `M =` [`magtype`](@ref)`(T)` the type of the covers.
`NLSCover{T}` stands for `NLSCover{T, magtype(T)}`.

- `grad::Vector{T}`: gradient of the loss.
- `h::Vector{M}`: certified cover of the full Hessian `JᵀJ + ∑ₖ rₖ ∇²rₖ`.
- `h_gn::Vector{M}`: certified cover of the Gauss–Newton matrix `JᵀJ`; always `h_gn <= h`.
- `m`: the number of residuals.

Both covers are read off one grouped reverse sweep with `G` residual channels and one
residual-curvature channel, so together they cost `O(G * L)` time and memory. With `G = m` — one
residual per channel, the default — `h_gn` is the per-residual quadrature `√(∑ₖ ũ²)`, which equals
`sqrt.(diag(J'J))` when no signed paths cancel.
"""
struct NLSCover{T, M}
    grad::Vector{T}
    h::Vector{M}
    h_gn::Vector{M}
    m::Int
end
NLSCover{T}(grad, h, h_gn, m) where {T} = NLSCover{T, magtype(T)}(grad, h, h_gn, m)

"""
    NLSDiagnostics{T, M}

The extended result returned by `cover_nls(WithDiagnostics(), ...)`. `T` is the element type of
the [`Tape`](@ref) it was computed from, which `grad` holds, and `M =` [`magtype`](@ref)`(T)` the
type of the covers and of `Utilde`. `NLSDiagnostics{T}` stands for
`NLSDiagnostics{T, magtype(T)}`. It carries the fields of [`NLSCover`](@ref) together with

- `Utilde::Matrix{M}`: `m × n` matrix of absolute residual sensitivities, satisfying
  `Utilde[k,j] >= abs(J[k,j])`.

Forming `Utilde` takes an `m`-channel reverse sweep of its own, `O(m * L)` time and `O(m * n)`
memory, whatever number of residual channels the covers use.
"""
struct NLSDiagnostics{T, M}
    grad::Vector{T}
    h::Vector{M}
    h_gn::Vector{M}
    Utilde::Matrix{M}
    m::Int
end
NLSDiagnostics{T}(grad, h, h_gn, Utilde, m) where {T} =
    NLSDiagnostics{T, magtype(T)}(grad, h, h_gn, Utilde, m)

"""
    nls_loss_tape(r, x0) -> (t, outidx, residx, sqidx)

Record the least-squares loss `f = ½∑ₖ rₖ(x)²` at `x0`, where `r` maps a vector of [`TVar`](@ref)s
to a vector of residual `TVar`s. Returns the [`Tape`](@ref), the loss output node index, the node
indices `residx` of the residuals `rₖ`, and the node indices `sqidx` of their squares `rₖ²`.

`r` may return residuals that are input or duplicated nodes; each is squared into its own fresh
node, so `sqidx` has one distinct entry per residual regardless. A constant residual — a fixed
offset of the loss, contributing `½c²` to `f` and a zero row to `J` — gets a node of its own
([`constantnode`](@ref)); an all-constant residual vector depends on no parameter and is an error,
as is a recording that [`check_values`](@ref) rejects.
"""
function nls_loss_tape(r, x0::AbstractVector)
    t = Tape(x0)
    xs = [TVar(t, i) for i in 1:t.ninputs]
    rs = r(xs)
    rs isa AbstractVector ||
        throw(ArgumentError("residual map must return an AbstractVector of TVars, got a $(typeof(rs))"))
    isempty(rs) && throw(ArgumentError("residual map returned no residuals"))
    all(v -> v isa TVar, rs) ||
        throw(ArgumentError("residual map must return TVars; got element types $(unique(typeof.(rs)))"))
    all(isconstant, rs) &&
        throw(ArgumentError("every residual is a constant; none of them depends on the parameters"))
    all(v -> isconstant(v) || v.tape === t, rs) ||
        throw(ArgumentError("residuals were recorded on a different tape than the inputs"))
    rnodes = [isconstant(v) ? constantnode(t, value(v)) : v for v in rs]
    residx = [v.idx for v in rnodes]
    sqs = [v^2 for v in rnodes]
    sqidx = [s.idx for s in sqs]
    loss = 0.5 * sum(sqs)
    check_values(t)
    return t, loss.idx, residx, sqidx
end

"""
    nls_partition(N, sqidx; ngroups=nothing) -> groups

Node partition for the residuals-as-groups hybrid merge of a least-squares loss tape with
`N` nodes and residual-square nodes `sqidx`. Each residual square is placed in a group of
its own (`ngroups === nothing`, one group per residual, giving the exact Gauss–Newton diagonal in
quadrature) or spread cyclically across `ngroups` groups (a cheaper upper bound). Every other node
— the residual-curvature subgraph — is placed in one further group, so its second-order
contribution is summed there and combined in quadrature with the Gauss–Newton groups.

That last group is always occupied, so `maximum(groups) - 1` recovers the number of residual
groups even when `ngroups` exceeds the residual count.
"""
function nls_partition(N::Integer, sqidx::AbstractVector{<:Integer}; ngroups = nothing)
    m = length(sqidx)
    m >= 1 || throw(ArgumentError("need at least one residual"))
    gres = ngroups === nothing ? m : min(Int(ngroups), m)
    gres >= 1 || throw(ArgumentError("ngroups must be positive, got $ngroups"))
    curv = gres + 1
    groups = fill(curv, N)
    for (i, k) in enumerate(sqidx)
        groups[k] = 1 + (i - 1) % gres
    end
    return groups
end

"""
    cover_nls([mode], r, x0; ngroups=nothing, check=true, scales=nothing, ratio=nothing)

Compute covers for the least-squares loss `f(x) = ½∑ₖ rₖ(x)²` at `x0`.

`r` must map a vector of [`TVar`](@ref)s to a vector of residuals. The default
[`CoverOnly`](@ref) mode returns an [`NLSCover`](@ref) holding a cover `h` of the full loss
Hessian and a cover `h_gn` of the Gauss–Newton matrix `JᵀJ`. [`WithDiagnostics`](@ref) returns an
[`NLSDiagnostics`](@ref), which adds the absolute-sensitivity surrogate `Utilde` for a further
`O(m * L)`.

`ngroups` sets the number of residual channels `G`, and with it the entire cost of the
[`CoverOnly`](@ref) mode: `O(G * L)` time and `O(G * n)` memory. The default `nothing` gives every
residual a channel of its own (`G = m`), the tightest pair of covers available. A smaller positive
integer spreads the residuals cyclically over `G` channels and sums their sensitivities within a
channel before the quadrature merge; both covers remain valid and can only loosen.

`scales` and `ratio` govern the split, as in [`cover_hessian`](@ref), and reach only the
residual-curvature channel: the residual channels carry unit atoms, which admit no split, so
neither `h_gn` nor `Utilde` depends on them. `check=true`
verifies the local covering conditions, and under [`WithDiagnostics`](@ref) also verifies `h_gn`
against `Utilde`.

# Examples

```jldoctest
julia> r(x) = [x[1] * x[2] - 1.0, exp(x[1]) - 2.0];

julia> nc = cover_nls(r, [0.5, 2.0]);

julia> round.(nc.h; digits=4)
2-element Vector{Float64}:
 2.7014
 0.5

julia> round.(nc.h_gn; digits=4)
2-element Vector{Float64}:
 2.592
 0.5

julia> all(nc.h_gn .<= nc.h)
true
```
"""
function cover_nls(
        ::CoverOnly, r, x0::AbstractVector; ngroups = nothing, check::Bool = true,
        scales = nothing, ratio = nothing
    )
    c = _nls_channels(r, x0, ngroups, check, scales, ratio)
    return NLSCover(c.grad, c.h, c.h_gn, length(c.residx))
end

function cover_nls(
        ::WithDiagnostics, r, x0::AbstractVector; ngroups = nothing, check::Bool = true,
        scales = nothing, ratio = nothing
    )
    c = _nls_channels(r, x0, ngroups, check, scales, ratio)
    Utilde = jacobian_surrogate(c.t, c.residx)
    check && _check_gn_channels(c.h_gn, Utilde, c.rgroups, c.gres)
    return NLSDiagnostics(c.grad, c.h, c.h_gn, Utilde, length(c.residx))
end

cover_nls(r, x0::AbstractVector; kwargs...) = cover_nls(CoverOnly(), r, x0; kwargs...)

# One grouped reverse sweep supplies both covers: the residual channels `1:gres` merge to `h_gn`,
# and all the channels together merge to `h`.
function _nls_channels(r, x0::AbstractVector, ngroups, check::Bool, scales, ratio)
    t, outidx, residx, sqidx = nls_loss_tape(r, x0)
    n = t.ninputs
    groups = nls_partition(nnodes(t), sqidx; ngroups)
    gres = maximum(groups) - 1
    vbar = adjoints(t, outidx)
    rhos, _ = local_covers(t, vbar; check, scales, ratio)
    Abar = abs_reverse(t, group_seeds(t, rhos, groups))
    h = [sqrt(sum(abs2, view(Abar, :, j))) for j in 1:n]
    h_gn = [sqrt(sum(abs2, view(Abar, 1:gres, j))) for j in 1:n]
    return (; t, residx, rgroups = groups[sqidx], gres, grad = vbar[1:n], h, h_gn)
end

# The residual channels are the square nodes' cover atoms, which are unit only because the loss
# weights each square by ½ and `v^2` has second partial 2. Reforming them from the independently
# swept surrogate audits that identity.
function _check_gn_channels(h_gn, Utilde::AbstractMatrix{T}, rgroups, gres) where {T}
    S = zeros(T, gres)
    rtol = eps(float(real(T)))^(3 // 4)
    ref = similar(h_gn, axes(Utilde, 2))
    for j in axes(Utilde, 2)
        fill!(S, 0)
        for (k, g) in enumerate(rgroups)
            S[g] += Utilde[k, j]
        end
        ref[j] = sqrt(sum(abs2, S))
    end
    # Relative disagreement in units of the tolerance, so `> 1` is a mismatch.
    s, j = worst_severity(@. abs(h_gn - ref) / (rtol * max(ref, one(T))))
    s > 1 && error(
        "grouped Gauss–Newton channels disagree with the surrogate at input $j: " *
            "sweep = $(h_gn[j]), surrogate = $(ref[j])"
    )
    return nothing
end

"""
    gn_ab(nls::NLSDiagnostics) -> (a, b)

A valid rank-1 Jacobian cover `(a, b)` — `a_k b_j ≥ |J_{kj}|` — read off the residual
sensitivities of `nls`: `a = ones(m)` and `b[j] = maximum(Utilde[:, j])`, so
`a_k b_j = b_j ≥ ũ_{rₖ,j} ≥ |J_{kj}|`. This is the uniform-weight cover, exposed for cross-checks
against a general asymmetric Jacobian cover; the tight *symmetric* Gauss–Newton cover is `nls.h_gn`,
not the corollary `‖a‖₂ b` of this pair.
"""
function gn_ab(nls::NLSDiagnostics{T, M}) where {T, M}
    a = ones(M, nls.m)
    b = [maximum(view(nls.Utilde, :, j)) for j in 1:size(nls.Utilde, 2)]
    return a, b
end
