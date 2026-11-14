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
#
# A single residual channel instead gives the sum merge c[j] = ∑ₖ ũ_{rₖ,j}, which `gramcover(t,
# outidxs)` computes in one O(L) sweep of the residual tape; it satisfies
# h_gn[j] ≤ c[j] ≤ √n_j h_gn[j], with n_j the number of residuals that input j reaches.
#
# With `merge = SignAveragedMerge()`, the paths combine in quadrature instead, summing squared path
# products, Ũ⁽²⁾ ≤ Ũ². One O(L) sweep seeded at the residuals gives the sign-averaged
# h_gn[j] = √(∑ₖ Ũ⁽²⁾_kj) (`gramvariance`), at most the sum merge's h_gn[j] for G = m: the
# root-mean-square column norm of J under random local signs. It does not cover JᵀJ.
#
# The same quadrature sweep carries a second channel, seeded with the atoms of the
# residual-curvature nodes: it is c_var, the sign-averaged form of the residual-curvature channel
# c, and c_var ≤ c because a sum of squares of nonnegative contributions is at most the square of
# their sum. The sign-averaged h = √(h_gn² + c_var²) is the sign-averaged cover of the full
# Hessian. Because the squared channel is linear in its squared seeds and each square node's atom
# is 1, it is `cover_hessian`'s sign-averaged cover of the loss tape, split into its two sources.
# Both channels accumulate with `hypot`, so no square is formed.

"""
    NLSCover{T, M, R<:MergeRule}

Result of [`cover_nls`](@ref) for the least-squares loss `f = ½∑ₖ rₖ²`. `T` is the element type of
the [`Tape`](@ref) it was computed from, `M =` [`magtype`](@ref)`(T)` the type of the covers, and
`R` the merge rule that produced them: [`SumMerge`](@ref) or [`SignAveragedMerge`](@ref).

- `grad::Vector{T}`: gradient of the loss.
- `h::Vector{M}`: cover of the full Hessian `JᵀJ + ∑ₖ rₖ ∇²rₖ`.
- `h_gn::Vector{M}`: cover of the Gauss–Newton matrix `JᵀJ`; always `h_gn <= h`.
- `m`: the number of residuals.
- `nguarded::Int`: the number of blocks at which the split fell back on the declared input
  scales, as in [`HessianCover`](@ref). Zero means the `scales` keyword played no part in this
  cover.

For `R = SumMerge`, `h` and `h_gn` are certified covers, read off one grouped reverse sweep with
`G` residual channels and one residual-curvature channel, so together they cost `O(G * L)` time and
memory. With `G = 1`, the default, `h_gn` is the sum `∑ₖ ũ`. With `G = m`, one residual per
channel, `h_gn` is the per-residual quadrature `√(∑ₖ ũ²)`, which equals `sqrt.(diag(J'J))` when no
signed paths cancel.

For `R = SignAveragedMerge`, `h` and `h_gn` are sign-averaged covers, root-mean-square column
scales under random local signs (one independent sign per parent slot, as for the sign-averaged
cover of [`cover_hessian`](@ref)), read off one two-channel `O(L)` sweep through squared partials.
They cover the sign ensemble's mean magnitudes, **not the Hessian or `JᵀJ`**. `h_gn` is the
[`gramvariance`](@ref) of the residuals, and `h² = h_gn² + c_var²`, where `c_var` is the
sign-averaged form of the residual-curvature channel. `h_gn` equals the sum merge's `h_gn` for
`G = m` where each entry of `J` has a single path, and `h` equals `h_gn` where the residuals have
no curvature. `h` is the sign-averaged cover of [`cover_hessian`](@ref) applied to the loss tape of
[`nls_loss_tape`](@ref). Under the same declared scales, each sign-averaged cover is at most the
corresponding sum cover.

Rescaling the inputs, `x -> D*x` with `D` diagonal, maps both covers `h -> inv(D)*h` on a program
without guarded elementals; `scales` and `ratio` reach `h` only through the residual-curvature
channel.
"""
struct NLSCover{T, M, R <: MergeRule}
    grad::Vector{T}
    h::Vector{M}
    h_gn::Vector{M}
    m::Int
    nguarded::Int
end
NLSCover{T}(::R, grad, h, h_gn, m, nguarded) where {T, R <: MergeRule} =
    NLSCover{T, magtype(T), R}(grad, h, h_gn, m, nguarded)

"""
    NLSDiagnostics{T, M, R<:MergeRule}

The extended result returned by `cover_nls(WithDiagnostics(), ...)`. `T` is the element type of
the [`Tape`](@ref) it was computed from, which `grad` holds, `M =` [`magtype`](@ref)`(T)` the
type of the covers and of `Utilde`, and `R` the merge rule of the covers. It carries the fields
of [`NLSCover`](@ref) together with

- `Utilde::Matrix{M}`: `m × n` matrix of absolute residual sensitivities, satisfying
  `Utilde[k,j] >= abs(J[k,j])`.

Forming `Utilde` takes an `m`-channel reverse sweep of its own, `O(m * L)` time and `O(m * n)`
memory, whatever number of residual channels the covers use.
"""
struct NLSDiagnostics{T, M, R <: MergeRule}
    grad::Vector{T}
    h::Vector{M}
    h_gn::Vector{M}
    Utilde::Matrix{M}
    m::Int
    nguarded::Int
end
NLSDiagnostics{T}(::R, grad, h, h_gn, Utilde, m, nguarded) where {T, R <: MergeRule} =
    NLSDiagnostics{T, magtype(T), R}(grad, h, h_gn, Utilde, m, nguarded)

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
    cover_nls([mode], r, x0; merge=SumMerge(), ngroups=1, check=true, scales=nothing, ratio=nothing)

Compute covers for the least-squares loss `f(x) = ½∑ₖ rₖ(x)²` at `x0`.

`r` must map a vector of [`TVar`](@ref)s to a vector of residuals. The default
[`CoverOnly`](@ref) mode returns an [`NLSCover`](@ref) holding a cover `h` of the full loss
Hessian and a cover `h_gn` of the Gauss–Newton matrix `JᵀJ`. With `merge =`
[`SumMerge`](@ref)`()`, the default, both are certified covers. With `merge =`
[`SignAveragedMerge`](@ref)`()`, both are sign-averaged covers, the latter the
[`gramvariance`](@ref) of the residuals, which cover the mean magnitudes of a sign ensemble rather
than these matrices. [`WithDiagnostics`](@ref) returns an [`NLSDiagnostics`](@ref), which adds the
absolute-sensitivity surrogate `Utilde` for a further `O(m * L)`.

For the sum covers, `ngroups` sets the number of residual channels `G`, and with it the entire cost
of the [`CoverOnly`](@ref) mode: `O(G * L)` time and `O(G * n)` memory. The residuals are spread
cyclically over the `G` channels, and their sensitivities are summed within a channel before the
quadrature merge. The default `1` sums them all, so `h_gn` is the [`gramcover`](@ref) of the
residuals and both covers cost `O(L)`. `ngroups=nothing` gives every residual a channel of its own
(`G = m`), the tightest pair of covers available, at `O(m * L)`; its `h_gn[j]` is smaller than the
default's by a factor of at most `sqrt(n_j)`, where `n_j` is the number of residuals input `j`
reaches. That factor varies with `j`, so it changes the dispersion of the cover as well as its
level. The sign-averaged covers cost `O(L)` and do not depend on `G`, so they accept only the
default `ngroups`.

`scales` and `ratio` govern the split, as in [`cover_hessian`](@ref), and reach only the
residual-curvature channel, and through it `h`: the residual channels carry unit atoms, which
admit no split, so neither `h_gn` nor `Utilde` depends on them. `check=true` verifies the local
covering conditions, and under [`WithDiagnostics`](@ref) also verifies `h_gn` against `Utilde`.

# Examples

```jldoctest
julia> r(x) = [x[1] * x[2] - 1.0, exp(x[1]) - 2.0];

julia> nc = cover_nls(r, [0.5, 2.0]);

julia> round.(nc.h; digits=4)
2-element Vector{Float64}:
 3.7272
 0.5

julia> round.(nc.h_gn; digits=4)
2-element Vector{Float64}:
 3.6487
 0.5

julia> all(nc.h_gn .<= nc.h)
true

julia> nv = cover_nls(r, [0.5, 2.0]; merge = SignAveragedMerge());

julia> round.(nv.h; digits=4)
2-element Vector{Float64}:
 2.7014
 0.5

julia> all(nv.h_gn .<= nv.h .<= nc.h)
true
```
"""
function cover_nls(
        ::CoverOnly, r, x0::AbstractVector; merge::MergeRule = SumMerge(), ngroups = 1,
        check::Bool = true, scales = nothing, ratio = nothing
    )
    c = _nls_channels(merge, r, x0, ngroups, check, scales, ratio)
    T = eltype(c.t)
    return NLSCover{T}(merge, c.grad, c.h, c.h_gn, length(c.residx), c.nguarded)
end

function cover_nls(
        ::WithDiagnostics, r, x0::AbstractVector; merge::MergeRule = SumMerge(), ngroups = 1,
        check::Bool = true, scales = nothing, ratio = nothing
    )
    c = _nls_channels(merge, r, x0, ngroups, check, scales, ratio)
    Utilde = jacobian_surrogate(c.t, c.residx)
    check && _check_gn_channels(merge, c.h_gn, Utilde, c.rgroups, c.gres)
    T = eltype(c.t)
    return NLSDiagnostics{T}(merge, c.grad, c.h, c.h_gn, Utilde, length(c.residx), c.nguarded)
end

cover_nls(r, x0::AbstractVector; kwargs...) = cover_nls(CoverOnly(), r, x0; kwargs...)

# The residual channels of a cover give `h_gn`, and together with the residual-curvature channel
# they give `h`. For the sum covers, one grouped reverse sweep carries the `gres` residual channels
# and the curvature channel; for the sign-averaged covers, a two-channel quadrature sweep carries
# one of each.
function _nls_channels(merge::MergeRule, r, x0::AbstractVector, ngroups, check::Bool, scales, ratio)
    check_sweep_merge(merge)
    merge isa SignAveragedMerge && ngroups != 1 && throw(
        ArgumentError(
            "ngroups sets the residual channels of the sum covers; the sign-averaged covers " *
                "do not depend on it, so it cannot be combined with merge = SignAveragedMerge() " *
                "(got ngroups=$(repr(ngroups)))"
        )
    )
    t, outidx, residx, sqidx = nls_loss_tape(r, x0)
    n = t.ninputs
    groups = nls_partition(nnodes(t), sqidx; ngroups)
    gres = maximum(groups) - 1
    vbar = adjoints(t, outidx)
    etas, _, nguarded = local_covers(t, vbar; check, scales, ratio, merge)
    h, h_gn = _nls_covers(merge, t, etas, groups, gres, residx)
    return (; t, residx, rgroups = groups[sqidx], gres, grad = vbar[1:n], h, h_gn, nguarded)
end

function _nls_covers(::SumMerge, t::Tape{T}, etas, groups, gres, residx) where {T}
    Abar = abs_reverse(t, group_seeds(t, etas, groups))
    cols = 1:t.ninputs
    h = tomag(T, [sqrt(sum(abs2, view(Abar, :, j))) for j in cols])
    h_gn = tomag(T, [sqrt(sum(abs2, view(Abar, 1:gres, j))) for j in cols])
    return h, h_gn
end

function _nls_covers(::SignAveragedMerge, t::Tape{T}, etas, groups, gres, residx) where {T}
    Rho = _nls_quadrature_seeds(t, etas, groups, gres + 1, residx)
    Rbar = _reverse_sweep(hypot, _absslot, t, Rho)
    cols = 1:t.ninputs
    h = tomag(T, [hypot(Rbar[1, j], Rbar[2, j]) for j in cols])
    h_gn = tomag(T, Rbar[1, cols])
    return h, h_gn
end

# Seeds of the two quadrature channels, in root form. Row 1 places unit weight at each residual, as
# `gramvariance` does; row 2 holds the `quadrature_seeds` of the nodes in the residual-curvature
# group `curv`.
function _nls_quadrature_seeds(t::Tape{T}, etas, groups, curv, residx) where {T}
    Rho = zeros(boundtype(T), 2, nnodes(t))
    for k in residx
        Rho[1, k] = hypot(Rho[1, k], one(eltype(Rho)))
    end
    for k in (t.ninputs + 1):nnodes(t)
        groups[k] == curv || continue
        eta = etas[k]
        isempty(eta) && continue
        for (i, p) in enumerate(t.parents[k])
            Rho[2, p] = hypot(Rho[2, p], tobound(T, eta[i]))
        end
    end
    return Rho
end

# The residual channels are the square nodes' cover atoms, which are unit only because the loss
# weights each square by ½ and `v^2` has second partial 2. Reforming them from the independently
# swept surrogate audits that identity.
function _check_gn_channels(::SumMerge, h_gn, Utilde::AbstractMatrix{T}, rgroups, gres) where {T}
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

# The sign-averaged channel sums squared path products, each at most the square of the path's
# absolute product, so `h_gn[j]^2 <= ∑ₖ Utilde[k,j]^2`, with the same unit atoms at the residuals.
function _check_gn_channels(::SignAveragedMerge, h_gn, Utilde::AbstractMatrix{T}, rgroups, gres) where {T}
    rtol = eps(float(real(T)))^(3 // 4)
    ref = [sqrt(sum(abs2, view(Utilde, :, j))) for j in axes(Utilde, 2)]
    # A ratio, so `> 1` is a violation; `0/0` is a vacuous comparison.
    s, j = worst_severity(@. h_gn / (ref * (1 + rtol)))
    s > 1 && error(
        "the sign-averaged Gauss–Newton channel exceeds the surrogate's column norm at input $j: " *
            "sweep = $(h_gn[j]), surrogate = $(ref[j])"
    )
    return nothing
end

"""
    gn_ab(nls::NLSDiagnostics) -> (a, b)

A valid rank-1 Jacobian cover `(a, b)` — `a_k b_j ≥ |J_{kj}|` — read off the residual
sensitivities of `nls`: `a = ones(m)` and `b[j] = maximum(Utilde[:, j])`, so
`a_k b_j = b_j ≥ ũ_{rₖ,j} ≥ |J_{kj}|`. `b` is the max merge for unit output weights: the
tightest cover of `J` whose row scale the unweighted least-squares loss fixes at one. It serves to
cross-check a general asymmetric Jacobian cover; the symmetric Gauss–Newton cover of `JᵀJ` is
`nls.h_gn`.
"""
function gn_ab(nls::NLSDiagnostics{T, M}) where {T, M}
    a = ones(M, nls.m)
    b = [maximum(view(nls.Utilde, :, j)) for j in 1:size(nls.Utilde, 2)]
    return a, b
end
