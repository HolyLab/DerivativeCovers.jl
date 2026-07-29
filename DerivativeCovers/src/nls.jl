# Covers for nonlinear least-squares objectives.
#
# For a residual map r: Rⁿ → Rᵐ the least-squares loss f = ½∑ₖ rₖ² has Hessian
#
#     H = JᵀJ + ∑ₖ rₖ ∇²rₖ,     J = ∂r/∂x,
#
# The first term is Gauss–Newton; the second is residual-weighted curvature. On the loss tape,
# each residual square contributes a unit-weight atom seeded at the residual. Combining these
# atoms in quadrature gives
#
#     h_gn[j] = √(∑ₖ ũ_{rₖ,j}²) ≥ √(diag(JᵀJ))[j],
#
# a symmetric cover of JᵀJ. All residual-curvature nodes share one additional group, preserving a
# cover of the full Hessian while keeping the Gauss–Newton contribution separate.

"""
    NLSCover

Result of [`cover_nls`](@ref) for the least-squares loss `f = ½∑ₖ rₖ²`.

- `grad`: gradient of the loss.
- `h`: certified cover of the full Hessian `JᵀJ + ∑ₖ rₖ ∇²rₖ`.
- `h_gn`: certified cover of the Gauss–Newton matrix `JᵀJ`; always `h_gn <= h`.
- `Utilde`: `m × n` matrix of absolute residual sensitivities, satisfying
  `Utilde[k,j] >= abs(J[k,j])`.
- `m`: the number of residuals.

Forming `Utilde` costs `O(m * L)`. When no signed paths cancel,
`h_gn == sqrt.(diag(J'J))`.
"""
struct NLSCover
    grad::Vector{Float64}
    h::Vector{Float64}
    h_gn::Vector{Float64}
    Utilde::Matrix{Float64}
    m::Int
end

"""
    nls_loss_tape(r, x0) -> (t, outidx, residx, sqidx)

Record the least-squares loss `f = ½∑ₖ rₖ(x)²` at `x0`, where `r` maps a vector of [`TVar`](@ref)s
to a vector of residual `TVar`s. Returns the [`Tape`](@ref), the loss output node index, the node
indices `residx` of the residuals `rₖ`, and the node indices `sqidx` of their squares `rₖ²`.

`r` may return residuals that are input or duplicated nodes; each is squared into its own fresh
node, so `sqidx` has one distinct entry per residual regardless. A constant residual — a fixed
offset of the loss, contributing `½c²` to `f` and a zero row to `J` — gets a node of its own
([`constantnode`](@ref)); an all-constant residual vector depends on no parameter and is an error.
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
    cover_nls(r, x0; ngroups=nothing, check=true, rule=ValueSplit()) -> NLSCover

Compute covers for the least-squares loss `f(x) = ½∑ₖ rₖ(x)²` at `x0`.

`r` must map a vector of [`TVar`](@ref)s to a vector of residuals. The returned
[`NLSCover`](@ref) contains a cover `h` of the full loss Hessian and a cover `h_gn` of the
Gauss–Newton matrix `JᵀJ`.

By default, each residual receives its own group, giving the tightest available Gauss–Newton
cover at `O(m * L)` cost. Set `ngroups` to a smaller positive integer to use fewer channels at
the cost of a potentially looser full-Hessian cover.

`rule` controls the local covers used for `h`; it does not affect `h_gn` or `Utilde`.
`check=true` verifies the local covering conditions.

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
        r, x0::AbstractVector; ngroups = nothing, check::Bool = true,
        rule::SplitRule = ValueSplit()
    )
    t, outidx, residx, sqidx = nls_loss_tape(r, x0)
    n, N = t.ninputs, nnodes(t)
    m = length(residx)

    # One reverse channel per residual produces the m×n absolute-sensitivity surrogate.
    Theta = zeros(m, N)
    for k in 1:m
        Theta[k, residx[k]] = 1.0
    end
    Abar = abs_reverse(t, Theta)
    Utilde = Abar[:, 1:n]
    h_gn = [sqrt(sum(abs2, view(Utilde, :, j))) for j in 1:n]

    groups = nls_partition(N, sqidx; ngroups)
    vbar = adjoints(t, outidx)
    h = _cover_hybrid(t, vbar, groups; check, rule)
    grad = vbar[1:n]

    return NLSCover(grad, h, h_gn, Utilde, m)
end

"""
    gn_ab(nls::NLSCover) -> (a, b)

A valid rank-1 Jacobian cover `(a, b)` — `a_k b_j ≥ |J_{kj}|` — read off the residual
sensitivities of `nls`: `a = ones(m)` and `b[j] = maximum(Utilde[:, j])`, so
`a_k b_j = b_j ≥ ũ_{rₖ,j} ≥ |J_{kj}|`. This is the uniform-weight cover, exposed for cross-checks
against a general asymmetric Jacobian cover; the tight *symmetric* Gauss–Newton cover is `nls.h_gn`,
not the corollary `‖a‖₂ b` of this pair.
"""
function gn_ab(nls::NLSCover)
    a = ones(nls.m)
    b = [maximum(view(nls.Utilde, :, j)) for j in 1:size(nls.Utilde, 2)]
    return a, b
end
