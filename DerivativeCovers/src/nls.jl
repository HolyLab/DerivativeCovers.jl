# Gauss–Newton / least-squares covers.
#
# For a residual map r: Rⁿ → Rᵐ the least-squares loss f = ½∑ₖ rₖ² has Hessian
#
#     H = JᵀJ + ∑ₖ rₖ ∇²rₖ,     J = ∂r/∂x,
#
# the Gauss–Newton term plus the residual-weighted curvature. Taping f directly makes each
# residual a node of an ordinary scalar program, so no multi-output tape is needed: the square
# node rₖ² is unary with second partial 2, and on the loss tape its adjoint is exactly ½, so its
# local cover contributes one atom of weight √(½·2) = 1 seeded at rₖ. That atom's cover vector is
# the residual's absolute sensitivity ũ_{rₖ} (Bauer's first-order graph), so combining the
# residual atoms in quadrature — each residual its own group in the hybrid merge — reproduces
#
#     h_gn[j] = √(∑ₖ ũ_{rₖ,j}²) ≥ √(diag(JᵀJ))[j],
#
# a symmetric cover of JᵀJ and the certified analog of the Gauss–Newton (GGN) diagonal. The
# remaining nonlinear nodes (the ∑ₖ rₖ ∇²rₖ term) ride along in one further group, so the full
# merge stays a valid cover of the whole Hessian while isolating the Gauss–Newton contribution.

"""
    NLSCover

Covers and diagnostics for a least-squares loss `f = ½∑ₖ rₖ²`, as returned by [`cover_nls`](@ref).

- `grad`: the gradient of the loss at the inputs.
- `h`: a valid cover of the full loss Hessian `H = JᵀJ + ∑ₖ rₖ ∇²rₖ`, obtained by the
  residuals-as-groups hybrid merge (each residual's Gauss–Newton atom in its own quadrature
  group, the residual-curvature nodes summed in one further group).
- `h_gn`: a valid symmetric cover of the Gauss–Newton matrix `JᵀJ`, `h_gn[j] = √(∑ₖ ũ_{rₖ,j}²)`.
  It bounds `√(diag(JᵀJ))` entrywise — the certified analog of the GGN diagonal — with equality
  when no path cancellation occurs (`ũ = |J|`). Always `h_gn[j] ≤ h[j]`.
- `Utilde`: the `m × n` matrix of residual absolute sensitivities, `Utilde[k, j] = ũ_{rₖ,j} ≥
  |J_{kj}|`. A diagnostic, `O(m·L)` to form; the accessor [`gn_ab`](@ref) reads a rank-1
  Jacobian cover off it.
- `m`: the number of residuals.
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
function nls_partition(N::Integer, sqidx::AbstractVector{<:Integer}; ngroups=nothing)
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
    cover_nls(r, x0; ngroups=nothing, check=true) -> NLSCover

Rank-1 covers of the least-squares loss `f = ½∑ₖ rₖ(x)²` at `x0`, where `r` maps a vector of
[`TVar`](@ref)s to the residual vector. Builds the loss tape itself (see [`nls_loss_tape`](@ref)),
so no vector-output recording is required: the residuals are ordinary intermediate nodes.

Returns an [`NLSCover`](@ref) carrying both a cover `h` of the full Hessian and a cover `h_gn` of
the Gauss–Newton matrix `JᵀJ`. `h` comes from [`cover_hybrid`](@ref) over the
[`nls_partition`](@ref); `ngroups` controls how finely the residual atoms are separated
(`nothing`, the default, gives one group per residual and the tightest Gauss–Newton diagonal at
`O(m·L)`; a smaller integer trades tightness for fewer channels). `h_gn` is formed independently
from the residual absolute sensitivities and satisfies `h_gn[j] ≤ h[j]`; on residuals that are
linear in `x` the two coincide and equal `√(diag(JᵀJ))` exactly.

`check` is forwarded to [`cover_hybrid`](@ref) to verify the per-node cover invariants.
"""
function cover_nls(r, x0::AbstractVector; ngroups=nothing, check::Bool=true)
    t, outidx, residx, sqidx = nls_loss_tape(r, x0)
    n, N = t.ninputs, nnodes(t)
    m = length(residx)

    # Residual absolute sensitivities ũ_{rₖ}: one channel per residual, seeded 1 at its node, in a
    # single vector-mode absolute reverse sweep. Reading the input columns gives the m×n Ũ.
    Theta = zeros(m, N)
    for k in 1:m
        Theta[k, residx[k]] = 1.0
    end
    Abar = abs_reverse(t, Theta)
    Utilde = Abar[:, 1:n]
    h_gn = [sqrt(sum(abs2, view(Utilde, :, j))) for j in 1:n]

    groups = nls_partition(N, sqidx; ngroups)
    h = cover_hybrid(t, outidx, groups; check)
    grad = adjoints(t, outidx)[1:n]

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
