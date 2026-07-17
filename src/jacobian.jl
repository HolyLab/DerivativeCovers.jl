# Asymmetric rank-1 covers of a Jacobian.
#
# For a vector-valued program r: Rⁿ → Rᵐ with Jacobian J = ∂r/∂x, an asymmetric cover is a pair
# (a, b) of nonnegative vectors — a over outputs, b over inputs — with a_k*b_j ≥ |J_kj| for every
# entry. Unlike the Hessian case, validity constrains each entry on its own, so the merge that
# produces b from the absolute sensitivities is a max, not a sum: for fixed a the minimal valid b
# is b_j = max_k Ũ_kj/a_k, and the cheaper sweep-friendly merges (sum, quadrature, group-max) are
# upper bounds on it. There are no local 2×2 cover problems and no ρ tie-break; the whole object is
# first-order.
#
# Ũ is the m×n surrogate of absolute sensitivities, Ũ_kj ≥ |J_kj|, one row per output — the
# first-order analog of the Hessian bound B, and loose by exactly the signed path cancellation
# between an output and an input.

"""
    jacobian_surrogate(t::Tape, outidxs::AbstractVector{<:Integer}) -> Utilde

The `m × n` surrogate of absolute sensitivities of the outputs `outidxs` with respect to the
inputs of `t`: `Utilde[k, j] ≥ |J_kj|`, tight when no signed path cancellation occurs between
output `k` and input `j`. Formed by one `m`-channel absolute-value reverse sweep seeded at the
output nodes (see [`abs_reverse`](@ref)), so it costs `O(m*L)` — a diagnostic, like the Hessian
bound `B`. A repeated entry in `outidxs` yields an identical row.
"""
function jacobian_surrogate(t::Tape, outidxs::AbstractVector{<:Integer})
    m, N = length(outidxs), nnodes(t)
    Theta = zeros(m, N)
    for k in 1:m
        Theta[k, outidxs[k]] += 1.0
    end
    return abs_reverse(t, Theta)[:, 1:t.ninputs]
end

"""
    JacobianCover

Asymmetric rank-1 covers of a Jacobian, as returned by [`cover_jacobian`](@ref). For the fixed
output weights `a`, each `b` satisfies `a_k * b_j ≥ |J_kj|` for every entry:

- `a`: the output weights the covers were built for (`ones(m)` by default).
- `b_sum`: the sum merge `b_j = ∑_k Ũ_kj/a_k` — the loosest, obtainable at `O(L)` from one
  reverse sweep (see [`cover_jacobian_hybrid`](@ref) with one group).
- `b_quad`: the quadrature merge `b_j = √(∑_k (Ũ_kj/a_k)²)`.
- `b_max`: the exact minimal `b` for this `a`, `b_j = max_k Ũ_kj/a_k`. Always
  `b_max ≤ b_quad ≤ b_sum` coordinatewise.
- `Utilde`: the `m × n` surrogate `Ũ` (see [`jacobian_surrogate`](@ref)), a diagnostic.
- `m`, `n`, `L`: output count, input count, and tape length.
"""
struct JacobianCover
    a::Vector{Float64}
    b_sum::Vector{Float64}
    b_quad::Vector{Float64}
    b_max::Vector{Float64}
    Utilde::Matrix{Float64}
    m::Int
    n::Int
    L::Int
end

"""
    cover_jacobian(t::Tape, outidxs::AbstractVector{<:Integer}; a=nothing, check=true) -> JacobianCover
    cover_jacobian(r, x0::AbstractVector; a=nothing, check=true) -> JacobianCover

Asymmetric rank-1 covers of the Jacobian of a vector-valued program: either a tape `t` with output
nodes `outidxs`, or a program `r` mapping a vector of [`TVar`](@ref)s to a vector of `TVar`s,
evaluated at `x0`. Returns a [`JacobianCover`](@ref) carrying the sum, quadrature, and exact
(max) merges of the absolute sensitivities for the fixed output weights `a`.

`a` must be strictly positive with one entry per output; `nothing` (the default) uses `ones(m)`.
Every merge is a valid cover for *any* positive `a` — the choice of `a` trades tightness between
the output and input sides but never validity. `check` verifies that `b_max` covers the surrogate.
"""
function cover_jacobian(t::Tape, outidxs::AbstractVector{<:Integer}; a=nothing, check::Bool=true)
    m, n, N = length(outidxs), t.ninputs, nnodes(t)
    a === nothing && (a = ones(m))
    length(a) == m ||
        throw(DimensionMismatch("a has length $(length(a)), expected $m (one per output)"))
    all(>(0), a) || throw(ArgumentError("output weights a must be strictly positive"))

    Utilde = jacobian_surrogate(t, outidxs)
    scaled = Utilde ./ a                      # scaled[k, j] = Ũ_kj / a_k
    b_sum = vec(sum(scaled; dims=1))
    b_quad = sqrt.(vec(sum(abs2, scaled; dims=1)))
    b_max = vec(maximum(scaled; dims=1))

    if check
        for j in 1:n, k in 1:m
            a[k] * b_max[j] >= Utilde[k, j] * (1 - 1e-10) ||
                error("b_max does not cover the surrogate at output $k, input $j: " *
                      "a*b = $(a[k] * b_max[j]) < Ũ = $(Utilde[k, j])")
        end
    end

    return JacobianCover(collect(float.(a)), b_sum, b_quad, b_max, Utilde, m, n, N)
end

function cover_jacobian(r, x0::AbstractVector; a=nothing, check::Bool=true)
    t, outidxs = tape_function(r, x0)
    outidxs isa AbstractVector ||
        throw(ArgumentError("cover_jacobian needs a vector-valued program; got a scalar output"))
    return cover_jacobian(t, outidxs; a, check)
end

"""
    cover_jacobian_hybrid(t::Tape, outidxs, groups::AbstractVector{<:Integer}; a=nothing, check=true) -> b
    cover_jacobian_hybrid(t::Tape, outidxs, ngroups::Integer; a=nothing, check=true) -> b

Group-max merge of the absolute sensitivities: partition the `m` outputs into groups, sum
`Ũ_kj/a_k` within each group, and combine the groups by `max`. The group partial sums come from a
`maximum(groups)`-channel absolute-value reverse sweep seeded `1/a_k` at the output nodes, so the
cost is `O(G*L)` without ever forming the `m × n` surrogate.

`b_j = max_g ∑_{k ∈ group g} Ũ_kj/a_k` is a valid cover for every partition: each group sum already
dominates every one of its terms `Ũ_kj/a_k`, and the max preserves that. One group reproduces the
sum merge `b_sum`; one output per group reproduces the exact merge `b_max`; intermediate partitions
interpolate, tightening as the partition is refined. An integer `ngroups` selects the cyclic
partition `1 + (k-1) % ngroups` over the outputs.

`a` is the fixed output weights (`nothing` uses `ones(m)`). `check` verifies `b` covers the
surrogate.
"""
function cover_jacobian_hybrid(t::Tape, outidxs::AbstractVector{<:Integer},
                               groups::AbstractVector{<:Integer}; a=nothing, check::Bool=true)
    m, n, N = length(outidxs), t.ninputs, nnodes(t)
    length(groups) == m ||
        throw(DimensionMismatch("groups has length $(length(groups)), expected $m (one per output)"))
    all(>=(1), groups) || throw(ArgumentError("group indices must be positive"))
    a === nothing && (a = ones(m))
    length(a) == m ||
        throw(DimensionMismatch("a has length $(length(a)), expected $m (one per output)"))
    all(>(0), a) || throw(ArgumentError("output weights a must be strictly positive"))

    G = maximum(groups)
    Theta = zeros(G, N)
    for k in 1:m
        Theta[groups[k], outidxs[k]] += 1.0 / a[k]
    end
    Abar = abs_reverse(t, Theta)              # G × N, group column-sums of Ũ/a at the inputs
    b = [maximum(view(Abar, :, j)) for j in 1:n]

    if check
        Utilde = jacobian_surrogate(t, outidxs)
        for j in 1:n, k in 1:m
            a[k] * b[j] >= Utilde[k, j] * (1 - 1e-10) ||
                error("group-max cover does not cover the surrogate at output $k, input $j: " *
                      "a*b = $(a[k] * b[j]) < Ũ = $(Utilde[k, j])")
        end
    end
    return b
end

function cover_jacobian_hybrid(t::Tape, outidxs::AbstractVector{<:Integer}, ngroups::Integer;
                               a=nothing, check::Bool=true)
    ngroups >= 1 || throw(ArgumentError("ngroups must be positive, got $ngroups"))
    m = length(outidxs)
    groups = [1 + (k - 1) % ngroups for k in 1:m]
    return cover_jacobian_hybrid(t, outidxs, groups; a, check)
end

"""
    jacobian_ratio(J, a, b) -> r

Worst-case ratio `max_kj abs(J[k,j]) / (a[k]*b[j])` over the nonzero entries of `J`. `(a, b)` is a
valid asymmetric cover of `J` exactly when `r ≤ 1`. Zero entries of `J` constrain nothing and are
skipped; a nonzero entry over a vanishing `a[k]*b[j]` gives `Inf`. The asymmetric twin of
[`cover_ratio`](@ref).
"""
function jacobian_ratio(J::AbstractMatrix, a::AbstractVector, b::AbstractVector)
    axes(J, 1) == axes(a, 1) && axes(J, 2) == axes(b, 1) ||
        throw(DimensionMismatch("J is $(axes(J)); a is $(axes(a)); b is $(axes(b))"))
    worst = zero(_ratio_type(J, a))
    for j in axes(J, 2), k in axes(J, 1)
        Jkj = abs(J[k, j])
        iszero(Jkj) && continue
        d = a[k] * b[j]
        worst = max(worst, d > 0 ? Jkj / d : oftype(worst, Inf))
    end
    return worst
end

"""
    jacobian_representability(M) -> excess

A rigorous lower bound on the worst-case excess `max_kj (a[k]*b[j]) / M[k,j]` of *any* rank-1 cover
`a[k]*b[j] ≥ M[k,j]` of the nonnegative matrix `M`: how much looseness rank-1 structure forces,
independent of the merge and surrogate. It is `√(max 2×2 defect)`, where a 2×2 submatrix on rows
`k, k'` and columns `j, j'` (all four entries nonzero) has defect
`(M_kj*M_k'j') / (M_kj'*M_k'j)`; any valid rank-1 cover has excess `≥ √defect` on that block. A
rank-1 `M` gives exactly `1`; block structure (partially separable Jacobians) drives it up.

Computed in `O(m²n)` from row-pair differences of `log M` over shared-support columns — a
diagnostic. Pass `abs.(J)` or a surrogate `Ũ` as `M`.
"""
function jacobian_representability(M::AbstractMatrix)
    m, n = size(M)
    all(>=(0), M) || throw(ArgumentError("M must be nonnegative"))
    worst = 0.0
    for k in 1:m, l in (k + 1):m
        lo = Inf
        hi = -Inf
        for j in 1:n
            (M[k, j] > 0 && M[l, j] > 0) || continue
            d = log(M[k, j]) - log(M[l, j])
            lo = min(lo, d)
            hi = max(hi, d)
        end
        hi >= lo && (worst = max(worst, hi - lo))     # ≥ 2 shared columns give a finite span
    end
    return exp(worst / 2)
end
