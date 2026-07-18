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
    abs_forward(t::Tape, theta::AbstractVector) -> s

One forward sweep in absolute-value arithmetic, the transpose of [`abs_reverse`](@ref):
`s[k] = theta[k] + sum_i abs(∂φ_k/∂v_{p_i}) * s[p_i]` over the parents `p_i` of node `k`, in tape
order. Seeding `theta[j] = 1` at a single input `j` gives, at every node, the absolute sensitivity
of that node to input `j` — the `j`-th column of the surrogate `Ũ` read at the output nodes — and
the sweep is linear in the seeds, so any weighted combination of inputs costs the same one `O(L)`
traversal.
"""
function abs_forward(t::Tape, theta::AbstractVector)
    length(theta) == nnodes(t) ||
        throw(DimensionMismatch("theta has length $(length(theta)), expected $(nnodes(t)) (one per node)"))
    s = collect(float.(theta))
    for k in (t.ninputs + 1):nnodes(t)
        acc = s[k]
        for (i, p) in enumerate(t.parents[k])
            a = abs(t.ds[k][i])
            a == 0 && continue
            acc += a * s[p]
        end
        s[k] = acc
    end
    return s
end

"""
    marginal_weights(t::Tape, outidxs::AbstractVector{<:Integer}, b::AbstractVector) -> a
    marginal_weights(Utilde::AbstractMatrix, b::AbstractVector) -> a

The output weights `a_k = ∑_j Ũ_kj / b_j` — the row marginal of the surrogate weighted by the
current input cover `b`. Each term of the sum already certifies its own entry (`a_k b_j ≥ Ũ_kj`),
so `(a, b)` is a valid cover of the surrogate for *any* positive `b`.

This is the covariant output-weighting rule: under an output rescaling `r → diag(c)*r` it picks up
exactly `diag(c)`, and under an input rescaling `x → diag(d)*x` (with `b → diag(d)⁻¹*b` alongside)
it is exactly invariant — each term `Ũ_kj/b_j` carries the units of `r_k` alone. The bare row
marginal `∑_j Ũ_kj` lacks the second property, and the value rule `a_k = |r_k|` degenerates
wherever residuals vanish.

The tape method costs one `O(L)` [`abs_forward`](@ref) sweep seeded `1/b_j` at the inputs, never
forming `Ũ`. A zero `b_j` is accepted only for a dead input (one no output depends on): it
contributes nothing to any marginal. A zero `b_j` on a live input is an error — no output
weighting can restore validity there.
"""
function marginal_weights(t::Tape, outidxs::AbstractVector{<:Integer}, b::AbstractVector)
    length(b) == t.ninputs ||
        throw(DimensionMismatch("b has length $(length(b)), expected $(t.ninputs) (one per input)"))
    all(>=(0), b) || throw(ArgumentError("input cover b must be nonnegative"))
    N = nnodes(t)
    theta = zeros(N)
    anydead = false
    for j in 1:t.ninputs
        if b[j] > 0
            theta[j] = 1.0 / b[j]
        else
            anydead = true
        end
    end
    if anydead
        # A zero b[j] is only coverable when input j is dead: check reachability with an
        # indicator sweep before silently dropping its contribution.
        ind = zeros(N)
        for j in 1:t.ninputs
            b[j] == 0 && (ind[j] = 1.0)
        end
        reach = abs_forward(t, ind)
        for k in outidxs
            reach[k] == 0 ||
                throw(ArgumentError("b is zero on an input that reaches output node $k; no output weighting can cover it"))
        end
    end
    s = abs_forward(t, theta)
    return s[outidxs]
end

function marginal_weights(Utilde::AbstractMatrix, b::AbstractVector)
    Base.require_one_based_indexing(Utilde, b)
    m, n = size(Utilde)
    length(b) == n ||
        throw(DimensionMismatch("b has length $(length(b)), expected $n (one per input)"))
    all(>=(0), b) || throw(ArgumentError("input cover b must be nonnegative"))
    a = zeros(m)
    for j in 1:n, k in 1:m
        Ukj = Utilde[k, j]
        Ukj == 0 && continue
        b[j] > 0 ||
            throw(ArgumentError("b is zero on live input $j (Utilde[$k, $j] = $Ukj); no output weighting can cover it"))
        a[k] += Ukj / b[j]
    end
    return a
end

"""
    balance_cover(t::Tape, outidxs; merge=:sum, arule=:marginal, a0=nothing,
                  maxiter=nothing, tol=1e-10, check=true)
    balance_cover(r, x0::AbstractVector; kwargs...)
    balance_cover(Utilde::AbstractMatrix; merge=:sum, arule=:marginal, a0=nothing,
                  maxiter=nothing, tol=1e-10)

Balanced asymmetric cover `(a, b)`: choose the output weights `a` by alternation, then pair them
with the `merge` b-step `b_j = ∑_k Ũ_kj/a_k` (`:sum`; `:quad` and `:max` combine the same terms in
quadrature or by maximum). Returns `(; a, b, iters, delta)` — the pair (whose `b` is the chosen
merge for the returned `a`, so it is a valid cover of the surrogate for any `arule`), the number
of alternation rounds, and the final gauge-invariant change bound
`delta = max_{kj} |log((a_k b_j)_new / (a_k b_j)_old)|`. Throws when `delta ≤ tol` is not reached
within `maxiter` rounds. Both a-rules are covariant: an output rescaling `r → diag(c)*r` maps the
fixed points by `a → diag(c)*a` (start `a0` transformed alongside), an input rescaling leaves them
untouched, and the remaining freedom is the gauge `(a, b) → (γa, γ⁻¹b)`.

`maxiter = nothing` (the default) selects a per-rule budget: `1000` rounds for the marginal
alternation, `10000` for the slower `:geomean` fit; pass an explicit count to override.

- `arule = :marginal` alternates `a = ` [`marginal_weights`](@ref)`(…, b)` with the `merge`
  b-step. Every iterate is a valid cover — each half-step is valid for any positive other side —
  so the alternation trades only tightness, never validity. With `merge = :sum` this is exactly
  Sinkhorn matrix scaling of `Ũ` (the b-step normalizes the columns of `R_kj = Ũ_kj/(a_k b_j)`,
  the a-step its rows), and it inherits Sinkhorn's convergence condition: the sparsity pattern of
  `Ũ` must have **total support** (every nonzero entry on some positive diagonal). Dense patterns
  qualify; a pattern with a nonzero entry on no positive diagonal (e.g. triangular) makes the
  iteration drift without bound — every iterate still covers, but the excess grows by a constant
  per round and the run ends in the non-convergence error. The `merge = :max` b-step is
  non-smooth and can instead settle into a **limit cycle**: the per-round product change plateaus
  at a positive constant that no larger `maxiter` reduces (independent of total support). This is
  detected and reported as a stall, distinct from the drift error; `arule = :geomean` is the
  convergent alternative in both cases.
- `arule = :geomean` sets `a_k` to the geometric mean of `Ũ_kj/b_j` over the nonzero entries of
  row `k`, alternating with the same rule on the b-side. In log space this is coordinate descent
  on the additive two-way fit `log Ũ_kj ≈ log a_k + log b_j` (row and column effects), so it
  converges for **every** sparsity pattern; the intermediate iterates undershoot validity, which
  the final `merge` b-step restores. Convergence is linear and can be slow on ill-conditioned
  supports (chains, wide bands): some sparse patterns need `10^4`–`10^5` rounds, so raise
  `maxiter` when the non-convergence error reports a delta still well above `tol`.

For `m ≠ n` the marginal alternation's two normalizations disagree by the factor `m/n`, a pure
gauge drift that is geometric in the round count; each round therefore renormalizes by the power
of two nearest `a`'s geometric mean, which is exact in floating point and leaves every product
`a_k b_j` bit-identical.

The tape form with `merge = :sum, arule = :marginal` (the default) runs both steps as `O(L)`
sweeps ([`abs_forward`](@ref) seeded `1/b_j` at the inputs, [`abs_reverse`](@ref) seeded `1/a_k`
at the output nodes) without ever forming `Ũ` — the production path. Every other combination
needs per-output separation, so the tape form forms the surrogate once (`O(mL)`, diagnostic tier)
and balances it explicitly. The matrix form accepts any nonnegative magnitude matrix (a surrogate
`Ũ`, or `abs.(J)`).

`check` verifies the returned pair covers the surrogate (`O(mL)`; matrix form checks against
`Utilde` directly). A zero row of `Ũ` (an output insensitive to every input) gets `a_k = 0`; a
zero column (a dead input) gets `b_j = 0`; both are exact and minimal.
"""
function balance_cover(t::Tape, outidxs::AbstractVector{<:Integer}; merge::Symbol=:sum,
                       arule::Symbol=:marginal, a0=nothing, maxiter::Union{Nothing,Integer}=nothing,
                       tol::Real=1e-10, check::Bool=true)
    m, N = length(outidxs), nnodes(t)
    maxiter = _resolve_maxiter(maxiter, arule)
    if merge !== :sum || arule !== :marginal
        # Anything beyond the sum/marginal alternation needs per-output channels, so balance the
        # explicit surrogate instead.
        return balance_cover(jacobian_surrogate(t, outidxs); merge, arule, a0, maxiter, tol)
    end
    a = _balance_start(a0, m)

    bstep = a -> begin
        theta = zeros(N)
        anydead = false
        for k in 1:m
            if a[k] > 0
                theta[outidxs[k]] += 1.0 / a[k]
            else
                anydead = true
            end
        end
        if anydead
            # A zero a[k] is only coverable when output k is dead: check with an indicator sweep.
            ind = zeros(N)
            for k in 1:m
                a[k] == 0 && (ind[outidxs[k]] = 1.0)
            end
            reach = abs_reverse(t, ind)
            for j in 1:t.ninputs
                reach[j] == 0 ||
                    throw(ArgumentError("a is zero on an output that input $j reaches; no input cover can cover it"))
            end
        end
        abs_reverse(t, theta)[1:t.ninputs]
    end
    astep = b -> marginal_weights(t, outidxs, b)

    a, b, iters, delta = _balance(bstep, astep, a, maxiter, tol, _marginal_hint(:sum))
    if check
        Utilde = jacobian_surrogate(t, outidxs)
        r = jacobian_ratio(Utilde, a, b)
        r <= 1 + 1e-10 ||
            error("balanced pair does not cover the surrogate: worst ratio $r")
    end
    return (; a, b, iters, delta)
end

function balance_cover(r, x0::AbstractVector; kwargs...)
    t, outidxs = tape_function(r, x0)
    outidxs isa AbstractVector ||
        throw(ArgumentError("balance_cover needs a vector-valued program; got a scalar output"))
    return balance_cover(t, outidxs; kwargs...)
end

function balance_cover(Utilde::AbstractMatrix; merge::Symbol=:sum, arule::Symbol=:marginal,
                       a0=nothing, maxiter::Union{Nothing,Integer}=nothing, tol::Real=1e-10)
    Base.require_one_based_indexing(Utilde)
    all(>=(0), Utilde) || throw(ArgumentError("the magnitude matrix must be nonnegative"))
    merge in (:sum, :quad, :max) ||
        throw(ArgumentError("merge must be :sum, :quad, or :max; got $(repr(merge))"))
    maxiter = _resolve_maxiter(maxiter, arule)
    m, n = size(Utilde)
    a = _balance_start(a0, m)
    bstep = a -> _merge_bstep(Utilde, a, merge)

    if arule === :marginal
        astep = b -> marginal_weights(Utilde, b)
        a, b, iters, delta = _balance(bstep, astep, a, maxiter, tol, _marginal_hint(merge))
    elseif arule === :geomean
        a, iters, delta = _geomean_fit(Utilde, a, maxiter, tol)
        b = bstep(a)
    else
        throw(ArgumentError("arule must be :marginal or :geomean; got $(repr(arule))"))
    end

    r = jacobian_ratio(Utilde, a, b)
    r <= 1 + 1e-10 || error("balanced pair does not cover the matrix: worst ratio $r")
    return (; a, b, iters, delta)
end

function _merge_bstep(U, a, merge)
    m, n = size(U)
    b = zeros(n)
    for j in 1:n, k in 1:m
        Ukj = U[k, j]
        Ukj == 0 && continue
        a[k] > 0 ||
            throw(ArgumentError("a is zero on live output $k (Utilde[$k, $j] = $Ukj); no input cover can cover it"))
        tkj = Ukj / a[k]
        if merge === :sum
            b[j] += tkj
        elseif merge === :quad
            b[j] += tkj^2
        else
            b[j] = max(b[j], tkj)
        end
    end
    merge === :quad && (b .= sqrt.(b))
    return b
end

# Per-rule default iteration budget: the marginal alternation converges in a few hundred rounds
# when it converges at all, while the geomean fit is linearly convergent and can need many more.
_resolve_maxiter(maxiter, arule::Symbol) =
    maxiter === nothing ? (arule === :geomean ? 10_000 : 1_000) : maxiter

# Non-convergence hint for the marginal alternation, specific to how each merge can fail.
_marginal_hint(merge::Symbol) =
    merge === :max ?
        "; the max merge's marginal alternation can enter a limit cycle — arule=:geomean converges for every pattern" :
        "; the marginal alternation requires the surrogate's sparsity pattern to have total support — arule=:geomean converges for every pattern"

# Alternating geometric-mean fit of log U_kj ≈ log a_k + log b_j over the nonzero entries:
# coordinate descent on the two-way least-squares objective, convergent for every pattern.
function _geomean_fit(U, a, maxiter, tol)
    m, n = size(U)
    gstep = (weights, dim) -> begin
        out = zeros(size(U, dim))
        cnt = zeros(Int, size(U, dim))
        for j in 1:n, k in 1:m
            U[k, j] == 0 && continue
            i = dim == 1 ? k : j
            w = dim == 1 ? weights[j] : weights[k]
            out[i] += log(U[k, j] / w)
            cnt[i] += 1
        end
        [cnt[i] > 0 ? exp(out[i] / cnt[i]) : 0.0 for i in eachindex(out)]
    end
    b = gstep(a, 2)
    _regauge!(a, b)
    delta = Inf
    for iters in 1:maxiter
        aprev, bprev = a, b
        a = gstep(b, 1)
        b = gstep(a, 2)
        _regauge!(a, b)
        ulo, uhi = _logchange_extrema(a, aprev)
        wlo, whi = _logchange_extrema(b, bprev)
        delta = max(uhi + whi, -(ulo + wlo))
        delta <= tol && return a, iters, delta
    end
    error("balance_cover(arule=:geomean) did not converge in $maxiter rounds: product still " *
          "changing by delta = $delta > tol = $tol. The geometric-mean fit converges for every " *
          "pattern but slowly on ill-conditioned supports; raise maxiter (some sparse patterns " *
          "need 1e4–1e5 rounds).")
end

function _balance_start(a0, m)
    a0 === nothing && return ones(m)
    length(a0) == m ||
        throw(DimensionMismatch("a0 has length $(length(a0)), expected $m (one per output)"))
    all(>(0), a0) || throw(ArgumentError("starting weights a0 must be strictly positive"))
    return collect(float.(a0))
end

# The alternation loop shared by the tape and matrix forms. Each round runs the a-step from the
# current b and the b-step from the new a, then renormalizes the gauge by the power of two
# nearest a's geometric mean (exact in floating point, so every product a_k*b_j is untouched).
# Convergence is judged on the products: with u = log.(a ./ a_prev) and w = log.(b ./ b_prev)
# (zero coordinates excluded — their zeros persist), the change of product (k, j) is u_k + w_j,
# whose extremes are attained at the extremes of u and w. delta bounds |u_k + w_j| exactly and is
# gauge-invariant, since a uniform shift of u cancels against the opposite shift of w.
function _balance(bstep, astep, a, maxiter, tol, hint="")
    maxiter >= 1 || throw(ArgumentError("maxiter must be at least 1, got $maxiter"))
    b = bstep(a)
    _regauge!(a, b)
    delta = Inf
    # Limit-cycle guard: the max-merge marginal alternation is non-smooth and can settle into a
    # cycle whose per-round product change plateaus at a positive constant — no larger maxiter
    # helps. When delta goes flat (unchanged to `stall_flat` over `stall_window` rounds) while
    # still above tol, the run is cycling, not converging; report the stall and point at geomean.
    # A genuinely converging alternation shrinks delta geometrically, so it never reads flat.
    stall_window, stall_flat = 128, 1e-6
    last_checked = Inf
    for iters in 1:maxiter
        aprev, bprev = a, b
        a = astep(b)
        b = bstep(a)
        _regauge!(a, b)
        ulo, uhi = _logchange_extrema(a, aprev)
        wlo, whi = _logchange_extrema(b, bprev)
        delta = max(uhi + whi, -(ulo + wlo))
        delta <= tol && return a, b, iters, delta
        if iters % stall_window == 0
            if delta > tol && abs(delta - last_checked) <= stall_flat * delta
                error("balance_cover stalled after $iters rounds: the product change " *
                      "delta = $delta has plateaued above tol = $tol — a limit cycle, not slow " *
                      "convergence$hint")
            end
            last_checked = delta
        end
    end
    error("balance_cover did not converge in $maxiter rounds: product still changing by delta = $delta > tol = $tol$hint")
end

function _regauge!(a, b)
    lsum = 0.0
    nnz = 0
    for x in a
        x > 0 || continue
        lsum += log2(x)
        nnz += 1
    end
    nnz == 0 && return
    gamma = exp2(round(lsum / nnz))
    a ./= gamma
    b .*= gamma
    return
end

function _logchange_extrema(v, vprev)
    lo, hi = Inf, -Inf
    for i in eachindex(v, vprev)
        x, xp = v[i], vprev[i]
        if x > 0 && xp > 0
            l = log(x / xp)
            lo = min(lo, l)
            hi = max(hi, l)
        elseif (x > 0) != (xp > 0)
            return -Inf, Inf                  # a coordinate appeared or vanished: not converged
        end
    end
    lo <= hi || return 0.0, 0.0               # all-zero vector: nothing changed
    return lo, hi
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
