# Asymmetric rank-1 Jacobian covers.
#
# A cover `(a, b)` satisfies a_k b_j >= |J_kj|. For fixed positive `a`, the smallest valid input
# factor is b_j = max_k Ũ_kj/a_k, where Ũ bounds the absolute Jacobian. Sum, quadrature, and
# group-max merges trade tightness for cheaper sweeps.
#
# The gap Ũ_kj - |J_kj| comes from signed cancellation among paths from input j to output k.

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
    for (k, o) in enumerate(outidxs)
        Theta[k, o] += 1.0
    end
    return abs_reverse(t, Theta)[:, 1:t.ninputs]
end

"""
    JacobianCover

Result of [`cover_jacobian`](@ref). Each pair `(a, b)` satisfies
`a[k] * b[j] >= abs(J[k,j])`.

- `a`: output weights; `ones(m)` by default.
- `b_sum`: sum merge, computed in one `O(L)` reverse sweep.
- `b_quad`: the quadrature merge `b_j = √(∑_k (Ũ_kj/a_k)²)`.
- `b_max`: smallest valid `b` for the fixed `a` and the surrogate `Utilde`.
- `Utilde`: `m × n` absolute-sensitivity surrogate satisfying `Utilde >= abs.(J)`.
- `m`, `n`: numbers of outputs and inputs.

The merges satisfy `b_max <= b_quad <= b_sum` coordinatewise.
"""
struct JacobianCover
    a::Vector{Float64}
    b_sum::Vector{Float64}
    b_quad::Vector{Float64}
    b_max::Vector{Float64}
    Utilde::Matrix{Float64}
    m::Int
    n::Int
end

"""
    cover_jacobian(t::Tape, outidxs::AbstractVector{<:Integer}; a=nothing, check=true) -> JacobianCover
    cover_jacobian(r, x0::AbstractVector; a=nothing, check=true) -> JacobianCover

Compute asymmetric rank-1 covers of a vector-valued function's Jacobian.

Pass either a tape and output nodes, or a function and evaluation point. The result contains sum,
quadrature, and max merges of the absolute sensitivities. For each returned `b`,
`a[k] * b[j] >= abs(J[k,j])`.

`a` must contain one strictly positive weight per output; the default is `ones(m)`. Changing `a`
redistributes scale between the output and input factors without affecting validity.
`check=true` verifies the max merge against the absolute-sensitivity surrogate.

# Examples

```jldoctest
julia> r(x) = [x[1] * x[2], exp(x[1]) - x[3], x[2] + 2x[3]];

julia> jc = cover_jacobian(r, [0.5, 2.0, -1.0]);

julia> round.(jc.b_max; digits=4)
3-element Vector{Float64}:
 2.0
 1.0
 2.0

julia> all(jc.b_max .<= jc.b_quad .<= jc.b_sum)
true
```
"""
function cover_jacobian(t::Tape, outidxs::AbstractVector{<:Integer}; a = nothing, check::Bool = true)
    m, n = length(outidxs), t.ninputs
    a === nothing && (a = ones(m))
    # Result fields use output positions 1:m, independent of the axes of outidxs.
    Base.require_one_based_indexing(a)
    length(a) == m ||
        throw(DimensionMismatch("a has length $(length(a)), expected $m (one per output)"))
    all(>(0), a) || throw(ArgumentError("output weights a must be strictly positive"))

    Utilde = jacobian_surrogate(t, outidxs)
    scaled = Utilde ./ a                      # scaled[k, j] = Ũ_kj / a_k
    b_sum = vec(sum(scaled; dims = 1))
    b_quad = sqrt.(vec(sum(abs2, scaled; dims = 1)))
    b_max = vec(maximum(scaled; dims = 1))

    if check
        rtol = eps(float(real(eltype(Utilde))))^(2 // 3)
        for j in 1:n, k in 1:m
            a[k] * b_max[j] >= Utilde[k, j] * (1 - rtol) ||
                error(
                "b_max does not cover the surrogate at output $k, input $j: " *
                    "a*b = $(a[k] * b_max[j]) < Ũ = $(Utilde[k, j])"
            )
        end
    end

    return JacobianCover(collect(float.(a)), b_sum, b_quad, b_max, Utilde, m, n)
end

function cover_jacobian(r, x0::AbstractVector; a = nothing, check::Bool = true)
    t, outidxs = tape_function(r, x0)
    outidxs isa AbstractVector ||
        throw(ArgumentError("cover_jacobian needs a vector-valued program; got a scalar output"))
    return cover_jacobian(t, outidxs; a, check)
end

"""
    cover_jacobian_hybrid(t::Tape, outidxs, groups::AbstractVector{<:Integer}; a=nothing, check=true) -> b
    cover_jacobian_hybrid(t::Tape, outidxs, ngroups::Integer; a=nothing, check=true) -> b

Compute a Jacobian cover by grouping outputs.

`groups[k]` assigns output `k` to a group. Absolute sensitivities are summed within each group,
then the group totals are combined with `max`. Alternatively, pass `ngroups` to assign outputs
cyclically.

Every grouping is valid. One group reproduces `b_sum`; one group per output reproduces `b_max`.
The cost is `O(G * L)` for `G = maximum(groups)`, without forming the full surrogate.

`a` supplies the output weights and defaults to `ones(m)`. `check=true` verifies the result
against the surrogate.

# Examples

The two endpoints, on a program with three outputs:

```jldoctest
julia> r(x) = [x[1] * x[2], exp(x[1]) - x[3], x[2] + 2x[3]];

julia> t, outidxs = tape_function(r, [0.5, 2.0, -1.0]);

julia> round.(cover_jacobian_hybrid(t, outidxs, 1); digits=4)   # one group: b_sum
3-element Vector{Float64}:
 3.6487
 1.5
 3.0

julia> round.(cover_jacobian_hybrid(t, outidxs, 3); digits=4)   # one output per group: b_max
3-element Vector{Float64}:
 2.0
 1.0
 2.0
```
"""
function cover_jacobian_hybrid(
        t::Tape, outidxs::AbstractVector{<:Integer},
        groups::AbstractVector{<:Integer}; a = nothing, check::Bool = true
    )
    m, n, N = length(outidxs), t.ninputs, nnodes(t)
    # groups and a use output positions 1:m, independent of the axes of outidxs.
    Base.require_one_based_indexing(groups)
    length(groups) == m ||
        throw(DimensionMismatch("groups has length $(length(groups)), expected $m (one per output)"))
    all(>=(1), groups) || throw(ArgumentError("group indices must be positive"))
    a === nothing && (a = ones(m))
    Base.require_one_based_indexing(a)
    length(a) == m ||
        throw(DimensionMismatch("a has length $(length(a)), expected $m (one per output)"))
    all(>(0), a) || throw(ArgumentError("output weights a must be strictly positive"))

    G = maximum(groups)
    Theta = zeros(G, N)
    for (k, o) in enumerate(outidxs)
        Theta[groups[k], o] += 1.0 / a[k]
    end
    Abar = abs_reverse(t, Theta)              # G × N, group column-sums of Ũ/a at the inputs
    b = [maximum(view(Abar, :, j)) for j in 1:n]

    if check
        Utilde = jacobian_surrogate(t, outidxs)
        rtol = eps(float(real(eltype(Utilde))))^(2 // 3)
        for j in 1:n, k in 1:m
            a[k] * b[j] >= Utilde[k, j] * (1 - rtol) ||
                error(
                "group-max cover does not cover the surrogate at output $k, input $j: " *
                    "a*b = $(a[k] * b[j]) < Ũ = $(Utilde[k, j])"
            )
        end
    end
    return b
end

function cover_jacobian_hybrid(
        t::Tape, outidxs::AbstractVector{<:Integer}, ngroups::Integer;
        a = nothing, check::Bool = true
    )
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
        # A zero input weight is valid only for an input that reaches no output.
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
    # The balancing iteration indexes weights by output position.
    a = Vector{eltype(s)}(undef, length(outidxs))
    for (k, o) in enumerate(outidxs)
        a[k] = s[o]
    end
    return a
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
    MergeRule

How [`balance_cover`](@ref)'s b-step combines a column of the scaled surrogate into one input
weight: given the entries `Ũ_kj/a_k` over the outputs `k`, it returns `b_j`. Every rule that
dominates each of its terms yields a valid cover, so the choice trades tightness, never validity.

[`SumMerge`](@ref), [`QuadMerge`](@ref) and [`MaxMerge`](@ref) are the `p = 1, 2, ∞` members of
one family, so a parameterized member (a `p`-norm merge) is reachable through the same b-step
without an API break. They are types rather than names so that the b-step's doubly-nested loop
specializes on the choice, and so that the non-convergence hint can dispatch per rule.

`arule`, the other strategy axis of `balance_cover`, is a `Symbol` instead, because it is closed:
see that function's docstring.
"""
abstract type MergeRule end

"""
    SumMerge <: MergeRule

`b_j = ∑_k Ũ_kj/a_k`, the loosest merge and the only one the tape form can run as `O(L)` sweeps.
"""
struct SumMerge <: MergeRule end

"""
    QuadMerge <: MergeRule

`b_j = √(∑_k (Ũ_kj/a_k)²)`, combining the same terms in quadrature.
"""
struct QuadMerge <: MergeRule end

"""
    MaxMerge <: MergeRule

`b_j = max_k Ũ_kj/a_k`, the exact minimal `b` for the given `a`. Non-smooth, so its marginal
alternation can settle into a limit cycle; see [`balance_cover`](@ref).
"""
struct MaxMerge <: MergeRule end

"""
    balance_cover(t::Tape, outidxs; merge=SumMerge(), arule=:marginal, a0=nothing,
                  maxiter=nothing, rtol=eps(eltype(t.vals))^(2//3), check=true)
    balance_cover(r, x0::AbstractVector; kwargs...)
    balance_cover(Utilde::AbstractMatrix; merge=SumMerge(), arule=:marginal, a0=nothing,
                  maxiter=nothing, rtol=eps(eltype(Utilde))^(2//3), check=true)

Balanced asymmetric cover `(a, b)`: choose the output weights `a` by alternation, then pair them
with the [`MergeRule`](@ref) b-step ([`SumMerge`](@ref), [`QuadMerge`](@ref) or
[`MaxMerge`](@ref)). Returns `(; a, b, iters, delta)` — the pair (whose `b` is the chosen
merge for the returned `a`, so it is a valid cover of the surrogate for any `arule`), the number
of alternation rounds, and the final gauge-invariant change bound
`delta = max_{kj} |log((a_k b_j)_new / (a_k b_j)_old)|`. Throws when `delta ≤ rtol` is not reached
within `maxiter` rounds. Both a-rules are covariant: an output rescaling `r → diag(c)*r` maps the
fixed points by `a → diag(c)*a` (start `a0` transformed alongside), an input rescaling leaves them
untouched, and the remaining freedom is the gauge `(a, b) → (γa, γ⁻¹b)`.

The two strategy axes are spelled differently. `merge` is a [`MergeRule`](@ref) type: its members
are the `p = 1, 2, ∞` points of one continuum, so the b-step loop specializes on the choice and a
parameterized member would fit the existing hook. `arule` is a `Symbol`: `:marginal` and
`:geomean` are a closed pair of distinct algorithms that carry no data and admit no obvious third
member.

`maxiter = nothing` (the default) selects a per-rule budget: `1000` rounds for the marginal
alternation, `10000` for the slower `:geomean` fit; pass an explicit count to override.

- `arule = :marginal` alternates `a = ` [`marginal_weights`](@ref)`(…, b)` with the `merge`
  b-step. Every iterate is a valid cover — each half-step is valid for any positive other side —
  so the alternation trades only tightness, never validity. With [`SumMerge`](@ref) this is exactly
  Sinkhorn matrix scaling of `Ũ` (the b-step normalizes the columns of `R_kj = Ũ_kj/(a_k b_j)`,
  the a-step its rows), and it inherits Sinkhorn's convergence condition: the sparsity pattern of
  `Ũ` must have **total support** (every nonzero entry on some positive diagonal). Dense patterns
  qualify; a pattern with a nonzero entry on no positive diagonal (e.g. triangular) makes the
  iteration drift without bound — every iterate still covers, but the excess grows by a constant
  per round and the run ends in the non-convergence error. The [`MaxMerge`](@ref) b-step is
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
  `maxiter` when the non-convergence error reports a delta still well above `rtol`.

For `m ≠ n` the marginal alternation's two normalizations disagree by the factor `m/n`, a pure
gauge drift that is geometric in the round count; each round therefore renormalizes by the power
of two nearest `a`'s geometric mean, which is exact in floating point and leaves every product
`a_k b_j` bit-identical.

The tape form with `merge = SumMerge(), arule = :marginal` (the default) runs both steps as `O(L)`
sweeps ([`abs_forward`](@ref) seeded `1/b_j` at the inputs, [`abs_reverse`](@ref) seeded `1/a_k`
at the output nodes) without ever forming `Ũ` — the production path. Every other combination
needs per-output separation, so the tape form forms the surrogate once (`O(mL)`, diagnostic tier)
and balances it explicitly. The matrix form accepts any nonnegative magnitude matrix (a surrogate
`Ũ`, or `abs.(J)`).

`check` verifies the returned pair covers the surrogate. The tape form must form `Ũ` to do so,
so checking costs `O(mL)` on top of the `O(L)` production path; the matrix form already holds
`Utilde` and checks against it for the cost of one pass over its entries.

A zero row of `Ũ` (an output insensitive to every input) gets `a_k = 0`; a zero column (a dead
input) gets `b_j = 0`; both are exact and minimal.
"""
function balance_cover(
        t::Tape, outidxs::AbstractVector{<:Integer}; merge::MergeRule = SumMerge(),
        arule::Symbol = :marginal, a0 = nothing, maxiter::Union{Nothing, Integer} = nothing,
        rtol::Real = eps(float(real(eltype(t.vals))))^(2 // 3), check::Bool = true
    )
    m, N = length(outidxs), nnodes(t)
    maxiter = _resolve_maxiter(maxiter, arule)
    if !(merge isa SumMerge) || arule !== :marginal
        # Other strategies need separate output channels; materialize the surrogate once.
        return balance_cover(jacobian_surrogate(t, outidxs); merge, arule, a0, maxiter, rtol, check)
    end
    a = _balance_start(a0, m)

    bstep = a -> begin
        theta = zeros(N)
        anydead = false
        for (k, o) in enumerate(outidxs)
            if a[k] > 0
                theta[o] += 1.0 / a[k]
            else
                anydead = true
            end
        end
        if anydead
            # A zero output weight is valid only for an output with no live input path.
            ind = zeros(N)
            for (k, o) in enumerate(outidxs)
                a[k] == 0 && (ind[o] = 1.0)
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

    a, b, iters, delta = _balance(bstep, astep, a, maxiter, rtol, _marginal_hint(SumMerge()))
    if check
        Utilde = jacobian_surrogate(t, outidxs)
        r = jacobian_ratio(Utilde, a, b)
        r <= 1 + eps(float(real(eltype(Utilde))))^(2 // 3) ||
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

function balance_cover(
        Utilde::AbstractMatrix; merge::MergeRule = SumMerge(), arule::Symbol = :marginal,
        a0 = nothing, maxiter::Union{Nothing, Integer} = nothing,
        rtol::Real = eps(float(real(eltype(Utilde))))^(2 // 3), check::Bool = true
    )
    Base.require_one_based_indexing(Utilde)
    all(>=(0), Utilde) || throw(ArgumentError("the magnitude matrix must be nonnegative"))
    maxiter = _resolve_maxiter(maxiter, arule)
    m, n = size(Utilde)
    a = _balance_start(a0, m)
    bstep = a -> _merge_bstep(Utilde, a, merge)

    if arule === :marginal
        astep = b -> marginal_weights(Utilde, b)
        a, b, iters, delta = _balance(bstep, astep, a, maxiter, rtol, _marginal_hint(merge))
    elseif arule === :geomean
        a, iters, delta = _geomean_fit(Utilde, a, maxiter, rtol)
        b = bstep(a)
    else
        throw(ArgumentError("arule must be :marginal or :geomean; got $(repr(arule))"))
    end

    if check
        r = jacobian_ratio(Utilde, a, b)
        r <= 1 + eps(float(real(eltype(Utilde))))^(2 // 3) ||
            error("balanced pair does not cover the matrix: worst ratio $r")
    end
    return (; a, b, iters, delta)
end

# Merge-specific accumulation methods keep branches out of the inner loop.
_merge_accum(::SumMerge, bj, tkj) = bj + tkj
_merge_accum(::QuadMerge, bj, tkj) = bj + tkj^2
_merge_accum(::MaxMerge, bj, tkj) = max(bj, tkj)

_merge_finalize!(::MergeRule, b) = b
_merge_finalize!(::QuadMerge, b) = (b .= sqrt.(b); b)

function _merge_bstep(U, a, merge::MergeRule)
    m, n = size(U)
    b = zeros(n)
    for j in 1:n, k in 1:m
        Ukj = U[k, j]
        Ukj == 0 && continue
        a[k] > 0 ||
            throw(ArgumentError("a is zero on live output $k (Utilde[$k, $j] = $Ukj); no input cover can cover it"))
        b[j] = _merge_accum(merge, b[j], Ukj / a[k])
    end
    return _merge_finalize!(merge, b)
end

# The linearly convergent geometric-mean fit needs a larger default budget.
_resolve_maxiter(maxiter, arule::Symbol) =
    maxiter === nothing ? (arule === :geomean ? 10_000 : 1_000) : maxiter

# Non-convergence hint for the marginal alternation, specific to how each merge can fail.
_marginal_hint(::MergeRule) =
    "; the marginal alternation requires the surrogate's sparsity pattern to have total support — arule=:geomean converges for every pattern"
_marginal_hint(::MaxMerge) =
    "; the max merge's marginal alternation can enter a limit cycle — arule=:geomean converges for every pattern"

# Coordinate descent for log U_kj ≈ log a_k + log b_j over nonzero entries.
function _geomean_fit(U, a, maxiter, rtol)
    m, n = size(U)
    gstep = (weights, dim) -> begin
        out = zeros(size(U, dim))
        cnt = zeros(Int, size(U, dim))
        for j in 1:n, k in 1:m
            U[k, j] == 0 && continue
            i = dim == 1 ? k : j
            w = dim == 1 ? weights[j] : weights[k]
            out[i] += log(U[k, j]) - log(w)
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
        delta <= rtol && return a, iters, delta
    end
    error(
        "balance_cover(arule=:geomean) did not converge in $maxiter rounds: product still " *
            "changing by delta = $delta > rtol = $rtol. The geometric-mean fit converges for every " *
            "pattern but slowly on ill-conditioned supports; raise maxiter (some sparse patterns " *
            "need 1e4–1e5 rounds)."
    )
end

function _balance_start(a0, m)
    a0 === nothing && return ones(m)
    length(a0) == m ||
        throw(DimensionMismatch("a0 has length $(length(a0)), expected $m (one per output)"))
    all(>(0), a0) || throw(ArgumentError("starting weights a0 must be strictly positive"))
    return collect(float.(a0))
end

# Shared balancing loop. Renormalizing by a power of two preserves every product exactly in
# floating point. Convergence is measured on products, making `delta` invariant to the gauge
# transformation `(a, b) -> (γa, b/γ)`.
function _balance(bstep, astep, a, maxiter, rtol, hint = "")
    maxiter >= 1 || throw(ArgumentError("maxiter must be at least 1, got $maxiter"))
    b = bstep(a)
    _regauge!(a, b)
    delta = Inf
    # Max-merge alternation can enter a limit cycle. A flat, positive delta identifies a stall;
    # increasing maxiter cannot resolve it.
    stall_window, stall_flat = 128, 1.0e-6
    last_checked = Inf
    for iters in 1:maxiter
        aprev, bprev = a, b
        a = astep(b)
        b = bstep(a)
        _regauge!(a, b)
        ulo, uhi = _logchange_extrema(a, aprev)
        wlo, whi = _logchange_extrema(b, bprev)
        delta = max(uhi + whi, -(ulo + wlo))
        delta <= rtol && return a, b, iters, delta
        if iters % stall_window == 0
            if delta > rtol && abs(delta - last_checked) <= stall_flat * delta
                error(
                    "balance_cover stalled after $iters rounds: the product change " *
                        "delta = $delta has plateaued above rtol = $rtol — a limit cycle, not slow " *
                        "convergence$hint"
                )
            end
            last_checked = delta
        end
    end
    error("balance_cover did not converge in $maxiter rounds: product still changing by delta = $delta > rtol = $rtol$hint")
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
    jacobian_minimax(M; atol=eps(eltype(M))^(3//4)) -> (; a, b, excess)

The exact minimax rank-1 cover of the nonnegative matrix `M`: a pair `a_k * b_j ≥ M_kj` whose
worst-case excess `max_kj (a_k*b_j)/M_kj` over the nonzero entries of `M` is the smallest any
rank-1 cover can achieve. The optimal value is classical (tropical spectral theory / max-balancing):
it equals the maximum over bipartite cycles of the support — rows `k_1, …, k_r` and columns
`j_1, …, j_r`, all entries `(k_i, j_i)` and `(k_i, j_{i+1})` nonzero — of the alternating
geometric mean

    [ (M_{k_1 j_1} ⋯ M_{k_r j_r}) / (M_{k_1 j_2} ⋯ M_{k_r j_1}) ]^{1/r} .

The `r = 2` cycles give [`jacobian_representability`](@ref); longer cycles can dominate, so this
is the exact form of the floor that function only bounds. Every unit cancels around a cycle
(each row and column appears once in the numerator and once in the denominator), so `excess` is
dimensionless and the pair is equivariant under `M → diag(c)*M*diag(d)` modulo the gauge
`(a, b) → (γa, γ⁻¹b)`.

Computed by bisection on the log-space excess with Bellman–Ford feasibility (difference
constraints `log M_kj ≤ log a_k + log b_j ≤ log M_kj + E`), `O(nnz·(m+n))` per probe;
deterministic, no solver. `atol` is the absolute log-space bisection width — `excess` is exact to
a relative `atol` and the returned pair attains it. A zero row of `M` gets `a_k = 0`, a zero
column `b_j = 0`; both are exact. Unlike [`balance_cover`](@ref) this minimizes the worst-case
excess itself, so it is the tightness yardstick — but the optimum is generally non-unique off
the critical cycles, so prefer `balance_cover`'s rules when covariance-by-uniqueness matters.
"""
function jacobian_minimax(M::AbstractMatrix; atol::Real = eps(float(real(eltype(M))))^(3 // 4))
    Base.require_one_based_indexing(M)
    all(>=(0), M) || throw(ArgumentError("M must be nonnegative"))
    atol > 0 || throw(ArgumentError("atol must be positive, got $atol"))
    m, n = size(M)
    ks, js, cs = Int[], Int[], Float64[]
    for j in 1:n, k in 1:m
        M[k, j] > 0 && (push!(ks, k); push!(js, j); push!(cs, log(M[k, j])))
    end
    isempty(cs) && return (; a = zeros(m), b = zeros(n), excess = 1.0)

    lo, hi = 0.0, maximum(cs) - minimum(cs)
    feasible, pot = _minimax_feasible(m, n, ks, js, cs, hi)
    feasible || error("Bellman–Ford infeasible at the row-span upper bound; this cannot happen")
    while hi - lo > atol
        mid = lo + (hi - lo) / 2
        f, p = _minimax_feasible(m, n, ks, js, cs, mid)
        f ? (hi = mid; pot = p) : (lo = mid)
    end

    a, b = exp.(pot[1:m]), exp.(.-pot[(m + 1):(m + n)])
    liverow, livecol = falses(m), falses(n)
    for (k, j) in zip(ks, js)
        liverow[k] = livecol[j] = true
    end
    a[.!liverow] .= 0.0
    b[.!livecol] .= 0.0
    _regauge!(a, b)
    excess = maximum(a[k] * b[j] / M[k, j] for (k, j) in zip(ks, js))
    excess >= 1 - eps(float(real(eltype(M))))^(2 // 3) ||
        error("minimax pair does not cover M: worst ratio $(1 / excess)")
    return (; a, b, excess)
end

# The minimax constraints are feasible iff their difference-constraint graph has no negative
# cycle. Multi-source Bellman–Ford both tests feasibility and returns valid potentials.
function _minimax_feasible(m, n, ks, js, cs, E)
    pot = zeros(m + n)
    for _ in 1:(m + n)
        changed = false
        for (k, j, c) in zip(ks, js, cs)
            if pot[k] - c < pot[m + j]
                pot[m + j] = pot[k] - c
                changed = true
            end
            if pot[m + j] + c + E < pot[k]
                pot[k] = pot[m + j] + c + E
                changed = true
            end
        end
        changed || return true, pot
    end
    return false, pot
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
diagnostic. Pass `abs.(J)` or a surrogate `Ũ` as `M`. This is the 2-cycle restriction of the
exact forced excess: longer support cycles can dominate it, and [`jacobian_minimax`](@ref)
computes the exact value together with a pair attaining it.
"""
function jacobian_representability(M::AbstractMatrix)
    Base.require_one_based_indexing(M)
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
