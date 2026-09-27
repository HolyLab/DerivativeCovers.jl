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
function jacobian_surrogate(t::Tape{T}, outidxs::AbstractVector{<:Integer}) where {T}
    m, N = length(outidxs), nnodes(t)
    Theta = zeros(magtype(T), m, N)
    for (k, o) in enumerate(outidxs)
        Theta[k, o] += 1
    end
    return abs_reverse(t, Theta)[:, 1:t.ninputs]
end

"""
    variance_surrogate(t::Tape, outidxs::AbstractVector{<:Integer}) -> U2

The `m × n` squared-channel counterpart of [`jacobian_surrogate`](@ref): `U2[k, j]` sums, over the
paths from input `j` to output `k`, the squared product of the local partial magnitudes along the
path. A sum of squares is at most the square of the sum, so `U2 <= Utilde.^2` entrywise, with
equality where an entry has a single path. If each parent slot's local partial carries an
independent, equiprobable sign, `U2[k, j] = E[J_kj^2]`. This is the matrix
[`balance_cover`](@ref) takes with `merge = VarianceMerge()`. Formed by one `m`-channel reverse
sweep through squared partials, `O(m*L)`.
"""
function variance_surrogate(t::Tape{T}, outidxs::AbstractVector{<:Integer}) where {T}
    m, N = length(outidxs), nnodes(t)
    Theta = zeros(magtype(T), m, N)
    for (k, o) in enumerate(outidxs)
        Theta[k, o] += 1
    end
    return _reverse_sweep(_sqslot, t, Theta)[:, 1:t.ninputs]
end

"""
    JacobianCover{T}

Result of [`cover_jacobian`](@ref): `a[k] * b_sum[j] >= abs(J[k,j])`. `T` is
[`magtype`](@ref) of the element type of the [`Tape`](@ref) it was computed from, which for a
real floating-point tape is that element type itself.

- `a`: output weights.
- `b_sum`: input weights from the sum merge, computed in one `O(L)` reverse sweep.
- `roots`, `nties`: the root data of the tape-potential start (see [`tape_potentials`](@ref)), or
  `nothing` when another start or an explicit `a` was used.
- `m`, `n`: numbers of outputs and inputs.
"""
struct JacobianCover{T}
    a::Vector{T}
    b_sum::Vector{T}
    roots::Union{Nothing, Vector{Int}}
    nties::Union{Nothing, Vector{Int}}
    m::Int
    n::Int
end

"""
    JacobianDiagnostics{T}

The extended result returned by `cover_jacobian(WithDiagnostics(), ...)`. `T` is
[`magtype`](@ref) of the element type of the [`Tape`](@ref) it was computed from. It carries the
fields of [`JacobianCover`](@ref) together with

- `b_quad`: the quadrature merge `b_j = √(∑_k (Ũ_kj/a_k)²)`.
- `b_max`: the smallest valid `b` for this `a`, `b_j = max_k Ũ_kj/a_k`.
- `Utilde`: `m × n` absolute-sensitivity surrogate satisfying `Utilde >= abs.(J)`.

The merges satisfy `b_max <= b_quad <= b_sum` coordinatewise. Forming `Utilde` takes an
`m`-channel reverse sweep, `O(m * L)` time and `O(m * n)` memory.
"""
struct JacobianDiagnostics{T}
    a::Vector{T}
    b_sum::Vector{T}
    b_quad::Vector{T}
    b_max::Vector{T}
    Utilde::Matrix{T}
    roots::Union{Nothing, Vector{Int}}
    nties::Union{Nothing, Vector{Int}}
    m::Int
    n::Int
end

"""
    cover_jacobian([mode,] t::Tape, outidxs; start=:covariant, rounds=4, a=nothing, check=true)
    cover_jacobian([mode,] r, x0::AbstractVector; ...)

Asymmetric rank-1 cover of a vector-valued function's Jacobian: weights with
`a[k] * b[j] >= abs(J[k,j])`. Pass either a tape and its output nodes, or a function and an
evaluation point.

`a` is chosen by the balancing construction: `start` is `:covariant`, whose products on the
support transform exactly under a rescaling of the program's inputs, outputs and intermediates, or
`:equal`, which is permutation equivariant instead; `rounds` alternation rounds then refine it,
and `rounds = nothing` runs [`balance_cover`](@ref)'s alternation to convergence instead.
Passing `a` yourself fixes the weights and rules out `start` and `rounds`. `a` must be strictly
positive on every output some input reaches and may be zero on the others, which then contribute
to no merge.

# Modes

- `CoverOnly()` (the default) returns a [`JacobianCover`](@ref) from `O(L)` sweeps, never forming
  the surrogate. Checking a pair against the surrogate is what the other mode is for, so `check`
  has no effect here; a factor that vanishes on a row or column the program actually uses is an
  error either way.
- `WithDiagnostics()` returns a [`JacobianDiagnostics`](@ref): the same `a` and `b_sum`, plus the
  surrogate and the two tighter merges, at `O(m * L)`. `check` then verifies `b_max` against the
  surrogate.

# Examples

```jldoctest
julia> r(x) = [x[1] * x[2], exp(x[1]) - x[3], x[2] + 2x[3]];

julia> jc = cover_jacobian(WithDiagnostics(), r, [0.5, 2.0, -1.0]; a = ones(3));

julia> round.(jc.b_max; digits=4)
3-element Vector{Float64}:
 2.0
 1.0
 2.0

julia> all(jc.b_max .<= jc.b_quad .<= jc.b_sum)
true
```
"""
function cover_jacobian(
        ::CoverOnly, t::Tape{T}, outidxs::AbstractVector{<:Integer};
        start = nothing, rounds = missing, a = nothing, check::Bool = true
    ) where {T}
    # `_jacobian_weights` rejects a factor that vanishes on a live row or column whatever `check`
    # says, because such a pair is not a cover; `check` has nothing further the sweeps can see.
    w = _jacobian_weights(t, outidxs, start, rounds, a, _live_info(t, outidxs))
    return JacobianCover(w.a, w.b, w.roots, w.nties, length(outidxs), t.ninputs)
end

function cover_jacobian(
        ::WithDiagnostics, t::Tape{T}, outidxs::AbstractVector{<:Integer};
        start = nothing, rounds = missing, a = nothing, check::Bool = true
    ) where {T}
    m, n = length(outidxs), t.ninputs
    li = _live_info(t, outidxs)
    w = _jacobian_weights(t, outidxs, start, rounds, a, li)
    Utilde = jacobian_surrogate(t, outidxs)
    _check_surrogate_support(t, outidxs, Utilde, li)
    M = magtype(T)
    # scaled[k, j] = Ũ_kj / a_k, and zero on a dead output, whose row is zero anyway.
    scaled = [w.a[k] > 0 ? Utilde[k, j] / w.a[k] : zero(M) for k in 1:m, j in 1:n]
    b_quad = sqrt.(vec(sum(abs2, scaled; dims = 1)))
    b_max = vec(maximum(scaled; dims = 1))

    if check
        rtol = eps(M)^(2 // 3)
        s, idx = worst_severity(Utilde .* (1 - rtol) ./ (w.a .* b_max'))
        if s > 1
            k, j = Tuple(idx)
            error(
                "b_max does not cover the surrogate at output $k, input $j: " *
                    "a*b = $(w.a[k] * b_max[j]) < Ũ = $(Utilde[k, j])"
            )
        end
    end
    return JacobianDiagnostics(w.a, w.b, b_quad, b_max, Utilde, w.roots, w.nties, m, n)
end

function cover_jacobian(mode::CoverMode, r, x0::AbstractVector; kwargs...)
    t, outidxs = tape_function(r, x0)
    outidxs isa AbstractVector ||
        throw(ArgumentError("cover_jacobian needs a vector-valued program; got a scalar output"))
    return cover_jacobian(mode, t, outidxs; kwargs...)
end

cover_jacobian(t::Tape, outidxs::AbstractVector{<:Integer}; kwargs...) =
    cover_jacobian(CoverOnly(), t, outidxs; kwargs...)
cover_jacobian(r, x0::AbstractVector; kwargs...) =
    cover_jacobian(CoverOnly(), r, x0; kwargs...)

# Output weights and their sum merge, without forming the surrogate. Explicit weights replace the
# balancing construction outright, so naming a start or a round count alongside them asks for two
# incompatible things.
function _jacobian_weights(t::Tape, outidxs, start, rounds, a, li::LiveInfo)
    aw, b, roots, nties = _jacobian_a(t, outidxs, start, rounds, a, li)
    b === nothing && (b = _sum_merge_b(t, outidxs, aw, li))
    _check_cover_factors(aw, b, li)
    return (; a = aw, b, roots, nties)
end

# The output weights alone, together with whatever the balancing construction already computed
# alongside them. Every entry point that needs weights goes through here, so they all read the same
# keywords the same way.
function _jacobian_a(t::Tape{T}, outidxs, start, rounds, a, li::LiveInfo) where {T}
    if a === nothing
        bal = _balance_tape(t, outidxs, li; start, rounds, check = false)
        return bal.a, bal.b, bal.roots, bal.nties
    end
    start === nothing && rounds === missing ||
        throw(ArgumentError("a fixes the output weights, so start and rounds do not apply to it"))
    m = length(outidxs)
    # Weights use output positions 1:m, independent of the axes of outidxs.
    Base.require_one_based_indexing(a)
    length(a) == m ||
        throw(DimensionMismatch("a has length $(length(a)), expected $m (one per output)"))
    # The weights are magnitudes, so they join the type the covers are computed in.
    aw = collect(magtype(T), a)
    _check_output_weights(aw, .!li.liverow)
    return aw, nothing, nothing, nothing
end

"""
    cover_jacobian_hybrid(t::Tape, outidxs, groups::AbstractVector{<:Integer}; kwargs...) -> (; a, b)
    cover_jacobian_hybrid(t::Tape, outidxs, ngroups::Integer; kwargs...) -> (; a, b)
    cover_jacobian_hybrid(r, x0::AbstractVector, groups | ngroups; kwargs...) -> (; a, b)

Compute a Jacobian cover by grouping outputs.

`groups[k]` assigns output `k` to a group. Absolute sensitivities are summed within each group,
then the group totals are combined with `max`. Alternatively, pass `ngroups` to assign outputs
cyclically.

Every grouping is valid. One group reproduces `b_sum`; one group per output reproduces `b_max`.
The cost is `O(G * L)` for `G = maximum(groups)`, without forming the full surrogate.

The output weights are [`cover_jacobian`](@ref)'s, chosen by the same keywords:

  - `start`: `:covariant` (the default) or `:equal`.
  - `rounds`: alternation rounds refining them, `4` by default.
  - `a`: fixed weights of your own instead, strictly positive on every output some input reaches
    and optionally zero on the others. It rules out `start` and `rounds`.
  - `check = true`: verify the returned `b` against the surrogate, at `O(m * L)`.

Returns the weights used and the grouped merge for them: `a[k] * b[j] >= abs(J[k,j])`.

# Examples

The two endpoints, on a program with three outputs:

```jldoctest
julia> r(x) = [x[1] * x[2], exp(x[1]) - x[3], x[2] + 2x[3]];

julia> t, outidxs = tape_function(r, [0.5, 2.0, -1.0]);

julia> round.(cover_jacobian_hybrid(t, outidxs, 1; a = ones(3)).b; digits=4)   # one group: b_sum
3-element Vector{Float64}:
 3.6487
 1.5
 3.0

julia> round.(cover_jacobian_hybrid(t, outidxs, 3; a = ones(3)).b; digits=4)   # per output: b_max
3-element Vector{Float64}:
 2.0
 1.0
 2.0
```
"""
function cover_jacobian_hybrid(
        t::Tape{T}, outidxs::AbstractVector{<:Integer},
        groups::AbstractVector{<:Integer}; start = nothing, rounds = missing,
        a = nothing, check::Bool = true
    ) where {T}
    m, n, N = length(outidxs), t.ninputs, nnodes(t)
    # groups uses output positions 1:m, independent of the axes of outidxs.
    Base.require_one_based_indexing(groups)
    length(groups) == m ||
        throw(DimensionMismatch("groups has length $(length(groups)), expected $m (one per output)"))
    all(>=(1), groups) || throw(ArgumentError("group indices must be positive"))
    li = _live_info(t, outidxs)
    a = first(_jacobian_a(t, outidxs, start, rounds, a, li))

    G = maximum(groups)
    Theta = zeros(magtype(T), G, N)
    for (k, o) in enumerate(outidxs)
        # A dead output seeds nothing: its sensitivities are zero whatever weight it carries.
        a[k] > 0 && (Theta[groups[k], o] += 1 / a[k])
    end
    Abar = abs_reverse(t, Theta)              # G × N, group column-sums of Ũ/a at the inputs
    b = [maximum(view(Abar, :, j)) for j in 1:n]
    _check_cover_factors(a, b, li)

    if check
        Utilde = jacobian_surrogate(t, outidxs)
        rtol = eps(float(real(eltype(Utilde))))^(2 // 3)
        s, idx = worst_severity(Utilde .* (1 - rtol) ./ (a .* b'))
        if s > 1
            k, j = Tuple(idx)
            error(
                "group-max cover does not cover the surrogate at output $k, input $j: " *
                    "a*b = $(a[k] * b[j]) < Ũ = $(Utilde[k, j])"
            )
        end
    end
    return (; a, b)
end

function cover_jacobian_hybrid(
        t::Tape, outidxs::AbstractVector{<:Integer}, ngroups::Integer; kwargs...
    )
    ngroups >= 1 || throw(ArgumentError("ngroups must be positive, got $ngroups"))
    m = length(outidxs)
    groups = [1 + (k - 1) % ngroups for k in 1:m]
    return cover_jacobian_hybrid(t, outidxs, groups; kwargs...)
end

function cover_jacobian_hybrid(r, x0::AbstractVector, groups; kwargs...)
    t, outidxs = tape_function(r, x0)
    outidxs isa AbstractVector ||
        throw(ArgumentError("cover_jacobian_hybrid needs a vector-valued program; got a scalar output"))
    return cover_jacobian_hybrid(t, outidxs, groups; kwargs...)
end

# A returned pair must cover every structurally live entry, so neither factor may vanish there.
# Only underflow can put a zero on a live row or column, and a cover with one is not a cover, so
# this runs on every path rather than only under `check`. `O(m + n)`.
function _check_cover_factors(a, b, li::LiveInfo)
    for k in eachindex(a)
        (isfinite(a[k]) && (a[k] > 0 || !li.liverow[k])) || throw(
            ArgumentError(
                "output weight a[$k] is $(a[k]) on an output that inputs reach: the products " *
                    "along those paths underflowed, so no input factor can cover its row"
            )
        )
    end
    for j in eachindex(b)
        (isfinite(b[j]) && (b[j] > 0 || !li.livecol[j])) || throw(
            ArgumentError(
                "input weight b[$j] is $(b[j]) on an input that reaches an output: the products " *
                    "along those paths underflowed, so no output factor can cover its column"
            )
        )
    end
    return nothing
end

# Output weights are finite and strictly positive wherever an input reaches the output. A zero
# weight is admitted only on a dead output, where it covers that row exactly; anything else —
# negative, NaN, infinite, or zero on a live output — cannot cover and is an error.
function _check_output_weights(a, deadrow)
    for k in eachindex(a, deadrow)
        isfinite(a[k]) ||
            throw(ArgumentError("output weights a must be finite; a[$k] = $(a[k])"))
        a[k] > 0 || deadrow[k] && a[k] == 0 || throw(
            ArgumentError(
                "output weights a must be strictly positive on every output an input reaches, " *
                    "and may be zero only on the others; a[$k] = $(a[k])"
            )
        )
    end
    return a
end

# The surrogate's zeros must agree with the tape's structural pattern. They can disagree only by
# underflow: every local derivative on some input-to-output path is nonzero, yet their product is
# too small to represent. That is a floating-point event the caller needs to know about, not a
# structurally dead entry to be covered vacuously. Entrywise, from `cld(n, 64)` integer passes and
# `m * cld(n, 64)` words — small beside the `m * n` surrogate already in hand.
function _check_surrogate_support(t::Tape, outidxs, Utilde, li::LiveInfo)
    P = _structural_pattern(t, outidxs, li)
    n = t.ninputs
    for j in 1:n, k in 1:length(outidxs)
        blk, bit = cld(j, 64), (j - 1) % 64
        (P[k, blk] >> bit) & UInt64(1) == 1 || continue
        iszero(Utilde[k, j]) && throw(
            ArgumentError(
                "output $k reaches input $j along nonzero local derivatives, but the surrogate " *
                    "entry underflowed to zero: the products along those paths are below the " *
                    "smallest representable $(eltype(Utilde))"
            )
        )
    end
    return Utilde
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
    return _forward_sweep(_absslot, t, theta)
end

# Forward sweep with a pluggable per-slot weight, the transpose of [`_reverse_sweep`](@ref).
function _forward_sweep(w::F, t::Tape, theta::AbstractVector) where {F}
    s = collect(float.(theta))
    for k in (t.ninputs + 1):nnodes(t)
        acc = s[k]
        for (i, p) in enumerate(t.parents[k])
            a = w(t.ds[k][i], k, i, p)
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
forming `Ũ`. A zero `b_j` is accepted only for an input no output depends on: it contributes
nothing to any marginal. A zero `b_j` on an input that does reach an output is an error — no output
weighting can restore validity there.
"""
marginal_weights(t::Tape, outidxs::AbstractVector{<:Integer}, b::AbstractVector) =
    _marginal_weights(t, outidxs, b, _live_info(t, outidxs))

# The balancing loop already holds the structural liveness, so it passes it in rather than paying
# for a reachability sweep in every round that has a zero `b`. `slot = _sqslot` gives the
# squared-channel marginal `∑_j Ũ⁽²⁾_kj / b_j` instead.
function _marginal_weights(
        t::Tape{T}, outidxs, b::AbstractVector, li::LiveInfo, slot::F = _absslot
    ) where {T, F}
    length(b) == t.ninputs ||
        throw(DimensionMismatch("b has length $(length(b)), expected $(t.ninputs) (one per input)"))
    all(>=(0), b) || throw(ArgumentError("input cover b must be nonnegative"))
    theta = zeros(magtype(T), nnodes(t))
    for j in 1:t.ninputs
        if b[j] > 0
            theta[j] = 1 / b[j]
        else
            # A zero input weight is valid only for an input that reaches no output.
            li.livecol[j] && throw(
                ArgumentError(
                    "b is zero on input $j, which reaches an output; no output weighting can cover it"
                )
            )
        end
    end
    s = _forward_sweep(slot, t, theta)
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
    a = zeros(_ratio_type(Utilde, b), m)
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

[`VarianceMerge`](@ref) is the exception: it selects the squared channel for the whole
alternation, not only the final b-step, and the pair it returns is a scale rather than a cover.

`arule`, the other strategy axis of `balance_cover`, is a `Symbol` instead, because it is closed:
see that function's docstring.
"""
abstract type MergeRule end

"""
    SumMerge <: MergeRule

`b_j = ∑_k Ũ_kj/a_k`, the loosest merge and, with [`VarianceMerge`](@ref), the only one the tape
form can run as `O(L)` sweeps.
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
alternation can settle into a *limit cycle*: the weights keep changing by a fixed amount every
round, which no larger budget reduces. [`balance_cover`](@ref) reports that as a stall.
"""
struct MaxMerge <: MergeRule end

"""
    VarianceMerge <: MergeRule

The variance pair: [`balance_cover`](@ref)'s marginal alternation run on the squared channel
`Ũ⁽²⁾` ([`variance_surrogate`](@ref)) with squared factors,

    a_k² ← (1/ρ_k) ∑_j Ũ⁽²⁾_kj / b_j²,    b_j² ← (1/γ_j) ∑_k Ũ⁽²⁾_kj / a_k²,

followed by the final b-step `b_j² = ∑_k Ũ⁽²⁾_kj / a_k²`. This is the [`SumMerge`](@ref)
alternation applied to `Ũ⁽²⁾` with factors `(a², b²)`, with the same targets, round budget and
convergence test; the start is the default pair's start, squared, and `balance_cover` returns the
square roots. On a tape each half-step is one sweep through squared local partials, so a round
costs `O(L)` and `Ũ⁽²⁾` is never formed; the final b-step is
[`gramvariance`](@ref)`(t, outidxs, w)` with `w_k = 1/a_k²` on the live outputs and zero on the
others.

The pair is a scale, **not a cover**. The final b-step gives every live column
`∑_k Ũ⁽²⁾_kj / (a_k² b_j²) = 1`, so `a_k² b_j² >= Ũ⁽²⁾_kj`, which is what `check = true` verifies.
But `Ũ⁽²⁾ <= Ũ.^2`, with equality only where an entry has a single path, so `a_k b_j` can fall
below `Ũ_kj`, and below `abs(J_kj)` where several paths add with the same sign. If each parent
slot's local partial carries an independent, equiprobable sign, `Ũ⁽²⁾_kj = E[J_kj²]`, and at
convergence the pair standardizes the signed Jacobian in expectation: the normalized iterate
`(a, b ./ sqrt.(γ))` has expected row sums of squares
`∑_j E[J_kj²] / (a_k² b_j² / γ_j) = ρ_k` and column sums `γ_j`, and the returned `b` has
expected column sums of squares `1`.
"""
struct VarianceMerge <: MergeRule end

"""
    balance_cover(t::Tape, outidxs; kwargs...) -> (; a, b, iters, delta, roots, nties)
    balance_cover(r, x0::AbstractVector; kwargs...)
    balance_cover(Utilde::AbstractMatrix; kwargs...)

Output and input weights covering the absolute-sensitivity surrogate: `a[k] * b[j] >= Ũ_kj`, and
so `>= abs(J_kj)`, for every `merge` except [`VarianceMerge`](@ref).

By default the weights come from a scale-covariant start, `rounds` rounds of a marginal
alternation toward row and column targets the sparsity pattern makes realizable, and a final
[`MergeRule`](@ref) b-step for the `a` those rounds reached. The b-step is what makes the pair a
cover, so the result covers whatever the rounds did or did not settle. The accompanying manuscript
derives the weighting rules and their convergence; the keywords below say what each one does
without it.

# Keywords

  - `start = :covariant`: products on the support then transform exactly under a rescaling of the
    inputs and outputs — and, on the tape path, the intermediates — at every round count.
    `start = :equal` starts at `a = 1` instead, which is permutation equivariant at every round
    count but covariant only in the limit. Both reach the same fixed point.
  - `a0`: starting weights of your own, positive on every live output and zero on the others.
    Combining it with an explicit `start` is an error.
  - `targets`: `:flow` (tape default; `O(L)`; maximizes a flow-weighted mean log-tightness, which
    can be noticeably looser than `:support` when outputs differ widely in the number of paths
    reaching them), `:support` (matrix default;
    maximizes the geometric-mean tightness of the sum-merge cover; on a tape it costs
    `cld(n, 64) + cld(m, 64)` bit passes), an explicit `(r, c)`, or `:uniform` for unit row and
    column sums. `:uniform` needs the pattern to have *total support* — every nonzero entry on some
    positive diagonal — and without it the weights *drift*, separating without bound while every
    iterate still covers, so a longer run returns a looser cover. Applies to `arule = :marginal`
    only, and `:flow` needs a tape.
  - `rounds = 4`: run exactly this many rounds, with no convergence test and no error. `0` merges
    the start as it is, without computing `targets`, and reports `delta = Inf`. `nothing` runs to
    `delta <= rtol` instead and throws if `maxiter` rounds do not reach it; naming `maxiter` or
    `rtol` and no `rounds` asks for that too. Such a run overrelaxes the alternation toward
    `targets`, in log space with a factor adapted to the observed contraction, which reaches the
    same fixed point in far fewer rounds on slowly converging patterns; a fixed budget always runs
    the plain rounds.
  - `maxiter`, `rtol`: bound and tolerance of a `rounds = nothing` run. `maxiter = nothing` picks
    `1000` rounds for `arule = :marginal` and `10000` for `:geomean`. Passing `maxiter` together
    with a `rounds` count is an error.
  - `merge`: the final b-step, [`SumMerge`](@ref) (default), [`QuadMerge`](@ref) or
    [`MaxMerge`](@ref). [`VarianceMerge`](@ref) instead runs the whole alternation on the squared
    channel `Ũ⁽²⁾` and returns a scale that is not a cover; the matrix form then takes `Ũ⁽²⁾`
    ([`variance_surrogate`](@ref)) in place of `Ũ`, and `check` verifies `a_k² b_j² >= Ũ⁽²⁾_kj`.
  - `arule`: `:marginal` (default) alternates [`marginal_weights`](@ref) with the b-step;
    `:geomean` fits `log Ũ_kj ≈ log a_k + log b_j` instead and prescribes no `targets`.
  - `root`, `cgiters`: passed to [`tape_potentials`](@ref) for the tape-potential start. Naming
    them on a call that does not reach that start is an error.
  - `check = true`: verify the returned pair against the surrogate.

# Returns

`a` and `b` by output position and input, `iters` rounds run, `delta` the quantity the convergence
test applies to (the gauge-invariant product change `max_kj |log((a_k b_j)_new/(a_k b_j)_old)|`,
or the relative row-marginal residual `max_k |∑_j Ũ_kj/(a_k b_j r_k) - 1|` for non-uniform
targets), and `roots`, `nties` from [`tape_potentials`](@ref) when the tape-potential start was
used, `nothing` otherwise. A zero row of `Ũ` gets `a[k] = 0` and a zero column `b[j] = 0`. With
`VarianceMerge()`, `delta` is measured on the squared channel: on `a_k² b_j²` and `Ũ⁽²⁾`.

# Cost

The tape form with `merge = SumMerge()` or `VarianceMerge()` and `arule = :marginal` runs as
`O(L)` sweeps per round and never forms `Ũ`. Every other combination, and `check = true`, forms it
(or `Ũ⁽²⁾`) at `O(m*L)`.

# Examples

```jldoctest
julia> r(x) = [x[1] * x[2], exp(x[1]) - x[3], x[2] + 2x[3]];

julia> bal = DerivativeCovers.balance_cover(r, [0.5, 2.0, -1.0]);

julia> jc = cover_jacobian(WithDiagnostics(), r, [0.5, 2.0, -1.0]; a = bal.a);

julia> bal.iters, all(jc.b_max .<= bal.b)
(4, true)
```
"""
function balance_cover(
        t::Tape{T}, outidxs::AbstractVector{<:Integer}; merge::MergeRule = SumMerge(),
        arule::Symbol = :marginal, targets = nothing, start = nothing, a0 = nothing,
        root = nothing, cgiters::Integer = 0,
        maxiter::Union{Nothing, Integer} = nothing, rounds = missing,
        rtol::Union{Nothing, Real} = nothing, check::Bool = true
    ) where {T}
    return _balance_tape(
        t, outidxs, _live_info(t, outidxs);
        merge, arule, targets, start, a0, root, cgiters, maxiter, rounds, rtol, check
    )
end

# Everything the tape path needs, with structural liveness supplied by the caller so one public
# call computes it once. Targets, budget and start are resolved here and nowhere else: the
# materialized branch hands `_balance_matrix` the resolved values and the structural pattern, so
# nothing is re-derived from the numeric zeros of `Ũ`.
function _balance_tape(
        t::Tape{T}, outidxs::AbstractVector{<:Integer}, li::LiveInfo;
        merge::MergeRule = SumMerge(), arule::Symbol = :marginal, targets = nothing,
        start = nothing, a0 = nothing, root = nothing, cgiters::Integer = 0,
        maxiter::Union{Nothing, Integer} = nothing, rounds = missing,
        rtol::Union{Nothing, Real} = nothing, check::Bool = true
    ) where {T}
    m = length(outidxs)
    M = magtype(T)
    maxiter, rounds, rtol = _resolve_budget(maxiter, rounds, rtol, arule, M)
    # No round reads the targets, so a zero budget validates them without sweeping for them.
    tgt = _tape_targets(targets, arule, t, outidxs, m, M, li; needed = rounds !== 0)
    variance = merge isa VarianceMerge
    matrixpath = !(merge isa SumMerge || variance) || arule !== :marginal
    _check_start(start, a0, root, cgiters, !matrixpath)
    # The surrogate the pair is balanced against: `Ũ`, or `Ũ⁽²⁾` for the variance pair.
    surrogate = variance ? variance_surrogate : jacobian_surrogate
    if matrixpath
        # Other strategies need separate output channels; materialize the surrogate once. Its zeros
        # must match the structural pattern entrywise before anything reads them.
        Utilde = surrogate(t, outidxs)
        _check_surrogate_support(t, outidxs, Utilde, li)
        return _balance_matrix(
            Utilde, tgt, li.liverow, li.livecol, merge, arule, start, a0,
            maxiter, rounds, rtol, check, :flow
        )
    end

    roots, nties = nothing, nothing
    if a0 === nothing && something(start, :covariant) === :covariant
        a0, roots, nties = _potential_start(t, outidxs, root, cgiters, li)
    end
    a = _balance_start(a0, m, M, li.liverow)
    # The variance pair runs the sum-merge alternation on squared factors through squared partials.
    variance && (a = _squared_start(a))
    slot = variance ? _sqslot : _absslot

    rawbstep = a -> _sum_merge_b(t, outidxs, a, li, slot)
    anumstep = b -> _marginal_weights(t, outidxs, b, li, slot)
    if tgt === nothing
        a, b, iters, delta =
            _balance(rawbstep, anumstep, a, maxiter, rtol, _marginal_hint(SumMerge(), :flow); rounds)
    else
        a, iters, delta = _balance_targets(anumstep, rawbstep, a, tgt..., maxiter, rtol, rounds)
        # The normalized iterate is not a cover; the plain sum merge for this `a` is.
        b = rawbstep(a)
    end
    _check_cover_factors(a, b, li)
    if check
        Utilde = surrogate(t, outidxs)
        _check_surrogate_support(t, outidxs, Utilde, li)
        rworst = jacobian_ratio(Utilde, a, b)
        rworst <= 1 + eps(float(real(eltype(Utilde))))^(2 // 3) ||
            error("balanced pair does not cover the surrogate: worst ratio $rworst")
    end
    variance && (a = sqrt.(a); b = sqrt.(b))
    return (; a, b, iters, delta, roots, nties)
end

# Squared starting weights for the variance pair, which must be representable in their own right.
function _squared_start(a)
    a2 = a .^ 2
    for k in eachindex(a, a2)
        a[k] > 0 || continue
        (isfinite(a2[k]) && a2[k] > 0) || throw(
            ArgumentError(
                "the variance pair's squared start is not representable: a[$k]^2 = $(a2[k]) " *
                    "from a[$k] = $(a[k]). Use start=:equal, or supply a0."
            )
        )
    end
    return a2
end

function balance_cover(r, x0::AbstractVector; kwargs...)
    t, outidxs = tape_function(r, x0)
    outidxs isa AbstractVector ||
        throw(ArgumentError("balance_cover needs a vector-valued program; got a scalar output"))
    return balance_cover(t, outidxs; kwargs...)
end

function balance_cover(
        Utilde::AbstractMatrix; merge::MergeRule = SumMerge(), arule::Symbol = :marginal,
        targets = nothing, start = nothing, a0 = nothing, root = nothing,
        cgiters::Integer = 0, maxiter::Union{Nothing, Integer} = nothing,
        rounds = missing, rtol::Union{Nothing, Real} = nothing, check::Bool = true
    )
    Base.require_one_based_indexing(Utilde)
    all(>=(0), Utilde) || throw(ArgumentError("the magnitude matrix must be nonnegative"))
    T = float(magtype(eltype(Utilde)))
    maxiter, rounds, rtol = _resolve_budget(maxiter, rounds, rtol, arule, T)
    m, n = size(Utilde)
    tgt = _matrix_targets(targets, arule, Utilde, m, n, T)
    _check_start(start, a0, root, cgiters, false)
    liverow = BitVector(any(!iszero, view(Utilde, k, :)) for k in 1:m)
    livecol = BitVector(any(!iszero, view(Utilde, :, j)) for j in 1:n)
    return _balance_matrix(
        Utilde, tgt, liverow, livecol, merge, arule, start, a0,
        maxiter, rounds, rtol, check, :support
    )
end

# The balancing itself, on a materialized magnitude matrix. `liverow`/`livecol` say which rows and
# columns must end up with a positive factor, and `suggest` names the `targets` value the call form
# that got here accepts.
function _balance_matrix(
        Utilde, tgt, liverow, livecol, merge::MergeRule, arule::Symbol, start, a0,
        maxiter, rounds, rtol, check::Bool, suggest::Symbol
    )
    T = float(magtype(eltype(Utilde)))
    m = size(Utilde, 1)
    # No round reads the targets, so a zero budget goes straight from the start to the merge.
    rounds === 0 && (tgt = nothing)
    # The variance pair is the sum-merge balancing of `Ũ⁽²⁾` with squared factors. The covariant
    # start computed from `Ũ⁽²⁾` is already squared; one supplied by the caller is not.
    variance = merge isa VarianceMerge
    variance && (merge = SumMerge())
    if a0 === nothing && something(start, :covariant) === :covariant
        a0 = _surrogate_start(Utilde, T)
        a = _balance_start(a0, m, T, liverow)
    else
        a = _balance_start(a0, m, T, liverow)
        variance && (a = _squared_start(a))
    end
    bstep = a -> _merge_bstep(Utilde, a, merge)

    if arule === :marginal && tgt !== nothing
        rtgt, ctgt = tgt
        anumstep = b -> marginal_weights(Utilde, b)
        bnumstep = a -> _merge_bstep(Utilde, a, SumMerge())
        a, iters, delta =
            _balance_targets(anumstep, bnumstep, a, rtgt, ctgt, maxiter, rtol, rounds)
        # The normalized iterate is not a cover; the `merge` b-step for this `a` is.
        b = bstep(a)
    elseif arule === :marginal
        astep = b -> marginal_weights(Utilde, b)
        a, b, iters, delta =
            _balance(bstep, astep, a, maxiter, rtol, _marginal_hint(merge, suggest); rounds)
    elseif arule === :geomean
        a, iters, delta = _geomean_fit(Utilde, a, maxiter, rtol, rounds)
        b = bstep(a)
    else
        throw(ArgumentError("arule must be :marginal or :geomean; got $(repr(arule))"))
    end

    for k in eachindex(a)
        (isfinite(a[k]) && (a[k] > 0 || !liverow[k])) ||
            throw(ArgumentError("output weight a[$k] is $(a[k]) on a row with nonzero entries"))
    end
    for j in eachindex(b)
        (isfinite(b[j]) && (b[j] > 0 || !livecol[j])) ||
            throw(ArgumentError("input weight b[$j] is $(b[j]) on a column with nonzero entries"))
    end
    if check
        rworst = jacobian_ratio(Utilde, a, b)
        rworst <= 1 + eps(float(real(eltype(Utilde))))^(2 // 3) ||
            error("balanced pair does not cover the matrix: worst ratio $rworst")
    end
    variance && (a = sqrt.(a); b = sqrt.(b))
    # The tape-potential start is the only one with a root, so the other paths report none.
    return (; a, b, iters, delta, roots = nothing, nties = nothing)
end

# `b_j = ∑_k Ũ_kj/a_k` from one reverse sweep seeded `1/a_k` at the output nodes, never forming
# the surrogate. A zero weight on a live output cannot be covered by any `b`, so it is an error
# rather than a silently dropped row. `slot = _sqslot` gives `∑_k Ũ⁽²⁾_kj/a_k` instead.
function _sum_merge_b(t::Tape{T}, outidxs, a, li::LiveInfo, slot::F = _absslot) where {T, F}
    for k in eachindex(a)
        a[k] > 0 || !li.liverow[k] || throw(
            ArgumentError(
                "output weight a[$k] is $(a[k]) on an output that inputs reach; no input " *
                    "cover can cover its row"
            )
        )
    end
    seeds = (a[k] > 0 ? inv(a[k]) : zero(a[k]) for k in eachindex(a))
    return _output_sweep(t, outidxs, seeds, slot)
end

# `∑_k seeds[k] * Ũ_kj` at every input `j`, from one absolute-value reverse sweep seeded at the
# output nodes and never forming the surrogate. `seeds` pairs with `outidxs` in iteration order,
# and the seeds of a repeated output node add, as its repeated rows of `Ũ` do.
_output_sweep(t::Tape, outidxs, seeds, slot::F = _absslot) where {F} =
    _reverse_sweep(slot, t, _output_seeds(t, outidxs, seeds))[1:t.ninputs]

# Per-node seed vector with `seeds[k]` placed at output node `outidxs[k]`; repeated nodes add.
function _output_seeds(t::Tape{T}, outidxs, seeds) where {T}
    theta = zeros(magtype(T), nnodes(t))
    for (o, sk) in zip(outidxs, seeds)
        theta[o] += sk
    end
    return theta
end

# Merge-specific accumulation methods keep branches out of the inner loop.
_merge_accum(::SumMerge, bj, tkj) = bj + tkj
_merge_accum(::QuadMerge, bj, tkj) = bj + tkj^2
_merge_accum(::MaxMerge, bj, tkj) = max(bj, tkj)

_merge_finalize!(::MergeRule, b) = b
_merge_finalize!(::QuadMerge, b) = (b .= sqrt.(b); b)

function _merge_bstep(U, a, merge::MergeRule)
    m, n = size(U)
    b = zeros(_ratio_type(U, a), n)
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

# `rounds` is a fixed budget and `maxiter` bounds a run to convergence, so exactly one of them
# governs a call. `rounds === missing` means the caller said nothing about it: tuning a convergence
# run with `maxiter` or `rtol` then asks for one, and otherwise the default budget applies.
function _resolve_budget(maxiter, rounds, rtol, arule::Symbol, ::Type{T}) where {T}
    rtolv = rtol === nothing ? eps(float(magtype(T)))^(2 // 3) : rtol
    rounds === missing &&
        (rounds = (maxiter === nothing && rtol === nothing) ? _DEFAULT_ROUNDS : nothing)
    rounds === nothing && return _resolve_maxiter(maxiter, arule), nothing, rtolv
    for (name, v) in (("maxiter", maxiter), ("rtol", rtol))
        v === nothing || throw(
            ArgumentError(
                "$name tunes a run to convergence and rounds is a fixed budget, so they cannot " *
                    "be combined; got $name=$v with rounds=$rounds (pass rounds=nothing to converge)"
            )
        )
    end
    rounds >= 0 || throw(ArgumentError("rounds must be nonnegative, got $rounds"))
    return 0, Int(rounds), rtolv
end

const _DEFAULT_ROUNDS = 4

# Non-convergence hint for the marginal alternation, specific to how each merge can fail.
# `suggest` is the pattern-feasible `targets` value of the call form that produced the failure.
_marginal_hint(::MergeRule, suggest::Symbol) =
    "; the marginal alternation with uniform targets requires the surrogate's sparsity pattern to have total support — targets=$(repr(suggest)) or arule=:geomean converges for every pattern"
_marginal_hint(::MaxMerge, suggest::Symbol) =
    "; the max merge's marginal alternation can enter a limit cycle — targets=$(repr(suggest)) or arule=:geomean converges for every pattern"

# `potentialstart` says whether this call form reaches the tape-potential start, the only one the
# root keywords mean anything for.
function _check_start(start, a0, root, cgiters::Integer, potentialstart::Bool)
    start === nothing || start === :covariant || start === :equal ||
        throw(ArgumentError("start must be :covariant or :equal; got $(repr(start))"))
    a0 === nothing || start === nothing ||
        throw(ArgumentError("a0 replaces the start, so it cannot be combined with start=$(repr(start))"))
    if !(potentialstart && a0 === nothing && something(start, :covariant) === :covariant)
        root === nothing && cgiters == 0 ||
            throw(
            ArgumentError(
                "root and cgiters belong to the tape-potential start, which this call does not " *
                    "reach: it needs the tape form with merge=SumMerge() or VarianceMerge(), " *
                    "arule=:marginal, start=:covariant and no a0"
            )
        )
    end
    cgiters >= 0 || throw(ArgumentError("cgiters must be nonnegative, got $cgiters"))
    return nothing
end

# Output weights `exp(θ)` from the tape potentials, shifted by a constant per component before
# exponentiating. The shift is a gauge change: it leaves every product `a_k b_j` within a component
# untouched. Centering on the midrange of each component's output potentials puts the largest and
# smallest weights the same distance from 1, which is the placement that survives the widest
# spread; a spread too wide even for that has no representable start at all. Dead outputs get zero.
function _potential_start(t::Tape{T}, outidxs, root, cgiters::Integer, li::LiveInfo) where {T}
    pot, comp = _tape_potentials(t, outidxs, li; root, cgiters)
    TL = magtype(T)
    ncomp = length(pot.roots)
    lo, hi = fill(TL(Inf), ncomp), fill(TL(-Inf), ncomp)
    for o in outidxs
        pot.live[o] || continue
        c = comp[o]
        lo[c] = min(lo[c], pot.theta[o])
        hi[c] = max(hi[c], pot.theta[o])
    end
    a0 = Vector{TL}(undef, length(outidxs))
    for (k, o) in enumerate(outidxs)
        if !pot.live[o]
            a0[k] = zero(TL)
            continue
        end
        c = comp[o]
        a0[k] = exp(pot.theta[o] - (hi[c] + lo[c]) / 2)
        (isfinite(a0[k]) && a0[k] > 0) || throw(
            ArgumentError(
                "the tape-potential start has no representable form: component $c spans " *
                    "$(hi[c] - lo[c]) in log-scale, wider than $(TL) can hold even centered. " *
                    "Use start=:equal, or supply a0."
            )
        )
    end
    return a0, pot.roots, pot.nties
end

# Output weights from MatrixCovers' scale-covariant asymmetric start on the surrogate. That
# function returns a full pair; only the a-side is a start here, and the b-step supplies the rest.
# `feasible = :none` keeps it a start rather than inflating it into a cover, which the b-step would
# discard anyway. Unsupported rows come back zero, which is the convention used here as well.
function _surrogate_start(Utilde, ::Type{T}) where {T}
    a0, b0 = initialize_cover(Utilde; strategy = :covariant, feasible = :none)
    # That routine balances the two sides against each other, so both must be representable for
    # the pair it returns to mean anything.
    (all(isfinite, a0) && all(isfinite, b0)) || throw(
        ArgumentError(
            "the covariant start has no representable form on this matrix: its log-scales span " *
                "more than $(T) can hold. Use start=:equal, or supply a0."
        )
    )
    return collect(T, a0)
end

const _TARGET_HINT =
    "; these targets are realizable on this pattern, so this is slow convergence rather than " *
    "drift — raise maxiter, take a fixed `rounds` budget, or use arule=:geomean"

# Divide a marginal by its target, leaving a dead row or column at zero.
function _divide_targets(v, tgt)
    out = similar(v)
    for i in eachindex(v, tgt)
        out[i] = tgt[i] > 0 ? v[i] / tgt[i] : zero(eltype(out))
    end
    return out
end

# Resolve `targets` for the tape form: `nothing` for `:uniform`, else the validated pair `(r, c)`.
# The keyword's own `nothing` means "unspecified", which is `:flow` here and no targets at all for
# the geometric-mean fit, which has none to prescribe.
function _tape_targets(
        targets, arule::Symbol, t::Tape, outidxs, m::Int, ::Type{T}, li::LiveInfo;
        needed::Bool = true
    ) where {T}
    arule === :marginal || return _reject_targets(targets, arule)
    targets === nothing && (targets = :flow)
    targets === :uniform && return nothing
    if targets === :flow || targets === :support
        needed || return nothing
        r, c = targets === :flow ? _flow_targets(t, outidxs, li) : _support_counts(t, outidxs, li)
        return (collect(T, r), collect(T, c))
    end
    targets isa Tuple{AbstractVector, AbstractVector} ||
        throw(ArgumentError("targets must be :flow, :support, :uniform, or a tuple (r, c) of marginals; got $(repr(targets))"))
    tgt = _check_targets(targets[1], targets[2], li.liverow, li.livecol, T)
    return needed ? tgt : nothing
end

# Resolve `targets` for the matrix form. `:support` counts the nonzero entries of each row and
# column — the marginals of the support indicator, which has the pattern of `Utilde`.
function _matrix_targets(targets, arule::Symbol, U, m::Int, n::Int, ::Type{T}) where {T}
    arule === :marginal || return _reject_targets(targets, arule)
    targets === nothing && (targets = :support)
    targets === :flow &&
        throw(ArgumentError("targets=:flow needs a tape: it reads the slot graph, which a magnitude matrix does not carry. Use :support"))
    targets === :uniform && return nothing
    rcount, ccount = zeros(T, m), zeros(T, n)
    for j in 1:n, k in 1:m
        U[k, j] == 0 && continue
        rcount[k] += 1
        ccount[j] += 1
    end
    targets === :support && return (rcount, ccount)
    targets isa Tuple{AbstractVector, AbstractVector} ||
        throw(ArgumentError("targets must be :support, :uniform, or a tuple (r, c) of marginals; got $(repr(targets))"))
    return _check_targets(targets[1], targets[2], rcount .> 0, ccount .> 0, T)
end

# Only the marginal alternation normalizes to prescribed marginals; the geometric-mean fit has no
# targets, so naming any is a mistake rather than something to ignore.
function _reject_targets(targets, arule::Symbol)
    targets === nothing ||
        throw(ArgumentError("targets applies to the marginal alternation; arule=$(repr(arule)) prescribes no marginals"))
    return nothing
end

# Explicit targets must be realizable on the pattern they are used with: positive exactly where a
# row or column carries an entry, and with equal totals (the whole-row-set instance of the
# feasibility condition).
function _check_targets(r, c, liverow, livecol, ::Type{T}) where {T}
    Base.require_one_based_indexing(r, c)
    length(r) == length(liverow) ||
        throw(DimensionMismatch("target r has length $(length(r)), expected $(length(liverow)) (one per output)"))
    length(c) == length(livecol) ||
        throw(DimensionMismatch("target c has length $(length(c)), expected $(length(livecol)) (one per input)"))
    for (name, v, live) in (("r", r, liverow), ("c", c, livecol))
        for i in eachindex(v, live)
            if live[i]
                v[i] > 0 ||
                    throw(ArgumentError("target $name[$i] = $(v[i]) is not positive on a live $(name == "r" ? "output" : "input")"))
            else
                v[i] == 0 ||
                    throw(ArgumentError("target $name[$i] = $(v[i]) is nonzero on a dead $(name == "r" ? "output" : "input"); its weight stays zero, so its target must too"))
            end
        end
    end
    sr, sc = sum(r), sum(c)
    # The totals are commensurate and dimensionless, so a relative comparison is the right one.
    abs(sr - sc) <= sqrt(eps(T)) * max(sr, sc) ||
        throw(ArgumentError("target totals must agree: sum(r) = $sr, sum(c) = $sc"))
    return (collect(T, r), collect(T, c))
end

# Coordinate descent for log U_kj ≈ log a_k + log b_j over nonzero entries.
function _geomean_fit(U, a, maxiter, rtol, rounds)
    m, n = size(U)
    gstep = (weights, dim) -> begin
        out = zeros(_ratio_type(U, a), size(U, dim))
        cnt = zeros(Int, size(U, dim))
        for j in 1:n, k in 1:m
            U[k, j] == 0 && continue
            i = dim == 1 ? k : j
            w = dim == 1 ? weights[j] : weights[k]
            out[i] += log(U[k, j]) - log(w)
            cnt[i] += 1
        end
        [cnt[i] > 0 ? exp(out[i] / cnt[i]) : zero(eltype(out)) for i in eachindex(out)]
    end
    b = gstep(a, 2)
    _regauge!(a, b)
    delta = Inf
    # A fixed budget stops early: every iterate of this fit is as covariant as the start, so what
    # the remaining rounds would buy is accuracy alone.
    budget = rounds === nothing ? maxiter : rounds
    for iters in 1:budget
        aprev, bprev = a, b
        a = gstep(b, 1)
        b = gstep(a, 2)
        _regauge!(a, b)
        ulo, uhi = _logchange_extrema(a, aprev)
        wlo, whi = _logchange_extrema(b, bprev)
        delta = max(uhi + whi, -(ulo + wlo))
        rounds === nothing && delta <= rtol && return a, iters, delta
    end
    rounds === nothing || return a, rounds, delta
    error(
        "balance_cover(arule=:geomean) did not converge in $maxiter rounds: product still " *
            "changing by delta = $delta > rtol = $rtol. The geometric-mean fit converges for every " *
            "pattern but slowly on ill-conditioned supports; raise maxiter (some sparse patterns " *
            "need 1e4–1e5 rounds)."
    )
end

# `liverow` marks the outputs some input reaches. A dead output's weight is zero throughout, so
# its start must be too; every live one must start positive.
function _balance_start(a0, m, ::Type{T}, liverow) where {T}
    if a0 === nothing
        return T[liverow[k] ? one(T) : zero(T) for k in 1:m]
    end
    length(a0) == m ||
        throw(DimensionMismatch("a0 has length $(length(a0)), expected $m (one per output)"))
    for k in eachindex(a0, liverow)
        ok = isfinite(a0[k]) && (liverow[k] ? a0[k] > 0 : a0[k] == 0)
        ok || throw(
            ArgumentError(
                "starting weights a0 must be strictly positive on every live output and zero " *
                    "on the rest; a0[$k] = $(a0[k]) on a $(liverow[k] ? "live" : "dead") output"
            )
        )
    end
    return collect(T, a0)
end

# Shared balancing loop. Renormalizing by a power of two preserves every product exactly in
# floating point. Convergence is measured on products, making `delta` invariant to the gauge
# transformation `(a, b) -> (γa, b/γ)`.
function _balance(bstep, astep, a, maxiter, rtol, hint = ""; rounds = nothing)
    rounds === nothing || return _balance_budget(bstep, astep, a, rounds)
    maxiter >= 1 || throw(ArgumentError("maxiter must be at least 1, got $maxiter"))
    b = bstep(a)
    _regauge!(a, b)
    delta = Inf
    # Max-merge alternation can enter a limit cycle. A flat, positive delta identifies a stall;
    # increasing maxiter cannot resolve it.
    stall_window, stall_flat = 128, 1.0e-6
    last_checked = Inf
    # Divergence is measured on the log-spread within each side, which no gauge affects. Sampling
    # it at doubling round counts turns the two families apart: a converging run's spread settles,
    # so the growth between samples shrinks geometrically, while a diverging one grows by a
    # constant (logarithmic drift) or a doubling (geometric drift) every time. Requiring the growth
    # to fail to shrink twice running keeps a slowly-settling run from being mistaken for drift.
    spread_next, spread_prev, growth_prev, strikes = 64, nothing, nothing, 0
    spread_shrink, spread_floor, spread_strikes = 0.9, 1.0e-3, 2
    for iters in 1:maxiter
        aprev, bprev = a, b
        a = astep(b)
        b = bstep(a)
        _regauge!(a, b)
        ulo, uhi = _logchange_extrema(a, aprev)
        wlo, whi = _logchange_extrema(b, bprev)
        delta = max(uhi + whi, -(ulo + wlo))
        delta <= rtol && return a, b, iters, delta
        if iters == spread_next
            spread = _logspread(a) + _logspread(b)
            if spread_prev !== nothing
                growth = spread - spread_prev
                if growth_prev !== nothing && growth > spread_floor && growth >= spread_shrink * growth_prev
                    strikes += 1
                    strikes >= spread_strikes && error(
                        "balance_cover diverged after $iters rounds: the weights are separating " *
                            "without bound (log-spread $spread, growing by $growth per doubling of the " *
                            "round count) while the products keep moving. Every iterate covers, but the " *
                            "excess grows with the round count, so a longer run returns a worse cover$hint"
                    )
                else
                    strikes = 0
                end
                growth_prev = growth
            end
            spread_prev = spread
            spread_next *= 2
        end
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

# Alternation toward prescribed marginals. `anum(b)` returns `∑_j Ũ_kj/b_j` and `bnum(a)` returns
# `∑_k Ũ_kj/a_k`; dividing them by `r` and `c` gives the two normalized steps.
#
# Convergence is measured on the relative row-marginal residual
# `max_k |∑_j R_kj/r_k - 1|` with `R_kj = Ũ_kj/(a_k b_j)`, which is what the iteration is trying to
# drive to zero. The product increment is not a substitute: on slowly converging supports it falls
# below `rtol` while the marginals are still far off. The column residual carries no information —
# the b-step sets `∑_k R_kj = c_j` exactly — and the row residual is read from `anum(b)`, the sweep
# the next round's a-step needs, so testing it costs one extra `anum` over the whole run and
# nothing per round.
#
# A fixed budget runs the plain alternation. A run to convergence is overrelaxed unless `relax`
# is false (see `_balance_targets_relaxed`).
function _balance_targets(anum, bnum, a, r, c, maxiter, rtol, rounds; relax::Bool = true)
    b = _divide_targets(bnum(a), c)
    _regauge!(a, b)
    an = anum(b)
    delta = _row_residual(an, a, r)
    if rounds !== nothing
        rounds >= 0 || throw(ArgumentError("rounds must be nonnegative, got $rounds"))
        for _ in 1:rounds
            a = _divide_targets(an, r)
            b = _divide_targets(bnum(a), c)
            _regauge!(a, b)
            an = anum(b)
            delta = _row_residual(an, a, r)
        end
        return a, rounds, delta
    end
    maxiter >= 1 || throw(ArgumentError("maxiter must be at least 1, got $maxiter"))
    delta <= rtol && return a, 0, delta
    relax && return _balance_targets_relaxed(anum, bnum, a, b, an, r, c, maxiter, rtol)
    for iters in 1:maxiter
        a = _divide_targets(an, r)
        b = _divide_targets(bnum(a), c)
        _regauge!(a, b)
        an = anum(b)
        delta = _row_residual(an, a, r)
        delta <= rtol && return a, iters, delta
    end
    return _target_nonconvergence(maxiter, delta, rtol)
end

_target_nonconvergence(maxiter, delta, rtol) = error(
    "balance_cover did not converge in $maxiter rounds: the relative row-marginal residual " *
        "is still delta = $delta > rtol = $rtol$_TARGET_HINT"
)

# Overrelaxation of the targeted alternation. In the log-scales `x = log a`, `y = log b`, the
# alternation is block coordinate descent on the convex function
# `∑_kj Ũ_kj exp(-x_k - y_j) + ∑_k r_k x_k + ∑_j c_j y_j`, and its restriction to one `x_k` is
# `r_k ψ(x_k - F_k)` plus a constant, with `F_k` the plain step and `ψ(z) = exp(-z) - 1 + z`
# (likewise for `y_j`). The relaxed step moves to `F_k + (1 - w)(x_k - F_k)`, and is taken only if
# it leaves at most `(1 + (w - 1)²)/2` of the block's excess `ψ`; otherwise `w - 1` is halved a few
# times before falling back to the plain step. Every half-step thus removes a fixed fraction of
# what the plain one would, which keeps the iteration convergent for any `ω < 2`.
#
# `ω` is estimated as in MatrixCovers' power-mean solver: a few plain rounds measure the
# contraction `μ` of the imbalance (the worst relative marginal residual before each half-step),
# which sets `ω = 2/(1 + √(1 - μ))`, the optimum for a 2-cyclic iteration with plain rate `μ`.
# Relaxed rounds are then judged over windows by their smallest imbalance, since a relaxed
# iteration's imbalance is not monotone: a window's observed rate can raise `ω` toward the
# optimum it implies, and several windows without progress halve `ω - 1`, eventually back to plain
# rounds and a fresh estimate. The imbalance, its contraction and the steps are all ratios of
# marginals, so `ω` is invariant under rescaling and the iterates are as covariant as the plain
# alternation's, up to roundoff.
#
# Relaxation amplifies roundoff in the marginals by about `1/(2 - ω)`, so `ω` is capped where
# that amplification would reach `rtol`. Convergence is declared on the plain criterion, the row
# residual after an exact b-step, so the fixed point and `delta` mean what they mean without
# relaxation.
function _balance_targets_relaxed(anum, bnum, a, b, an, r, c, maxiter, rtol)
    T = float(real(eltype(a)))
    ωmax = _relax_omega_max(T, rtol)
    ω = one(T)
    nplain, r1, r2 = 0, T(Inf), T(Inf)          # plain rounds and the imbalance 1 and 2 back
    rbest, rwin, nwin, nstall = T(Inf), T(Inf), 0, 0
    delta = T(Inf)
    for iters in 1:maxiter
        ra = _relaxed_step!(a, _divide_targets(an, r), r, ω)
        rb = _relaxed_step!(b, _divide_targets(bnum(a), c), c, ω)
        _regauge!(a, b)
        imb = max(ra, rb)
        # Near convergence a relaxed round finishes with the plain b-step, so that the row
        # residual after it is the plain convergence test. A plain round took that step already.
        exact = ω == 1 || imb <= rtol
        if ω != 1 && exact
            b = _divide_targets(bnum(a), c)
            _regauge!(a, b)
        end
        an = anum(b)
        if exact
            delta = _row_residual(an, a, r)
            delta <= rtol && return a, iters, delta
        end
        if ω == 1
            nplain += 1
            if nplain >= _RELAX_NESTIMATE
                μ = sqrt(imb / r2)
                if zero(T) < μ < one(T)
                    ω = min(ωmax, 2 / (1 + sqrt(1 - μ)))
                    rbest, rwin, nwin, nstall = imb, T(Inf), 0, 0
                end
            end
            r2, r1 = r1, imb
        else
            nwin += 1
            rwin = min(rwin, imb)
            wlen = _relax_window(ω)
            nwin < wlen && continue
            λ = (rwin / rbest)^(one(T) / wlen)
            if !(λ < 1)
                nstall += 1
                if nstall >= _RELAX_NSTALL
                    nstall = 0
                    ω = 1 + (ω - 1) / 2
                    if ω - 1 < (ωmax - 1) / 16
                        ω, nplain, r1, r2 = one(T), 0, T(Inf), T(Inf)
                    end
                end
            else
                nstall = 0
                if λ > ω - 1
                    μ2 = (λ + ω - 1)^2 / (ω^2 * λ)
                    μ2 < 1 && (ω = max(ω, min(ωmax, 2 / (1 + sqrt(1 - μ2)))))
                end
            end
            rbest = min(rbest, rwin)
            rwin, nwin = T(Inf), 0
        end
    end
    return _target_nonconvergence(maxiter, delta, rtol)
end

# Plain rounds behind the first estimate of the contraction; the window, in rounds, over which a
# relaxed rate is measured, which lengthens as `ω → 2` because the rate per round is at best
# `ω - 1`; the windows without progress that halve `ω - 1`; the halvings a single relaxed step may
# try; and the largest relaxation factor.
const _RELAX_NESTIMATE = 4
const _RELAX_WINDOW = 16
const _RELAX_WINDOW_SCALE = 4.0
const _RELAX_NSTALL = 4
const _RELAX_NBACKTRACK = 4
const _RELAX_OMEGA_MAX = 1.995

_relax_window(ω) = _RELAX_WINDOW + round(Int, _RELAX_WINDOW_SCALE / (2 - ω))

# Largest `ω` whose roundoff amplification `1/(2 - ω)` keeps `64 eps` below `rtol`; `1`, no
# relaxation, when even that is too much.
_relax_omega_max(::Type{T}, rtol) where {T} =
    clamp(2 - 64 * eps(T) / T(rtol), one(T), T(_RELAX_OMEGA_MAX))

# `ψ(z) = exp(-z) - 1 + z`, the excess of one block's objective over its minimum, per unit target.
_relax_ψ(z) = expm1(-z) + z

# Move each live `x[i]` from its current value toward the plain step `F[i]`, overrelaxed by `ω` in
# log space and safeguarded as described at `_balance_targets_relaxed`. Entries with a zero target
# are dead and stay zero. Returns the imbalance `max |F[i]/x[i] - 1|` before the step.
function _relaxed_step!(x, F, tgt, ω)
    imb = zero(float(real(eltype(x))))
    for i in eachindex(x, F, tgt)
        tgt[i] > 0 || continue
        δ = log(x[i] / F[i])
        imb = max(imb, abs(expm1(-δ)))
        xi = F[i]
        if ω != 1
            ψδ = _relax_ψ(δ)
            w = ω
            for _ in 0:_RELAX_NBACKTRACK
                if _relax_ψ((1 - w) * δ) <= (1 + (w - 1)^2) / 2 * ψδ
                    xi = F[i] * exp((1 - w) * δ)
                    break
                end
                w = 1 + (w - 1) / 2
            end
        end
        x[i] = xi
    end
    return imb
end

# `∑_j R_kj / r_k - 1` in absolute value, worst over the live rows. `anum[k] = ∑_j Ũ_kj/b_j`, so
# `∑_j R_kj = anum[k]/a[k]`. Dead rows carry zero weight and zero target and are skipped.
function _row_residual(anum, a, r)
    worst = zero(float(real(eltype(anum))))
    for k in eachindex(anum, a, r)
        r[k] > 0 || continue
        a[k] > 0 ||
            throw(ArgumentError("a is zero on live output $k; no input cover can cover it"))
        worst = max(worst, abs(anum[k] / (a[k] * r[k]) - 1))
    end
    return worst
end

# Exactly `rounds` alternation rounds, with no convergence, drift or stall test: the caller has
# chosen a budget rather than a tolerance, and whatever the rounds reach is the answer.
function _balance_budget(bstep, astep, a, rounds)
    rounds >= 0 || throw(ArgumentError("rounds must be nonnegative, got $rounds"))
    b = bstep(a)
    _regauge!(a, b)
    delta = Inf
    for _ in 1:rounds
        aprev, bprev = a, b
        a = astep(b)
        b = bstep(a)
        _regauge!(a, b)
        ulo, uhi = _logchange_extrema(a, aprev)
        wlo, whi = _logchange_extrema(b, bprev)
        delta = max(uhi + whi, -(ulo + wlo))
    end
    return a, b, rounds, delta
end

# Log-ratio of the largest to the smallest positive weight. Zero weights mark dead outputs or
# inputs and are excluded: they are exact, and stay zero for the whole run.
function _logspread(v)
    lo, hi = Inf, -Inf
    for x in v
        x > 0 || continue
        l = log(x)
        lo = min(lo, l)
        hi = max(hi, l)
    end
    return lo <= hi ? hi - lo : zero(hi - hi)
end

function _regauge!(a, b)
    # `a` and `1 ./ b` both scale by the same factor under the gauge, so centering their combined
    # log range is the shift that keeps the most room on both sides at once.
    lo, hi = Inf, -Inf
    alo, ahi, blo, bhi = Inf, -Inf, Inf, -Inf
    for x in a
        x > 0 || continue
        l = log2(x)
        alo, ahi = min(alo, l), max(ahi, l)
        lo, hi = min(lo, l), max(hi, l)
    end
    for x in b
        x > 0 || continue
        l = -log2(x)
        blo, bhi = min(blo, -l), max(bhi, -l)
        lo, hi = min(lo, l), max(hi, l)
    end
    lo <= hi || return
    gamma = exp2(round((lo + hi) / 2))
    a ./= gamma
    b .*= gamma
    for (i, x) in pairs(a)
        (isfinite(x) && (x > 0 || x == 0)) || throw(
            ArgumentError(
                "no gauge keeps both factors representable: a spans $(ahi - alo) and b spans " *
                    "$(bhi - blo) in log2, and together they need more than $(eltype(a)) has"
            )
        )
    end
    for (j, x) in pairs(b)
        isfinite(x) || throw(
            ArgumentError(
                "no gauge keeps both factors representable: a spans $(ahi - alo) and b spans " *
                    "$(bhi - blo) in log2, and together they need more than $(eltype(b)) has"
            )
        )
    end
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
skipped; a nonzero entry over a vanishing `a[k]*b[j]` gives `Inf`. The asymmetric counterpart of
[`cover_ratio`](@ref).
"""
function jacobian_ratio(J::AbstractMatrix, a::AbstractVector, b::AbstractVector)
    axes(J, 1) == axes(a, 1) && axes(J, 2) == axes(b, 1) ||
        throw(DimensionMismatch("J is $(axes(J)); a is $(axes(a)); b is $(axes(b))"))
    worst = zero(_ratio_type(J, a))
    for j in axes(J, 2), k in axes(J, 1)
        Jkj = magnitude(J[k, j])
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
function jacobian_minimax(M::AbstractMatrix; atol::Real = eps(float(magtype(eltype(M))))^(3 // 4))
    Base.require_one_based_indexing(M)
    all(>=(0), M) || throw(ArgumentError("M must be nonnegative"))
    atol > 0 || throw(ArgumentError("atol must be positive, got $atol"))
    m, n = size(M)
    T = float(magtype(eltype(M)))
    ks, js, cs = Int[], Int[], T[]
    for j in 1:n, k in 1:m
        M[k, j] > 0 && (push!(ks, k); push!(js, j); push!(cs, log(M[k, j])))
    end
    isempty(cs) && return (; a = zeros(T, m), b = zeros(T, n), excess = one(T))

    lo, hi = zero(T), maximum(cs) - minimum(cs)
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
    a[.!liverow] .= 0
    b[.!livecol] .= 0
    _regauge!(a, b)
    excess = maximum(a[k] * b[j] / M[k, j] for (k, j) in zip(ks, js))
    excess >= 1 - eps(T)^(2 // 3) ||
        error("minimax pair does not cover M: worst ratio $(1 / excess)")
    return (; a, b, excess)
end

# The minimax constraints are feasible iff their difference-constraint graph has no negative
# cycle. Multi-source Bellman–Ford both tests feasibility and returns valid potentials.
function _minimax_feasible(m, n, ks, js, cs, E)
    pot = zeros(eltype(cs), m + n)
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
diagnostic. Pass [`magnitude`](@ref)`.(J)` (which is `abs.(J)` for a numeric `J`) or a surrogate
`Ũ` as `M`. This is the 2-cycle restriction of the
exact forced excess: longer support cycles can dominate it, and [`jacobian_minimax`](@ref)
computes the exact value together with a pair attaining it.
"""
function jacobian_representability(M::AbstractMatrix)
    Base.require_one_based_indexing(M)
    m, n = size(M)
    all(>=(0), M) || throw(ArgumentError("M must be nonnegative"))
    T = float(magtype(eltype(M)))
    worst = zero(T)
    for k in 1:m, l in (k + 1):m
        lo = T(Inf)
        hi = T(-Inf)
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
