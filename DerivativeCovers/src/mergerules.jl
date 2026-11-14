# How a construction combines the contributions that reach one input weight.

"""
    MergeRule

How [`balance_cover`](@ref)'s b-step combines a column of the scaled surrogate into one input
weight: given the entries `Ũ_kj/a_k` over the outputs `k`, it returns `b_j`. Every rule that
dominates each of its terms yields a valid cover, so the choice trades tightness, never validity.

[`SumMerge`](@ref), [`QuadMerge`](@ref) and [`MaxMerge`](@ref) are the `p = 1, 2, ∞` members of
one family, so a parameterized member (a `p`-norm merge) is reachable through the same b-step
without an API break. They are types rather than names so that the b-step's doubly-nested loop
specializes on the choice, and so that the non-convergence hint can dispatch per rule.

[`SignAveragedMerge`](@ref) is the exception: it runs the whole alternation, not only the final
b-step, in quadrature on `R`, and the pair it returns is a sign-averaged cover, which does not
cover `J`.

`arule`, the other strategy axis of `balance_cover`, is a `Symbol` instead, because it is closed:
see that function's docstring.
"""
abstract type MergeRule end

"""
    SumMerge <: MergeRule

`b_j = ∑_k Ũ_kj/a_k`, the loosest merge and, with [`SignAveragedMerge`](@ref), the only one the tape
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
    SignAveragedMerge <: MergeRule

The sign-averaged pair: [`balance_cover`](@ref)'s marginal iteration run in quadrature on
`R` ([`quadrature_surrogate`](@ref)),

    a_k ← sqrt((1/r_k) ∑_j R_kj² / b_j²),    b_j ← sqrt((1/c_j) ∑_k R_kj² / a_k²),

followed by the final b-step `b_j = sqrt(∑_k R_kj² / a_k²)`. Squared, this is the
[`SumMerge`](@ref) alternation applied to the squared channel `Ũ⁽²⁾ = R.^2` with factors
`(a², b²)`, the same targets and round budget, and the squared start. The iteration works with
`a`, `b` and `R` themselves and accumulates every sum of squares with `hypot`, so it never forms
the square of a Jacobian-scale quantity, and the pair is representable wherever the plain pair
is. On a tape each half-step is
one sweep combining local partials in quadrature, so a round costs `O(L)` and `R` is never
formed; the final b-step equals [`gramvariance`](@ref)`(t, outidxs, w)` with `w_k = 1/a_k²` on
the live outputs and zero on the others, where those squares are representable.

The pair is a sign-averaged cover, **not a cover of `J`**. The final b-step gives every live column
`∑_k R_kj² / (a_k b_j)² = 1`, so `a_k b_j >= R_kj`, which is what `check = true` verifies. But
`R <= Ũ`, with equality only where an entry has a single path, so `a_k b_j` can fall below `Ũ_kj`,
and below `abs(J_kj)` where several paths add with the same sign. If each parent slot's local
partial carries an independent, equiprobable sign, `R_kj² = E[J_kj²]`, and at convergence the pair
standardizes the signed Jacobian in expectation: the normalized iterate `(a, b ./ sqrt.(c))` has
expected row sums of squares `∑_j E[J_kj²] / (a_k² b_j² / c_j) = r_k` and column sums `c_j`, and
the returned `b` has expected column sums of squares `1`.
"""
struct SignAveragedMerge <: MergeRule end
