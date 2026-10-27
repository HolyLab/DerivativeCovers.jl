```@meta
CurrentModule = DerivativeCovers
```

# Covers over a box

A cover computed at a point bounds the Hessian at that point. Some uses, such
as global optimization and validated numerics, need a bound that holds over a
whole region. Loading [IntervalArithmetic](https://github.com/JuliaIntervals/IntervalArithmetic.jl)
allows the program to be recorded on a box of intervals; the covers computed
from that recording hold at every point of the box:

```jldoctest box
julia> using DerivativeCovers, IntervalArithmetic

julia> f(x) = exp(x[1]) + x[1] * x[2]^2;

julia> X = interval.([0.4, 1.0], [0.6, 1.5]);    # 0.4 ≤ x[1] ≤ 0.6, 1.0 ≤ x[2] ≤ 1.5

julia> c = cover_hessian(f, X);

julia> round.(c.h_sum; digits=4)
2-element Vector{Float64}:
 3.2864
 2.6446

julia> H(x) = [exp(x[1]) 2x[2]; 2x[2] 2x[1]];    # the exact Hessian, for comparison

julia> all(all(abs.(H([x1, x2])) .<= c.h_sum .* c.h_sum') for x1 in 0.4:0.05:0.6, x2 in 1.0:0.1:1.5)
true
```

The cover is a vector of floating-point numbers, while the gradient `c.grad` is
a vector of intervals enclosing the gradient over the box.

Every function that computes a cover accepts a box, including
[`forward_cover`](@ref), [`cover_nls`](@ref), and [`cover_jacobian`](@ref).
When the inputs are intervals, the arithmetic that accumulates the bounds
rounds upward (see [`Upward`](@ref) and [`boundtype`](@ref)), so the covers
hold exactly rather than up to rounding.

Some operations are not defined over a box:

- control flow that the box does not decide, such as a comparison of a
  variable's [`value`](@ref) when its interval contains the comparison value,
  raises IntervalArithmetic's `InconclusiveBooleanOperation`;
- an operation evaluated outside its domain on part of the box, such as `log`
  of an interval containing zero, raises a `DomainError` that identifies the
  operation;
- `abs` of an interval containing zero is accepted only as the base of a
  constant power `p ≥ 2`, since `abs` is not twice differentiable at zero;
- `atan(y, x)` refuses a box that touches its branch cut, `y = 0` with
  `x ≤ 0`, across which it is discontinuous;
- functions that evaluate the program at a point, such as
  [`scaled_spectral_radius`](@ref) with a tape, reject a tape recorded over a
  box.

Covers of a composition `F₂ ∘ F₁` can be formed from covers of the two pieces,
each computed over the appropriate region; see
[Combining covers of pieces](@ref).

## Supporting other set-valued types

The interval support is implemented by a small number of methods, which
another set-valued number type can also implement: [`magnitude`](@ref) (the
largest absolute value over the set), [`boundtype`](@ref),
[`decided_sign`](@ref), [`decided_one` and `decided_zero`](@ref decided_one),
[`separation`](@ref), and [`check_values`](@ref).
