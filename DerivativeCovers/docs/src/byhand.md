```@meta
CurrentModule = DerivativeCovers
```

# Writing a cover by hand

[`cover_hessian`](@ref) records a generic program and sweeps the recording.
The recording is convenient and general, but it stores every scalar operation,
so its cost can be many times that of a hand-written gradient. When one
program is evaluated many times, the same sweep can instead be written out for
that program, typically as array operations that add a few extra lines to the
gradient code. This page states the rules such code follows and shows how to
test it against `cover_hessian`, which serves as the reference implementation.

A hand-written cover reproduces the cover of one particular program. Two
programs that compute the same function generally have different covers (see
[Tightness](@ref)), so the hand-written code and the program it is tested
against must perform the same operations.

## The rules

A hand-written cover extends reverse-mode differentiation. The reverse pass
computes the adjoint ``\bar v`` of each intermediate value; alongside it, each
value receives one more accumulator, ``s`` for the sum cover or ``q`` for the
sign-averaged cover (see [Computing covers](@ref)). Both start at zero. Each
operation passes its accumulator back to its arguments and adds a *seed* for
its own curvature:

| Operation | adjoint | sum cover | sign-averaged cover |
|:--- |:--- |:--- |:--- |
| `y = A*x`, constant `A` | `x̄ += A'*ȳ` | `s_x += abs.(A)'*s_y` | `q_x += abs2.(A)'*q_y` |
| `y = φ(x)` | `x̄ += φ′*ȳ` | `s_x += abs(φ′)*s_y + sqrt(abs(ȳ*φ″))` | `q_x += φ′^2*q_y + abs(ȳ*φ″)` |
| `z = u*v` | `ū += v*z̄` | `s_u += abs(v)*s_z + sqrt(abs(z̄)*t)` | `q_u += v^2*q_z + abs(z̄)*t` |
|  | `v̄ += u*z̄` | `s_v += abs(u)*s_z + sqrt(abs(z̄)/t)` | `q_v += u^2*q_z + abs(z̄)/t` |

At the inputs, the cover is `h = s` for the sum cover and `h = sqrt.(q)` for
the sign-averaged cover. Sums, differences, and other linear operations follow
the first row. A value used by several operations accumulates a contribution
from each. In `x*x` the two arguments are separate, so `x` receives both
contributions.

The *split ratio* `t > 0` divides the curvature of a product between its two
arguments: any `t` gives a valid cover, and the choice determines its shape
(see [Choosing among equally valid local covers](@ref)). The package chooses
it as follows:

- **From the argument values:** `t = abs(v) / abs(u)`. This choice is
  covariant.
- **Guarded:** when the argument magnitudes differ by more than a factor
  `ratio` (default `sqrt(eps())`), or both are zero, the values carry no
  scale. Then `t = σ_v / σ_u`, where the scales `σ` are propagated forward from
  scales of the inputs by `σ_y = sum(abs(∂y/∂x_k) * σ_{x_k})`. The input scales
  are either declared by the caller or derived from the program's curvature
  (see [When argument values carry no scale](@ref)).
- **Dead argument:** if the gradient of `u` or `v` with respect to the inputs
  is exactly zero, the product contributes no seed to either argument.

Division and powers with a variable exponent also have curvature in both
arguments. Their split clamps the same equalizing ratio to an interval
determined by the diagonal second partials; see [`split_block`](@ref).

Scales derived from the curvature, `scales = CurvatureScales()`, are
`σ = 1 ./ h_drop`, where `h_drop` is the cover computed with the seeds of the
guarded products left out. The other seeds do not depend on `σ`, and the
accumulators are linear in the seeds, so a hand-written cover can compute
`h_drop` once and then add the guarded seeds. A guarded product whose scales
include an infinite `σ` (an input with `h_drop[k] == 0`) has no scale the
program determines. The hand-written code should then throw an error, as
`cover_hessian` does.

## A least-squares objective

This objective applies a robust (Cauchy) loss to the residuals of a linear
model. It contains a linear map, two unary operations per residual (`abs2` and
`log`), and a quadratic penalty. It has no products of two variables, so no
split is needed:

```jldoctest byhand
julia> using DerivativeCovers

julia> A = [1.0 2.0; -0.5 3.0; 2.0 -1.0]; b = [1.0, 0.0, -2.0]; λ = 0.5;

julia> f(x) = sum(log.(1 .+ abs2.(A * x .- b))) + λ / 2 * sum(abs2, x);

julia> function f_by_hand(x)
           r = A * x .- b
           s = abs2.(r)
           grad = A' * (2 .* r ./ (1 .+ s)) .+ λ .* x
           # Seeds: log(1 + s) has adjoint 1 and φ″ = -1/(1 + s)^2; abs2(r) has
           # adjoint 1/(1 + s) and φ″ = 2; each abs2(x[k]) has adjoint λ/2 and φ″ = 2.
           s_r = 2 .* abs.(r) ./ (1 .+ s) .+ sqrt.(2 ./ (1 .+ s))
           h_sum = abs.(A)' * s_r .+ sqrt(λ)
           q_r = 4 .* s ./ (1 .+ s) .^ 2 .+ 2 ./ (1 .+ s)
           h_var = sqrt.(abs2.(A)' * q_r .+ λ)
           return grad, h_sum, h_var
       end;

julia> x0 = [0.3, -0.7];

julia> grad, h_sum, h_var = f_by_hand(x0);

julia> round.(h_var; digits = 4)
2-element Vector{Float64}:
 1.8968
 3.5768

julia> c = cover_hessian(f, x0);

julia> v = cover_hessian(f, x0; merge = SignAveragedMerge());

julia> grad ≈ c.grad && h_sum ≈ c.h && h_var ≈ v.h
true
```

With an `m × n` matrix `A`, `f_by_hand` costs a few matrix-vector products,
whereas the recording stores all ``mn`` coefficients of `A` on the tape.

The example computes both covers to show the two rules side by side; in practice, compute only the one you need.
It may also be convenient to return the function value at the same time as the gradient and cover factor.

## A product whose argument can vanish

The program `f2` of [Scale covariance](@ref) contains the product
`x[1] * x[2]`, which is guarded where either argument is zero. Its
sign-averaged cover, with the same `scales` options as `cover_hessian`, is:

```jldoctest byhand
julia> f2(x) = x[1] * x[2] + exp(x[2]) + x[1]^2;

julia> isguarded(a, b, ratio) = (a == 0 && b == 0) || a <= ratio * b || b <= ratio * a;

julia> function f2_by_hand(x; scales = nothing, ratio = sqrt(eps()))
           x1, x2 = x
           grad = [x2 + 2x1, x1 + exp(x2)]
           q = [2.0, exp(x2)]                # seeds of x[1]^2 and exp(x[2])
           # The product x[1]*x[2] has adjoint 1.
           guarded = isguarded(abs(x1), abs(x2), ratio)
           if guarded
               σ = scales isa CurvatureScales ? 1 ./ sqrt.(q) : something(scales, [1.0, 1.0])
               all(isfinite, σ) ||
                   throw(ArgumentError("the program does not determine the input scales"))
               t = σ[2] / σ[1]
           else
               t = abs(x2) / abs(x1)
           end
           q .+= [t, 1 / t]
           return grad, sqrt.(q), Int(guarded)
       end;

julia> grad, h, nguarded = f2_by_hand([2.0, 0.0]; scales = CurvatureScales());

julia> nguarded
1

julia> h ≈ cover_hessian(f2, [2.0, 0.0]; merge = SignAveragedMerge(), scales = CurvatureScales()).h
true
```

Here `x[1]` receives curvature from `x[1]^2` and `x[2]` from `exp(x[2])`, so
`h_drop` is positive and the derived scales exist at every point. Without the
term `x[1]^2`, `x[1]` would receive curvature only through the product, and at
`x[2] == 0` the hand-written code would have to throw.

## Testing against the recording

A hand-written cover should agree with `cover_hessian` in its gradient, its
cover, and its count of guarded operations. Where `cover_hessian` throws
because the program does not determine a scale, the hand-written code should
throw too:

```jldoctest byhand
julia> function matches_recording(by_hand, f, x; scales = nothing, ratio = sqrt(eps()))
           recorded = try
               cover_hessian(f, x; merge = SignAveragedMerge(), scales, ratio)
           catch err
               err isa ArgumentError || rethrow()
               err
           end
           mine = try
               by_hand(x; scales, ratio)
           catch err
               err isa ArgumentError || rethrow()
               err
           end
           (recorded isa Exception || mine isa Exception) &&
               return recorded isa Exception && mine isa Exception
           grad, h, nguarded = mine
           return grad ≈ recorded.grad && h ≈ recorded.h && nguarded == recorded.nguarded
       end;

julia> points = [[2.0, 0.5], [-3.0, 1.5], [2.0, 0.0], [0.0, 0.5], [0.0, 0.0],
                 [1.0, 1e-9], [1e-9, 1.0], [1.0, -1e-7]];

julia> all(matches_recording(f2_by_hand, f2, x; scales)
           for x in points, scales in (nothing, [1.0, 100.0], CurvatureScales()))
true
```

Points drawn at random almost never reach a guarded operation, so the test
points should be chosen to reach every branch of the hand-written code:

- arguments of each product that are exactly zero, that lie just inside and just
  outside the `ratio` threshold, and that have either sign;
- points where an argument's gradient vanishes, if any;
- each source of scales: unit, declared, and `CurvatureScales()`;
- points where `CurvatureScales()` throws, if the program has any.

Array code that guards a product of two arrays, such as `U * V'`, is simpler
with `ratio = 0`: then a pair is guarded exactly when one of its entries is
zero, which a mask on `U` and `V` expresses. Test such code with the same
`ratio` passed to `cover_hessian`.
