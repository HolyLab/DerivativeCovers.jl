```@meta
CurrentModule = DerivativeCovers
```

# Writing a cover by hand

[`cover_hessian`](@ref) records a generic program and sweeps the recording. The
recording is convenient and general, but it stores every scalar operation, and
its recording and evaluation are more analogous to running code in an interpreter
than to executing compiled code. Thus its cost can be many times that of a
compiled hand-written gradient. When one program is evaluated many times, the
cover can instead be computed by code written for that program. In order of
preference, there are three ways to write it:

1. **Cover the analytic Hessian without forming it.** If ``H`` has a known
   form, such as a weighted Gram matrix plus a diagonal, a cover can often be
   written in closed form at a cost comparable to that of the gradient.
2. **Form the Hessian and cover it** with `symcover(H)` from
   [MatrixCovers](https://github.com/HolyLab/MatrixCovers.jl). This needs
   storage for every entry of ``H`` and time to compute them. For many
   important problems that cost is prohibitive.
3. **Write the sweep** that `cover_hessian` performs on the recording, as
   array operations added to the gradient code. This applies to any program
   and costs a few times as much as the gradient.

The first two cover ``H`` itself. The third covers one particular program
for the function, and is subject to the losses described in
[Tightness](@ref): contributions that would cancel in ``H`` are bounded
separately, and contributions that reach an input along several paths are
added. A cover of ``H`` can therefore be much tighter than the cover of any
program for it.

The approaches can also be mixed. When the objective is a sum of terms, each
term can be covered by whichever approach suits it, and the covers of the
terms combine into a cover of the whole; see [Combining the approaches](@ref).

## Covering the analytic Hessian

The following rules build covers of common forms of ``H`` without forming
the matrix. Each one satisfies `abs(H[i,j]) ≤ h[i] * h[j]` and is covariant
(see [Scale covariance](@ref)):

| Hessian | cover |
|:--- |:--- |
| diagonal `Diagonal(d)` | `sqrt.(abs.(d))` |
| Gram matrix `A' * Diagonal(w) * A` | `sqrt.(abs2.(A)' * abs.(w))` |
| sum `H_a + H_b`, given covers `h_a`, `h_b` | `sqrt.(h_a.^2 .+ h_b.^2)` |
| `[d₁ e; e d₂]` | `c .* sqrt.(abs.([d₁, d₂]))` with `c = sqrt(max(1, abs(e) / sqrt(abs(d₁ * d₂))))` |

The Gram and sum rules follow from the Cauchy–Schwarz inequality. In the
Gram rule the weights `w` may have either sign; `gramcover(A, w)` from
MatrixCovers computes the same cover and enlarges it slightly to absorb
the rounding of the sums. A block-diagonal Hessian is covered by covering each
block.

### A least-squares objective

This objective applies a robust (Cauchy) loss to the residuals of a linear
model and adds a quadratic penalty:

```math
f(x) = \sum_i \log\left(1 + r_i^2\right) + \frac{λ}{2} \lVert x \rVert^2,
\qquad r = Ax - b .
```

Its Hessian is ``A^{\mathsf T} \operatorname{Diagonal}(w) A + λI`` with
``w_i = 2(1 - r_i^2)/(1 + r_i^2)^2``, so the Gram and sum rules give the cover:

```jldoctest byhand
julia> using DerivativeCovers, MatrixCovers, LinearAlgebra

julia> A = [1.0 2.0; -0.5 3.0; 2.0 -1.0]; b = [1.0, 0.0, -2.0]; λ = 0.5;

julia> f(x) = sum(log.(1 .+ abs2.(A * x .- b))) + λ / 2 * sum(abs2, x);

julia> function f_cover(x)
           r = A * x .- b
           grad = A' * (2 .* r ./ (1 .+ r .^ 2)) .+ λ .* x
           w = 2 .* (1 .- r .^ 2) ./ (1 .+ r .^ 2) .^ 2
           h = sqrt.(abs2.(A)' * abs.(w) .+ λ)
           return grad, h
       end;

julia> x0 = [0.3, -0.7];

julia> grad, h = f_cover(x0);

julia> round.(h; digits = 4)
2-element Vector{Float64}:
 1.161
 1.8872
```

With an `m × n` matrix `A`, `f_cover` costs a few matrix-vector products. The
recording stores all ``mn`` coefficients of `A`, and forming ``H`` costs
``O(mn^2)``.

To test the cover, form ``H`` at a few points where that is affordable and
check that `max_coverage_ratio(h, H)`, from MatrixCovers, is at most 1 (up to
rounding; see [Measuring tightness](@ref)). Computing ``H`` with
[`hvp`](@ref) from the program `f`, rather than from the formula used to
derive the cover, also checks the formula. The ratio for the sum cover of
`cover_hessian` shows how much looser the cover of this program is:

```jldoctest byhand
julia> H = reduce(hcat, [hvp(f, x0, e) for e in eachcol(Matrix(1.0I, 2, 2))]);

julia> grad ≈ cover_hessian(f, x0).grad
true

julia> round(max_coverage_ratio(h, H); digits = 3)
0.719

julia> round(max_coverage_ratio(cover_hessian(f, x0).h, H); digits = 3)
0.036
```

### A product whose argument can vanish

The program `f2` of [Scale covariance](@ref) contains the product
`x[1] * x[2]`. The local Hessian of a product, `[0 1; 1 0]`, has no diagonal
entries, so the sweep must choose how to divide its curvature between `x[1]`
and `x[2]`. It makes that choice from the argument values, which carry no
scale where either is zero; there it falls back on declared or derived input
scales (see [When argument values carry no scale](@ref)).

The Hessian of the whole function does not have this problem:

```math
H = \begin{bmatrix} 2 & 1 \\ 1 & e^{x_2} \end{bmatrix},
```

and its diagonal sets the scale of each input at every point. The 2 × 2 rule
gives a covariant cover with no special cases:

```jldoctest byhand
julia> f2(x) = x[1] * x[2] + exp(x[2]) + x[1]^2;

julia> function f2_cover(x)
           x1, x2 = x
           grad = [x2 + 2x1, x1 + exp(x2)]
           d = [2.0, exp(x2)]                       # diagonal of H; H[1,2] == 1
           c = sqrt(max(1, 1 / sqrt(d[1] * d[2])))
           return grad, c .* sqrt.(d)
       end;

julia> f2_cover([2.0, 0.0])[2]
2-element Vector{Float64}:
 1.4142135623730951
 1.0
```

If the term `x[1]^2` were absent, `H[1,1]` would be zero and the Hessian
would determine no scale for `x[1]`. No covariant cover exists then, by any
of the three methods: the sweep with `scales = CurvatureScales()` throws, and
`symcover` returns a cover that does not transform with the units.

## Forming the Hessian

When ``n`` is small, or ``H`` is sparse with few nonzeros, ``H`` can be
formed explicitly, from its formula or from ``n`` Hessian-vector products,
and covered with `symcover(H)`. The cost of `symcover` is proportional to the
number of stored entries of `H`; `symcover_min(H)` gives the ℓ₂-minimal cover
at greater cost. On `f2`, `symcover` gives the same cover as the 2 × 2 rule:

```jldoctest byhand
julia> H2(x) = [2.0 1.0; 1.0 exp(x[2])];

julia> all(f2_cover(x)[2] ≈ symcover(H2(x)) for x in ([2.0, 0.0], [0.0, -3.0], [-3.0, 1.5]))
true
```

The ``O(n^2)`` storage and the ``n`` Hessian-vector products make this method
impractical for large dense problems. Its main role there is as a reference
for testing a cover from the first method on small instances.

## Writing the sweep

When ``H`` has no form that the first method can cover and is too large to
form, the sweep of `cover_hessian` can be written out for one program. Such
code reproduces the cover of that particular program, and `cover_hessian`
serves as the reference implementation for testing it. The hand-written code
and the program it is tested against must therefore perform the same
operations.

### The rules

The sweep extends reverse-mode differentiation. The reverse pass computes the
adjoint ``\bar v`` of each intermediate value; alongside it, each value
receives one more accumulator, ``s`` for the sum cover or ``q`` for the
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

### The sweep for the least-squares objective

The objective `f` above contains a linear map, two unary operations per
residual (`abs2` and `log`), and a quadratic penalty. It has no products of
two variables, so no split is needed:

```jldoctest byhand
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

The example computes both covers to show the two rules side by side; in
practice, compute only the one you need. It may also be convenient to return
the function value at the same time as the gradient and cover.

Compare `q_r` with the weights `w` of `f_cover`: the sweep bounds the
curvature of `abs2` and of `log` separately, giving
``(6r_i^2 + 2)/(1 + r_i^2)^2`` where ``H`` has ``\lvert w_i \rvert =
2\lvert 1 - r_i^2 \rvert/(1 + r_i^2)^2``.

### A guarded product

The sweep for `f2` must handle the guarded product. Its sign-averaged cover,
with the same `scales` options as `cover_hessian`, is:

```jldoctest byhand
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
`h_drop` is positive and the derived scales exist at every point. These are
the diagonal entries of ``H`` that `f2_cover` uses directly.

### Testing against the recording

A hand-written sweep should agree with `cover_hessian` in its gradient, its
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

## Combining the approaches

When the objective is a sum ``f = f_a + f_b``, each term can be covered by a
different method, and the sum rule combines their covers:
`h = sqrt.(h_a.^2 .+ h_b.^2)`. Because the sum rule applies only to covers,
a sweep used for one of the terms should compute the sum cover.

The cover `h_a` of a term covered analytically also supplies scales for the
guarded products in a term covered by a sweep. Passing `σ = 1 ./ h_a` as the
declared scales of the sweep keeps the combined cover covariant at guarded
points, because `h_a` itself is covariant. The scales are computed before the
sweep and do not depend on its seeds, so `cover_hessian` with
`scales = 1 ./ h_a` remains an exact reference for the sweep, including at
guarded points. An input with `h_a[k] == 0` has no scale, and a guarded
product that reaches it should throw.

The following objective adds a product to the least-squares objective `f`. The
cover of `f` comes from `f_cover`, and the product is swept with the scales
`1 ./ h_a`:

```jldoctest byhand
julia> f3(x) = f(x) + x[1] * x[2];

julia> f3_b(x) = x[1] * x[2];

julia> function f3_b_by_hand(x; scales, ratio = sqrt(eps()))
           x1, x2 = x
           grad = [x2, x1]
           guarded = isguarded(abs(x1), abs(x2), ratio)
           if guarded
               all(isfinite, scales) ||
                   throw(ArgumentError("the program does not determine the input scales"))
               t = scales[2] / scales[1]
           else
               t = abs(x2) / abs(x1)
           end
           return grad, [sqrt(t), 1 / sqrt(t)], Int(guarded)
       end;

julia> function f3_cover(x; ratio = sqrt(eps()))
           grad_a, h_a = f_cover(x)
           grad_b, h_b, nguarded = f3_b_by_hand(x; scales = 1 ./ h_a, ratio)
           return grad_a .+ grad_b, sqrt.(h_a .^ 2 .+ h_b .^ 2), nguarded
       end;
```

Here `h_a ≥ sqrt(λ) > 0`, so the scales are always finite.

Testing splits along the same lines as the cover. `f_cover` is tested against
the Hessian of `f`, as above. The sweep is tested against `cover_hessian` of
the term it sweeps, with the same scales:

```jldoctest byhand
julia> points = [[0.3, -0.7], [0.3, 0.0], [0.0, -0.7], [0.0, 0.0], [1.0, 1e-9], [1.0, 1e-7]];

julia> all(points) do x
           _, h_a = f_cover(x)
           grad_b, h_b, nguarded = f3_b_by_hand(x; scales = 1 ./ h_a)
           ref = cover_hessian(f3_b, x; scales = 1 ./ h_a)
           grad_b ≈ ref.grad && h_b ≈ ref.h && nguarded == ref.nguarded
       end
true
```

The sum rule needs no test of its own. A check of the combined cover against
the Hessian of `f3` at a guarded point confirms the assembly, and shows the
gain over sweeping the whole objective:

```jldoctest byhand
julia> x = [0.3, 0.0];

julia> grad3, h3, nguarded = f3_cover(x);

julia> H3 = reduce(hcat, [hvp(f3, x, e) for e in eachcol(Matrix(1.0I, 2, 2))]);

julia> grad3 ≈ cover_hessian(f3, x).grad
true

julia> round(max_coverage_ratio(h3, H3); digits = 3)
0.85

julia> round(max_coverage_ratio(cover_hessian(f3, x; scales = CurvatureScales()).h, H3); digits = 3)
0.121
```
