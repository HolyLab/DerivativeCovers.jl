```@meta
CurrentModule = DerivativeCovers
```

# Scale covariance

A cover is useful as a set of natural scales only if it does not depend on the
units in which the inputs are expressed. This page states when the covers
computed by this package have this property.

## Covariance

Rescaling the inputs, `x → D*x` for a positive diagonal `D`, changes the
Hessian to `inv(D)*H*inv(D)`. Rescaling the output, `f → c*f` with `c > 0`,
changes it to `c*H`. A Hessian cover is *covariant* if it transforms the same
way: `h → inv(D)*h` and `h → sqrt(c)*h`. For a Jacobian cover, rescaling the
inputs by `D` and the outputs by `C` should give `b → inv(D)*b` and
`a → C*a`, up to the scalar factor that a Jacobian cover leaves free.

In the following example, the program `g` is `f` with its inputs expressed in
different units:

```jldoctest cov
julia> using DerivativeCovers

julia> f(x) = x[1] * x[2] + exp(x[2]);

julia> x0 = [2.0, 0.5];

julia> D = [10.0, 0.1];

julia> g(y) = f(y ./ D);

julia> cover_hessian(g, D .* x0).h_sum ≈ cover_hessian(f, x0).h_sum ./ D
true

julia> cover_hessian(x -> 9 * f(x), x0).h_sum ≈ 3 * cover_hessian(f, x0).h_sum
true
```

The covers of this package are covariant on any program in which every
operation can choose its local cover from the values of its arguments. The
rest of this page explains what this means and what happens otherwise.

## Choosing among equally valid local covers

The product `x * y` has second partial derivatives `[0 1; 1 0]`. Any pair
`(ρ₁, ρ₂)` with `ρ₁ * ρ₂ ≥ 1` covers this matrix, so the operation has a
one-parameter family of equally valid local covers, and the choice determines
how its contribution is distributed between `x` and `y`. A choice based on a
fixed number, such as `ρ₁ = ρ₂ = 1`, would not be covariant: if `x` is
measured in millimeters instead of meters, its share of the cover should change
by a factor of 1000.

The package resolves the choice using the magnitudes of the arguments. For a
product, it chooses `ρ₁ = sqrt(|y| / |x|)` and `ρ₂ = sqrt(|x| / |y|)`, which
transform correctly under any rescaling of `x` and `y`. In the example above,
`x0 = [2.0, 0.5]` gives `ρ₁ = 0.5` and `ρ₂ = 2`; the second entry of the cover
adds `sqrt(exp(0.5)) ≈ 1.284` for the `exp` term:

```jldoctest cov
julia> cover_hessian(f, x0).h_sum
2-element Vector{Float64}:
 0.5
 3.2840254166877414
```

Other operations with a family of local covers, such as division and powers
with a variable exponent, are resolved in the same way. This choice is made
by [`split_block`](@ref).

## When argument values carry no scale

The rule above fails when an argument is zero: `ρ₂ = sqrt(|x| / |y|)` is
infinite at `y = 0`, and no choice based on the argument values is both finite
and covariant. At such operations the package uses *declared* scales for the
inputs instead. The `scales` keyword, accepted by every cover function,
supplies one positive scale per input; the default, `nothing`, sets them all
to 1, which amounts to asserting that the inputs share units.
[`guards_any_block`](@ref) reports whether a recorded program contains such an
operation:

```jldoctest cov
julia> x1 = [2.0, 0.0];

julia> t, out = tape_function(f, x1);

julia> DerivativeCovers.guards_any_block(t)
true

julia> cover_hessian(f, x1).h_sum
2-element Vector{Float64}:
 1.0
 2.0

julia> cover_hessian(f, x1; scales=[1.0, 100.0]).h_sum
2-element Vector{Float64}:
 10.0
  1.1
```

The declared scales affect only which valid cover is chosen, never validity.
When the inputs are rescaled, a cover that depends on declared scales is
covariant only if the scales are rescaled along with them, `scales → D*scales`:

```jldoctest cov
julia> cover_hessian(g, D .* x1; scales=D).h_sum .* D
2-element Vector{Float64}:
 1.0
 2.0
```

On a program for which [`guards_any_block`](@ref) returns `false`, the
`scales` keyword has no effect and the cover is covariant without it.

The test for "carries no scale" is not limited to exact zeros. An argument
pair whose magnitudes differ by more than a factor `ratio` is treated the same
way, so that a value that is zero only up to roundoff does not lead to an
enormous local cover. The `ratio` keyword sets this threshold; the default,
`sqrt(eps())` for `Float64`, is described in [`split_ratio`](@ref). Because
this test compares two values that may have different units, a rescaling can
move an operation across the threshold, and covariance then holds only
approximately.

Near such points the cover can be discontinuous. For a quotient `x / y`, as
`x` approaches zero the local cover remains covariant but grows without bound;
at `x = 0` it drops to the finite cover given by the declared scales.

## Jacobian covers

The default `start=:covariant` of [`cover_jacobian`](@ref) produces products
`a[k] * b[j]` that transform exactly under rescaling of the inputs, the
outputs, and the intermediate values of the program. The `:equal` start
instead treats all outputs alike: permuting the outputs permutes the cover,
but the products do not transform exactly under rescaling.
