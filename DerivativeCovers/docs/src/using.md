```@meta
CurrentModule = DerivativeCovers
```

# Using a cover

The examples on this page use the following program and its cover:

```jldoctest use
julia> using DerivativeCovers, LinearAlgebra

julia> f(x) = exp(x[1]) + x[1] * x[2] + x[2] * x[3] + x[3] * x[4];

julia> x0 = [0.5, -1.2, 0.3, 0.8];

julia> t, out = tape_function(f, x0);

julia> h = cover_hessian(t, out).h_sum;

julia> H = reduce(hcat, [hvp(t, out, e) for e in eachcol(Matrix(1.0I, 4, 4))]);   # for comparison
```

## What a cover bounds

If `abs.(H) .<= h .* h'`, then:

- each entry satisfies `abs(H[i,j]) ≤ h[i] * h[j]`;
- each Hessian-vector product satisfies `abs.(H * d) .≤ h .* (h' * abs.(d))`;
- the scaled Hessian `Hhat = H ./ (h .* h')` has entries in `[-1, 1]`, so its
  spectral radius is at most the number of nonzeros in the densest row of `H`,
  and in particular at most `n`.

The last is a statement about `Hhat` rather than `H`. When the inputs have
different units, the entries of `H` have different units too, and the
eigenvalue equation `H * v = λ * v` cannot be satisfied by a scalar `λ` with a
single unit. `Hhat` is dimensionless, and because the cover transforms with
the units (see [Scale covariance](@ref)), its eigenvalues do not depend on
them.

```jldoctest use
julia> d = [1.0, -2.0, 0.5, 0.0];

julia> all(abs.(H * d) .<= h .* (h' * abs.(d)))
true

julia> Hhat = H ./ (h .* h');

julia> maximum(abs, eigvals(Hhat)) <= maximum(count(!iszero, H; dims=2))
true
```

## Scale-invariant quantities

When the inputs are rescaled, `x → D*x`, the cover transforms as `h → inv(D)*h`
(see [Scale covariance](@ref)). Quantities formed by dividing by the cover
are therefore independent of the units of the inputs:

- the scaled Hessian `Hhat = H ./ (h .* h')`, whose entries lie in `[-1, 1]`;
- the scaled gradient `g ./ h`, whose norm [`Diagnostics.cover_norm`](@ref DerivativeCovers.Diagnostics.cover_norm)
  computes, and which serves as a convergence test;
- the scaled step `h .* p`, in terms of which the Newton equation `H * p = -g`
  becomes `Hhat * (h .* p) = -(g ./ h)`.

## Regularizing an indefinite Hessian

Modified Newton methods and Tikhonov regularization shift the Hessian to make
it positive definite. A shift by a multiple of the identity, `H + τ*I`, does
not respect units: the same `τ` perturbs each variable by an amount that
depends on its units. The diagonal of the cover respects them. The shift
`H + τ * Diagonal(h .^ 2)` is positive semidefinite when `τ ≥ -eigmin(Hhat)`,
so any upper bound on the spectral radius of `Hhat` gives a sufficient `τ`.

The number of nonzeros in the densest row of `H` (see
[What a cover bounds](@ref)) is such a bound. A tighter bound comes
from [`scaled_spectral_radius`](@ref), which estimates the spectral radius of
`Hhat` by power iteration using Hessian-vector products, without forming `H`:

```jldoctest use
julia> λ = DerivativeCovers.scaled_spectral_radius(t, out, h);

julia> round(λ; digits=3)
0.559

julia> isposdef(H + 1.1 * λ * Diagonal(h .^ 2))
true
```

Power iteration approaches the spectral radius from below, so the estimate is
inflated by a safety factor (here 1.1) before use. When `H` is available
explicitly, [`Diagnostics.psd_shift`](@ref DerivativeCovers.Diagnostics.psd_shift)
computes the smallest sufficient `τ`.

## Tightening a cover

If the spectral radius of `Hhat` is `λ < 1`, then every entry of `Hhat` is at
most `λ` in magnitude, and `sqrt(λ) * h` is also a cover. The power-iteration
estimate of `λ` needs the same safety factor:

```jldoctest use
julia> hs = sqrt(1.1 * λ) .* h;

julia> round(DerivativeCovers.cover_ratio(H, h); digits=3), round(DerivativeCovers.cover_ratio(H, hs); digits=3)
(0.449, 0.73)
```

[`cover_ratio`](@ref) is the largest value of `abs(H[i,j]) / (h[i] * h[j])`;
values closer to 1 indicate a tighter cover (see [Tightness](@ref)).

## Jacobians

For a Jacobian cover `(a, b)`, the scale-invariant matrix is
`Jhat = J ./ (a .* b')`, computed by [`equilibrate`](@ref). The Jacobian
counterparts of the functions above are [`scaled_singular_value`](@ref), which
computes the largest singular value of `Jhat` (by power iteration with
[`jvp`](@ref) and [`vjp`](@ref) when given a tape). If `σ < 1`, then
`(sqrt(σ) * a, sqrt(σ) * b)` is also a cover.

```jldoctest jac
julia> using DerivativeCovers

julia> r(x) = [x[1] * x[2], exp(x[1]) - x[3], x[2] + 2x[3]];

julia> t, outs = tape_function(r, [0.5, 2.0, -1.0]);

julia> jc = cover_jacobian(t, outs);

julia> σ = DerivativeCovers.scaled_singular_value(t, outs, jc.a, jc.b_sum);

julia> round(σ; digits=3)
0.866

julia> a, b = sqrt(1.1 * σ) .* jc.a, sqrt(1.1 * σ) .* jc.b_sum;
```

## Matrix-free products

The tape also supports exact products with the derivative matrices, each in
one or two sweeps over the tape (``O(L)`` time and memory):

- [`hvp`](@ref): Hessian-vector product `H * v`;
- [`jvp`](@ref): Jacobian-vector product `J * v`;
- [`vjp`](@ref): vector-Jacobian product `J' * u`.

The mutating forms [`hvp!`](@ref), [`jvp!`](@ref), and [`vjp!`](@ref) write
the result into a supplied array.
