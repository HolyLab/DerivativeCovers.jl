```@meta
CurrentModule = DerivativeCovers
```

# Tightness

Every cover computed by this package is valid, but some are much looser than
others. This page describes how to measure looseness, where it comes from, and
the tools for studying its effect on downstream algorithms.

## Measuring tightness

When the Hessian `H` is available, [`cover_ratio`](@ref) computes the largest
ratio `abs(H[i,j]) / (h[i] * h[j])` over the nonzero entries of `H`. A value
of at most 1 confirms that `h` is a cover, and values closer to 1 indicate a
tighter cover. The ratio reflects only the entry on which the cover is
tightest, however: if `r = cover_ratio(H, h)`, then `sqrt(r) * h` has ratio
exactly 1, however loose it remains elsewhere.

A reference cover computed from `H` itself, such as `symcover(H)` from
[MatrixCovers](https://github.com/HolyLab/MatrixCovers.jl), allows a
comparison coordinate by coordinate. [`tightness`](@ref) summarizes the
profile `h ./ href` by its maximum and geometric mean, which show whether the
looseness is spread evenly or concentrated in a few coordinates.

[`rank1_ratio`](@ref) and [`bound_ratio`](@ref) compare a cover and the
elementwise bound `B` of [`HessianDiagnostics`](@ref) with each other and with
`H`. For Jacobians, [`jacobian_ratio`](@ref) is the counterpart of
`cover_ratio`.

## Where looseness comes from

### The rank-one form

A rank-one bound cannot match every matrix. For a Jacobian (or any nonnegative
matrix `M`), [`jacobian_representability`](@ref) gives a lower bound on the
looseness that any rank-one cover must have, and
[`jacobian_minimax`](@ref) computes the smallest achievable worst-case
looseness together with a cover that attains it. These depend only on the
matrix, not on the program.

### The structure of the program

The covers are computed from the operations of the program, not from the
matrix, so different programs for the same function can give different
covers. A contribution reaches an input along every path through the program,
and contributions that reach the same input along several paths are summed
rather than allowed to cancel. Programs whose paths reconverge often therefore
give looser covers.

The [`quadratic`](@ref DerivativeCoverProblems.quadratic) problem family of
the companion package
[DerivativeCoverProblems](https://github.com/HolyLab/DerivativeCovers.jl/tree/main/DerivativeCoverProblems)
writes the same quadratic form ``x^{\mathsf T}Ax/2`` as three different
programs:

```jldoctest structure
julia> using DerivativeCovers, DerivativeCoverProblems

julia> function ratios(form)
           p = quadratic(6; rank=2, form)
           H = exact_hessian(p)
           hsum = cover_hessian(p.f, p.x0).h_sum
           hquad = cover_hessian(WithDiagnostics(), p.f, p.x0).h_quad
           return round.((DerivativeCovers.cover_ratio(H, hsum), DerivativeCovers.cover_ratio(H, hquad)); digits=3)
       end;

julia> ratios(:entrywise)   # sum_ij A[i,j] * x[i] * x[j] / 2
(0.051, 0.332)

julia> ratios(:matvec)      # x' * (A * x) / 2
(0.145, 0.397)

julia> ratios(:factored)    # sum((W' * x) .^ 2) / 2, with A = W * W'
(0.879, 1.0)
```

The entrywise form sends each of the `n^2` entries of `A` through its own
product, and all of them reconverge on the inputs. The factored form has only
`rank` nonlinear operations, which do not overlap. The quadrature merge
recovers part of the difference.

Without the Hessian, the `severity` returned by [`cover_variance`](@ref)
estimates how much the contributions to each coordinate overlap: it is 1 when
each coordinate receives a single contribution and grows with the number of
overlapping contributions.

### Lost cancellation

The construction loses tightness wherever it takes absolute values of terms
that would have cancelled. This happens in two places. The input gradients carry the
influence of each input forward through the program, and the adjoints carry
the sensitivity of the output backward. Either can be computed with signs or in
absolute-value arithmetic. [`cover_lattice`](@ref) computes the covers for all
four combinations, which shows what each choice costs for a given program:

```jldoctest lattice
julia> using DerivativeCovers

julia> g(x) = (d = exp(x[1]) - 1 - x[1]; u = d * x[2]; exp(u) - u);

julia> lat = cover_lattice(g, [0.4, 1.5]);

julia> round.(lat.h_quad_signed ./ lat.h_quad; digits=3)     # gain from signed input gradients
2-element Vector{Float64}:
 0.222
 1.0

julia> round.(lat.h_quad ./ lat.h_quad_absadj; digits=3)     # gain from signed adjoints
2-element Vector{Float64}:
 0.362
 0.364
```

In this program, `d` is a difference of nearly equal terms, so its exact
input gradient is much smaller than its absolute-value bound. The value `u` reaches
the output along two paths whose contributions have opposite signs, so the
signed adjoint is smaller than its absolute-value bound. The default
[`cover_hessian`](@ref) keeps the adjoint signs and bounds the input gradients;
[`SignedTangents`](@ref) keeps both (see
[Choosing the input-gradient channel](@ref)).

## Test problems

The `quadratic` family used above is one of several in DerivativeCoverProblems.
Each generator returns a
[`CoverProblem`](@ref DerivativeCoverProblems.CoverProblem) (or, for
vector-valued programs, a
[`JacobianProblem`](@ref DerivativeCoverProblems.JacobianProblem)) containing
the program, an evaluation point, and, where available, the exact Hessian:

```jldoctest problems
julia> using DerivativeCovers, DerivativeCoverProblems

julia> p = quadratic(6; rank=2, form=:factored);

julia> c = cover_hessian(p.f, p.x0);

julia> H = exact_hessian(p);

julia> all(abs.(H) .<= c.h_sum .* c.h_sum')
true
```

[`families`](@ref DerivativeCoverProblems.families) returns a standard set of
generators that differ in how strongly the paths of the program reconverge.
[`rescale`](@ref DerivativeCoverProblems.rescale) and
[`logscales`](@ref DerivativeCoverProblems.logscales) change the units of a
problem's inputs, for testing covariance. Problems with cancellation in the
input gradients or adjoints include
[`cancellation`](@ref DerivativeCoverProblems.cancellation),
[`adjoint_cancellation`](@ref DerivativeCoverProblems.adjoint_cancellation),
and [`converged_inner_solve`](@ref DerivativeCoverProblems.converged_inner_solve).

## Effects on downstream algorithms

How much looseness matters depends on how the cover is used. The
[`DerivativeCovers.Diagnostics`](@ref) module provides tools to inflate a cover
in controlled patterns and to measure the effect on several uses of a cover.

[`inflate`](@ref DerivativeCovers.Diagnostics.inflate) loosens a cover by
coordinatewise factors, generated by an
[`InflationPattern`](@ref DerivativeCovers.Diagnostics.InflationPattern) with
a given peak factor. The inflated cover is still valid. Each of the following
compares a use of the reference cover with the same use of the inflated one:

- [`step_degradation`](@ref DerivativeCovers.Diagnostics.step_degradation):
  conjugate gradients preconditioned by the cover
  ([`cover_cg`](@ref DerivativeCovers.Diagnostics.cover_cg));
- [`psd_excess`](@ref DerivativeCovers.Diagnostics.psd_excess): the shift that
  makes the Hessian positive semidefinite
  ([`psd_shift`](@ref DerivativeCovers.Diagnostics.psd_shift));
- [`regularization_bias`](@ref DerivativeCovers.Diagnostics.regularization_bias):
  the regularized Newton step
  ([`regularized_step`](@ref DerivativeCovers.Diagnostics.regularized_step));
- [`stopping_shift`](@ref DerivativeCovers.Diagnostics.stopping_shift): the
  iteration at which a convergence test based on
  [`cover_norm`](@ref DerivativeCovers.Diagnostics.cover_norm) stops.

Some uses respond only to the shape of the inflation. Multiplying a cover by a
constant does not change the shift `τ * Diagonal(h .^ 2)` that makes a matrix
positive semidefinite, but inflating a single coordinate does:

```jldoctest consumers
julia> using DerivativeCovers, DerivativeCovers.Diagnostics, LinearAlgebra, MatrixCovers

julia> A = [2.0 1.0 0.0; 1.0 -1.0 0.5; 0.0 0.5 3.0];

julia> h = symcover(A);

julia> maximum(psd_excess(A, h, inflate(UniformInflation(), h, 100.0)).excess) ≈ 1
true

julia> maximum(psd_excess(A, h, inflate(SpikeInflation(2), h, 100.0)).excess) > 1
true
```
