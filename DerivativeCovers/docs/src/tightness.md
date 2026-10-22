```@meta
CurrentModule = DerivativeCovers
```

# Tightness

Every cover computed by this package is valid, but some are much looser than
others. This page describes how to measure looseness and where it comes from.

## Measuring tightness

When the Hessian `H` is available,
[`max_coverage_ratio`](https://github.com/HolyLab/MatrixCovers.jl) from
MatrixCovers computes the largest ratio `abs(H[i,j]) / (h[i] * h[j])` over the
nonzero entries of `H`. A value of at most 1 confirms that `h` is a cover, and
values closer to 1 indicate a tighter cover. The ratio reflects only the entry
on which the cover is tightest, however: if `r = max_coverage_ratio(h, H)`,
then `sqrt(r) * h` has ratio exactly 1, however loose it remains elsewhere.
For a Jacobian cover `(a, b)`, the ratio is `max_coverage_ratio(a, b, J)`.

A reference cover computed from `H` itself, such as `symcover(H)` from
[MatrixCovers](https://github.com/HolyLab/MatrixCovers.jl), allows a
comparison entry by entry, which separates a uniform excess from a nonuniform
one.

## Where looseness comes from

### The rank-one form

A rank-one bound cannot match every matrix. A Jacobian with two blocks of
large entries, for example, forces any rank-one cover to be loose on the
small entries that join them. This floor depends only on the matrix, not on
the program.

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
julia> using DerivativeCovers, DerivativeCoverProblems, MatrixCovers

julia> function ratios(form)
           p = quadratic(6; rank=2, form)
           H = exact_hessian(p)
           hsum = cover_hessian(p.f, p.x0).h_sum
           hquad = cover_hessian(WithDiagnostics(), p.f, p.x0).h_quad
           return round.((max_coverage_ratio(hsum, H), max_coverage_ratio(hquad, H)); digits=3)
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

Without the Hessian, the ratio `h_sum ./ h_var` of the two vectors in a
[`HessianCover`](@ref) estimates how much the contributions to each coordinate
overlap: it is 1 when each coordinate receives a single contribution and grows
with the number of overlapping contributions.

### Lost cancellation

The construction loses tightness wherever it takes absolute values of terms
that would have cancelled. This happens in two places. The intermediate gradients carry the
influence of each input forward through the program, and the adjoints carry
the sensitivity of the output backward. Either can be computed with signs or in
absolute-value arithmetic. The default [`cover_hessian`](@ref) keeps the
adjoint signs and bounds the intermediate gradients, which is what lets it run
in a single ``O(L)`` sweep; [`forward_cover`](@ref) does the opposite (see
[Propagating a cover forward](@ref)). A difference of nearly equal terms, such
as `exp(x) - 1 - x` near zero, has an exact intermediate gradient much smaller
than its absolute-value bound; a value that reaches the output along paths
with opposite signs has an adjoint much smaller than its absolute-value bound.
When the program is small enough to form `H`, a cover computed from `H`
itself, such as `symcover(H)`, loses neither.

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
intermediate gradients or adjoints include
[`cancellation`](@ref DerivativeCoverProblems.cancellation),
[`adjoint_cancellation`](@ref DerivativeCoverProblems.adjoint_cancellation),
and [`converged_inner_solve`](@ref DerivativeCoverProblems.converged_inner_solve).
