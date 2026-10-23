# DerivativeCoverProblems

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/DerivativeCovers.jl/stable/reference/problems/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/DerivativeCovers.jl/dev/reference/problems/)
[![Build Status](https://github.com/HolyLab/DerivativeCovers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/DerivativeCovers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/DerivativeCovers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HolyLab/DerivativeCovers.jl)

Parameterized families of scalar and vector-valued programs with reference Hessians and
Jacobians. Use them to measure the looseness of a derivative cover and investigate its causes.

The programs are written in generic Julia and do not depend on a differentiation package, so any
operator-overloading tool can run them. They are the test problems used by the
[DerivativeCovers](../DerivativeCovers) studies. This package supports development and
benchmarking rather than serving as a general-purpose library. Use it from a repository checkout,
with
`Pkg.develop(path="DerivativeCoverProblems")` in whatever environment the study runs in.

## Usage

A generator returns a `CoverProblem` containing the program `f`, the point `x0`, and—when a
closed form exists—the reference Hessian:

```julia
using DerivativeCovers, DerivativeCoverProblems, MatrixCovers

p = quadratic(6; rank=2, form=:factored)
c = cover_hessian(p.f, p.x0)
H = exact_hessian(p)

all(abs.(H) .<= c.h_sum * c.h_sum')             # true: the cover is guaranteed
max_coverage_ratio(c.h_sum, H)                  # 0.879: and it is nearly tight
```

The ratio approaches `1` as the cover approaches the tightest cover that the Hessian admits.
Holding `n`, `rank`, and the random seed fixed while varying `form` expresses the same quadratic
as three different programs, so the spread across `:entrywise` (0.051), `:matvec` (0.145), and
`:factored` (0.879) isolates the cost of graph structure alone.

The [rendered documentation](https://HolyLab.github.io/DerivativeCovers.jl/dev/reference/problems/)
describes every family and its parameters, including size, reconvergence, and dynamic range.
