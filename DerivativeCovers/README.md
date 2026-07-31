# DerivativeCovers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/DerivativeCovers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/DerivativeCovers.jl/dev/)
[![Build Status](https://github.com/HolyLab/DerivativeCovers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/DerivativeCovers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/DerivativeCovers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HolyLab/DerivativeCovers.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Compute certified rank-1 bounds on a program's Hessian for about twice the cost of computing its
gradient.

A cover is a nonnegative vector `h` such that `abs(H[i,j]) <= h[i] * h[j]` for every entry of
the Hessian `H`. DerivativeCovers computes `h` without forming `H`. The result is deterministic
and certified rather than sampled or estimated. It also transforms consistently when the inputs
or objective are rescaled.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/HolyLab/DerivativeCovers.jl", subdir="DerivativeCovers")
```

## Quick start

Pass a scalar function and an evaluation point to `cover_hessian`:

```julia
using DerivativeCovers

f(x) = exp(x[1]) + sum(x[i] * x[i+1] for i in 1:length(x)-1)
x0 = [0.5, -1.2, 0.3, 0.8]

c = cover_hessian(f, x0)
c.h_sum   # 4-element Vector{Float64}: [2.833, 1.145, 3.633, 0.612]
c.grad    # the gradient comes along for free
```

The computation uses one forward pass and one paired reverse sweep: `O(L)` work for a tape of
length `L`. A cover bounds individual Hessian entries, the spectral radius, and the magnitudes of
Hessian-vector products:

```julia
v = [1.0, 0.0, -1.0, 0.5]
hvp(f, x0, v)                                      # the exact product, one more sweep
all(abs.(hvp(f, x0, v)) .<= c.h_sum .* (c.h_sum' * abs.(v)))  # true
```

Use `cover_jacobian` for vector-valued functions, `cover_nls` for nonlinear least-squares
objectives, and `cover_variance` for a cheaper, uncertified estimate. `cover_hybrid` lets you
trade computation for a tighter cover, while `SplitRule` controls how individual nodes select
among equally valid local covers.

Two options improve tightness at a cost of `O(n*L)`. `TangentChannel` selects the tangents over
which merges operate. The default absolute-value bound makes the `O(L)` sweep possible;
`SignedTangents` gives the tightest cover the package computes. Alternatively, `forward_cover`
propagates a cover forward without a tape, using memory that does not grow with the length of the
program. It occupies a different corner of the same tradeoff: it keeps every tangent sign and
discards every adjoint sign.

## Documentation

See the [rendered documentation](https://HolyLab.github.io/DerivativeCovers.jl/dev/) for the full
API, cover-quality utilities such as `cover_ratio` and `tightness`, and tools for studying how
cover looseness affects downstream algorithms. Reusable test problems live in the companion
package [DerivativeCoverProblems](../DerivativeCoverProblems).
