# DerivativeCovers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/DerivativeCovers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/DerivativeCovers.jl/dev/)
[![Build Status](https://github.com/HolyLab/DerivativeCovers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/DerivativeCovers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/DerivativeCovers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HolyLab/DerivativeCovers.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This package computes **covers** of the Hessians and Jacobians of programs,
without forming either matrix. A Hessian cover is a nonnegative vector `h` such
that `abs(H[i,j]) <= h[i] * h[j]` for every entry of the Hessian `H`; a
Jacobian cover is a pair `a`, `b` such that `abs(J[k,j]) <= a[k] * b[j]`.

Covers are the natural scales of a problem: when the inputs are rescaled,
`x → D*x`, the cover transforms as `h → inv(D)*h`, so quantities such as
`H ./ (h .* h')` and `g ./ h` do not depend on the units of the inputs. See
[MatrixCovers](https://github.com/HolyLab/MatrixCovers.jl), which computes
covers of explicit matrices, for more on their uses.

DerivativeCovers records the program once and computes a Hessian cover in
`O(L)` work for a program of `L` operations, a small multiple of the cost of
the gradient, which it also returns. The bound is certified: it holds for the
exact derivatives, up to floating-point rounding, or exactly and over a whole
box when the program is recorded on intervals.

## Example

```julia
julia> using DerivativeCovers

julia> f(x) = exp(x[1]) + x[1] * x[2]^2;

julia> c = cover_hessian(f, [0.5, 2.0]);

julia> c.h_sum                      # abs(H[i,j]) <= h[i] * h[j]
2-element Vector{Float64}:
 4.112452541433932
 2.414213562373095

julia> c.grad
2-element Vector{Float64}:
 5.648721270700128
 2.0

julia> H = [exp(0.5) 4.0; 4.0 1.0];  # the exact Hessian, for comparison

julia> all(abs.(H) .<= c.h_sum .* c.h_sum')
true
```

Rescaling the inputs rescales the cover:

```julia
julia> D = [10.0, 0.1];

julia> cover_hessian(y -> f(y ./ D), D .* [0.5, 2.0]).h_sum ≈ c.h_sum ./ D
true
```

Tighter covers are available at higher cost, and other functions cover
Jacobians (`cover_jacobian`) and least-squares objectives (`cover_nls`). See
the [documentation](https://HolyLab.github.io/DerivativeCovers.jl/dev/) for a
guide to these options and the API reference.

## Installation

DerivativeCovers is not yet registered. Install it from its subdirectory in the
repository:

```julia
using Pkg
Pkg.add(url="https://github.com/HolyLab/DerivativeCovers.jl", subdir="DerivativeCovers")
```

Test problems for studying cover tightness are provided by the companion
package [DerivativeCoverProblems](../DerivativeCoverProblems).
