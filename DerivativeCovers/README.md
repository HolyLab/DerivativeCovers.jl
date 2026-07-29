# DerivativeCovers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/DerivativeCovers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/DerivativeCovers.jl/dev/)
[![Build Status](https://github.com/HolyLab/DerivativeCovers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/DerivativeCovers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/DerivativeCovers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HolyLab/DerivativeCovers.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Rank-1 covers of a program's Hessian at about twice the cost of a gradient.

A cover is a nonnegative vector `h` with `abs(H[i,j]) <= h[i]*h[j]` for every entry of the
Hessian `H`, obtained without ever forming `H`. The covers are deterministic and guaranteed —
not sampled, not estimated — and they are covariant under rescaling of the inputs and of the
objective, so they carry the units of the problem rather than of the way it was written down.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/HolyLab/DerivativeCovers.jl", subdir="DerivativeCovers")
```

## Quick start

A scalar program is recorded by running it on `TVar` inputs; nothing about the function has to
be declared in advance:

```julia
using DerivativeCovers

f(x) = exp(x[1]) + sum(x[i] * x[i+1] for i in 1:length(x)-1)
x0 = [0.5, -1.2, 0.3, 0.8]

c = cover_hessian(f, x0)
c.h_sum   # 4-element Vector{Float64}: [2.833, 1.145, 3.633, 0.612]
c.grad    # the gradient comes along for free
```

One forward pass and one paired reverse sweep, `O(L)` in the tape length. The cover then bounds
things the Hessian would have told you — an entry by `h[i]*h[j]`, the spectral radius by
`norm(h)^2`, a Hessian-vector product by `h .* (h'*abs.(v))`:

```julia
v = [1.0, 0.0, -1.0, 0.5]
hvp(f, x0, v)                            # the exact product, one more sweep
all(abs.(hvp(f, x0, v)) .<= c.h_sum .* (c.h_sum'abs.(v)))   # true
```

`cover_jacobian`, `cover_nls`, and `cover_variance` are the entry points for vector-valued
programs, least-squares losses, and the variance-channel estimate. `cover_hybrid` trades cost
against tightness through a group count, and a `SplitRule` selects which cover within the valid
family a node contributes.

## Documentation

The [rendered documentation](https://HolyLab.github.io/DerivativeCovers.jl/dev/) covers the entry
points, the quantities defined on a cover (`cover_ratio`, `tightness`, `equilibrate`, …), and the
`Diagnostics` submodule, which supplies controlled distortions of a cover and the task-level
consumers that reveal what a distortion costs. The programs those instruments are exercised on
live in the companion package [DerivativeCoverProblems](../DerivativeCoverProblems).
