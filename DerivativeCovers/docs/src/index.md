```@meta
CurrentModule = DerivativeCovers
DocTestSetup = quote
    using DerivativeCovers
end
```

# DerivativeCovers

```@docs
DerivativeCovers
```

## Installation

The package is not registered, and lives in a subdirectory of its repository:

```julia
using Pkg
Pkg.add(url="https://github.com/HolyLab/DerivativeCovers.jl", subdir="DerivativeCovers")
```

## Quick start

Record a scalar program by running it on [`TVar`](@ref) inputs, then ask for a cover:

```jldoctest quickstart
julia> f(x) = exp(x[1]) + sum(x[i] * x[i+1] for i in 1:length(x)-1);

julia> x0 = [0.5, -1.2, 0.3, 0.8];

julia> c = cover_hessian(f, x0);

julia> round.(c.h_sum; digits=4)
4-element Vector{Float64}:
 2.8332
 1.1455
 3.633
 0.6124
```

`c.h_sum` is a nonnegative vector `h` with `abs(H[i,j]) <= h[i]*h[j]` for every entry of the
Hessian `H`, which was never formed. The gradient comes along for free:

```jldoctest quickstart
julia> round.(c.grad; digits=4)
4-element Vector{Float64}:
  0.4487
  0.8
 -0.4
  0.3
```

That is the whole "do" path: one forward pass and one paired reverse sweep, `O(L)` in the tape
length. [`cover_jacobian`](@ref), [`cover_nls`](@ref), and [`cover_variance`](@ref) are the
corresponding entry points for vector-valued programs, least-squares losses, and the
variance-channel estimate.

## Using a cover

A cover is a bound, so what you do with it is bound something. `h[i]*h[j]` bounds a Hessian
entry, `norm(h)^2` bounds the spectral radius, and the products `h .* d` bound the change in the
gradient under a step `d`. [`hvp`](@ref), [`jvp`](@ref), and [`vjp`](@ref) supply the exact
matrix-free products alongside, each at one sweep.

Two knobs change what you get:

- **Tightness against cost.** [`cover_hybrid`](@ref) and [`cover_jacobian_hybrid`](@ref) take an
  integer group count that interpolates between the cheap sum merge and the tighter quadrature
  merge, at `O(G*L)`.
- **Which cover within the valid family.** A node with an active off-diagonal admits a
  one-parameter family of rank-1 covers and nothing about the node selects within it; a
  [`SplitRule`](@ref) supplies the scale reference. [`ValueSplit`](@ref) (the default) uses parent
  value magnitudes, [`SensitivitySplit`](@ref) uses input-weighted sensitivities.

The quantities defined *on* a cover — [`cover_ratio`](@ref), [`tightness`](@ref),
[`equilibrate`](@ref), and the rest — are public but unexported, and live on the
[Quantities and extension points](@ref) page.

## Studying a cover

The looseness of a cover is itself an object of study, and the package carries the instruments for
measuring it. [`WithDiagnostics`](@ref) asks [`cover_hessian`](@ref) for the elementwise bound `B`
and the alternative merges alongside the cover, so a ratio can be computed directly:

```jldoctest quickstart
julia> d = cover_hessian(WithDiagnostics(), f, x0);

julia> round(DerivativeCovers.cover_ratio(d.B, d.h_sum); digits=4)
0.4495
```

This form costs `O(n*L)` time and `O(n^2)` memory — it forms the very matrix the production path
exists to avoid — so it is opt-in and never the default.

Beyond that, [Diagnostics](@ref) supplies controlled distortions of a cover and the task-level
consumers that reveal what a distortion costs, and the companion package
`DerivativeCoverProblems` supplies the [Problem families](@ref) those instruments are exercised
on.

## How this documentation is organized

The three reference pages are the package's three-tier name partition, rendered:

- This page documents the exported names — everything `using DerivativeCovers` brings into scope.
- [Quantities and extension points](@ref) documents the names that are `public` but unexported:
  reached by qualification, and covered by the same compatibility promise.
- [Internals](@ref) documents everything else. These names are reachable and documented, but
  carry no promise: they are the sweep machinery a reimplementation is free to restructure.

## Recording a program

```@docs
Tape
TVar
tape_function
```

## Cover entry points

```@docs
cover_hessian
cover_jacobian
cover_nls
cover_variance
cover_hybrid
cover_jacobian_hybrid
```

## Selecting what `cover_hessian` returns

```@docs
CoverOnly
WithDiagnostics
```

## Result types

```@docs
HessianCover
HessianDiagnostics
JacobianCover
NLSCover
VarianceCover
```

## Split rules

```@docs
SplitRule
ValueSplit
SensitivitySplit
```

## Matrix-free products

```@docs
hvp
hvp!
jvp
jvp!
vjp
vjp!
```

```@index
Pages = ["index.md"]
```
