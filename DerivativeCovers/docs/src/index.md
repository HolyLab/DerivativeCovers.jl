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

DerivativeCovers is not yet registered. Install it directly from its subdirectory in the
repository:

```julia
using Pkg
Pkg.add(url="https://github.com/HolyLab/DerivativeCovers.jl", subdir="DerivativeCovers")
```

## Quick start

Call [`cover_hessian`](@ref) with a scalar function and the point where you want to evaluate it:

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

The result contains a nonnegative vector `h` such that
`abs(H[i,j]) <= h[i] * h[j]` for every entry of the Hessian `H`. The Hessian itself is never
formed. The same reverse sweep also computes the gradient:

```jldoctest quickstart
julia> round.(c.grad; digits=4)
4-element Vector{Float64}:
  0.4487
  0.8
 -0.4
  0.3
```

This computation takes one forward pass and one paired reverse sweep, or `O(L)` work for a tape
of length `L`. Use [`cover_jacobian`](@ref) for vector-valued functions, [`cover_nls`](@ref) for
nonlinear least-squares objectives, and [`cover_variance`](@ref) for the cheaper, uncertified
variance estimate.

## Using a cover

If `h` covers a Hessian `H`, then `h[i] * h[j]` bounds `abs(H[i,j])`, `norm(h)^2` bounds the
spectral radius, and `h .* (h' * abs.(d))` bounds `abs.(H * d)`. The package can also compute
exact matrix-free products with [`hvp`](@ref), [`jvp`](@ref), and [`vjp`](@ref), each in one
additional sweep.

Two options control the result:

- **Cost versus tightness.** [`cover_hybrid`](@ref) and
  [`cover_jacobian_hybrid`](@ref) divide contributions into `G` groups. Larger `G` usually gives
  a tighter cover and costs `O(G * L)`.
- **How local covers are scaled.** Some nodes admit a family of equally valid rank-1 covers. A
  [`SplitRule`](@ref) chooses among them. [`ValueSplit`](@ref), the default, uses the magnitudes
  of parent values; [`SensitivitySplit`](@ref) uses input-weighted sensitivities.

Utilities such as [`cover_ratio`](@ref), [`tightness`](@ref), and [`equilibrate`](@ref) are
public but not exported. Call them with the `DerivativeCovers.` prefix; see
[Quantities and extension points](@ref).

## Studying a cover

Pass [`WithDiagnostics`](@ref) to [`cover_hessian`](@ref) to inspect the elementwise bound `B`
and alternative merges:

```jldoctest quickstart
julia> d = cover_hessian(WithDiagnostics(), f, x0);

julia> round(DerivativeCovers.cover_ratio(d.B, d.h_sum); digits=4)
0.4495
```

Diagnostics cost `O(n * L)` time and `O(n^2)` memory because they form an explicit matrix.
They are therefore opt-in.

A cover loses tightness wherever it propagates in absolute-value arithmetic a quantity that would
have cancelled with signs. Two channels do so: the tangents carrying an input's influence forward,
and the adjoints carrying the output's sensitivity back. [`cover_lattice`](@ref) computes the
cover with all four combinations of signed and absolute channels, so each one's cost on a given
program can be read off separately:

```jldoctest
julia> g(x) = (d = exp(x[1]) - 1 - x[1]; u = d * x[2]; exp(u) - u);

julia> lat = cover_lattice(g, [0.4, 1.5]);

julia> round.(lat.h_quad_signed ./ lat.h_quad; digits=3)   # what the tangent signs buy
2-element Vector{Float64}:
 0.222
 1.0

julia> round.(lat.h_quad ./ lat.h_quad_absadj; digits=3)   # what the adjoint signs buy
2-element Vector{Float64}:
 0.362
 0.364
```

Here `d` cancels along the two paths from `x[1]`, and `u` reaches the output through arms of
opposite adjoint sign, so both channels have something to lose.

The [Diagnostics](@ref) module can perturb a cover in controlled ways and measure the effect on
downstream computations. The companion `DerivativeCoverProblems` package provides the
[Problem families](@ref) used in these experiments.

## How this documentation is organized

The reference is divided by API status:

- This page lists exported names.
- [Quantities and extension points](@ref) lists public, unexported names. These are part of the
  supported API and require the `DerivativeCovers.` prefix.
- [Internals](@ref) lists implementation details, which may change without notice.

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
cover_lattice
```

## Selecting what `cover_hessian` and `cover_nls` return

```@docs
CoverOnly
WithDiagnostics
```

## Result types

```@docs
HessianCover
HessianDiagnostics
CoverLattice
JacobianCover
NLSCover
NLSDiagnostics
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
