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

Call [`cover_hessian`](@ref) with a scalar function and an evaluation point:

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

This computation takes one forward pass and one paired reverse sweep: `O(L)` work for a tape
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

A cover loses tightness whenever absolute-value arithmetic prevents cancellation. This can happen
in two channels: tangents carry an input's influence forward, while adjoints carry the output's
sensitivity backward. [`cover_lattice`](@ref) computes covers for all four combinations of signed
and absolute channels, making the cost of each choice visible for a given program:

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

Here `d` cancels along the two paths from `x[1]`, and `u` reaches the output along paths with
opposite adjoint signs, so both channels lose information.

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

## Propagating a cover forward

A cover can also be propagated forward without a tape. Each variable carries its value, its
signed tangent, and a squared cover of its own Hessian; the last quantity directly gives the
output's cover. Working memory is `O(n)` per *live* variable rather than per recorded operation, so
the cost does not grow with the length of the program.

This reaches a specific corner of the sign lattice: every tangent sign is kept and every adjoint
sign is discarded, because a forward pass cannot know how the output depends on its current
variable. This corner is *incomparable* with the result from [`cover_hessian`](@ref), which keeps
adjoint signs and discards tangent signs. Which result is tighter depends on the program.

```@docs
FVar
forward_cover
```

## Cover entry points

At the same `O(n*L)` cost, the tangent channel distinguishes the covers below. By default,
`cover_hessian` merges an absolute-value bound on the tangents, allowing its sum merge to collapse
into a single `O(L)` reverse sweep. [`SignedTangents`](@ref) instead gives the tightest cover the
package computes. It is tighter than the default at every coordinate and discards only
cancellation *between* elementals. Its cost is `O(n*L)` because taking magnitudes inside the merge
leaves nothing for a reverse sweep to collapse.

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
ForwardCover
CoverLattice
JacobianCover
NLSCover
NLSDiagnostics
VarianceCover
```

## Split rules

Some second-derivative blocks admit a family of covers, and a rule selects one member. Two rules
are exact optima: `SensitivitySplit` for the sum merge and `VectorNormSplit` for the quadrature
merge. `ValueSplit` does not optimize either merge; it is the default because its scale reference
is already available during cover computation.

```@docs
SplitRule
ValueSplit
SensitivitySplit
VectorNormSplit
```

## Tangent channels

```@docs
TangentChannel
AbsoluteTangents
SignedTangents
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
