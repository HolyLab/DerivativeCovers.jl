```@meta
CurrentModule = DerivativeCovers
```

# DerivativeCovers

DerivativeCovers computes rank-one bounds on the magnitudes of the entries of a
program's Hessian or Jacobian, without forming either matrix. For a scalar
function `f` whose Hessian at `x0` is `H`, a *Hessian cover* is a nonnegative
vector `h` satisfying

```math
|H_{ij}| \;\le\; h_i\, h_j \quad \text{for all } i, j.
```

For a vector-valued function whose Jacobian at `x0` is `J`, a *Jacobian cover*
is a pair of nonnegative vectors `a` and `b` satisfying
``|J_{kj}| \le a_k\, b_j`` for all `k`, `j`.

These are the covers of
[MatrixCovers](https://github.com/HolyLab/MatrixCovers.jl), which computes them
from an explicit matrix. Forming a Hessian takes ``O(nL)`` work for a program
with `n` inputs and `L` operations. DerivativeCovers records the program once
and computes a Hessian cover from the recording in ``O(L)`` work, a small
multiple of the cost of the gradient, which it returns as well.

The bound is *certified*: it holds for the exact derivatives at `x0`, up to the
floating-point rounding of the computation. When the program is recorded over
intervals, the bound holds at every point of the box, with no rounding caveat
(see [Covers over a box](@ref)).

## Why covers?

Covers provide the natural scales of an optimization problem. The variables of
a problem often have different units, and many quantities used by optimization
algorithms depend on the choice of those units. Changing the units of the
variables, `x → D*x` for a positive diagonal `D`, changes the Hessian to
`inv(D)*H*inv(D)`, and the cover changes with it, `h → inv(D)*h`. Quantities
formed by dividing by the cover are therefore independent of the units.

Consider fitting a decaying exponential to data. The two parameters, an
amplitude and a time constant in seconds, differ in size by six orders of
magnitude:

```jldoctest decay
julia> using DerivativeCovers, LinearAlgebra

julia> ts = [0.0, 0.001, 0.002, 0.004, 0.008];    # sample times, in seconds

julia> ys = 2000 .* exp.(-ts ./ 0.003);            # amplitude 2000, time constant 3 ms

julia> loss(p) = sum((p[1] * exp(-ts[i] / p[2]) - ys[i])^2 for i in eachindex(ts)) / 2;

julia> p0 = [1800.0, 0.0035];

julia> c = cover_hessian(loss, p0);

julia> round.(c.h_sum; sigdigits=4)
2-element Vector{Float64}:
 3.118
 1.123e6

julia> round.(c.grad; sigdigits=4)
2-element Vector{Float64}:
 -246.8
    3.456e6
```

[`cover_hessian`](@ref) returns the gradient `c.grad` along with the cover
`c.h_sum`. The two entries of each have different units: the first entry of
the gradient is in units of the loss per unit of amplitude, and the second in
units of the loss per second. Expressing the time constant in milliseconds
instead of seconds changes the second entry of both the gradient and the cover
by a factor of 1000, while leaving the problem unchanged:

```jldoctest decay
julia> loss_ms(q) = loss([q[1], q[2] / 1000]);   # time constant in milliseconds

julia> q0 = [1800.0, 3.5];

julia> cms = cover_hessian(loss_ms, q0);

julia> round.(cms.h_sum ./ c.h_sum; sigdigits=4)
2-element Vector{Float64}:
 1.0
 0.001

julia> round.(cms.grad ./ c.grad; sigdigits=4)
2-element Vector{Float64}:
 1.0
 0.001
```

The cover changed exactly as the units did, so the gradient measured relative
to the cover, `g ./ h`, did not change at all:

```jldoctest decay
julia> c.grad ./ c.h_sum ≈ cms.grad ./ cms.h_sum
true
```

Each entry of `g ./ h` has the units of the square root of the loss, whatever
the units of its parameter, so its entries can be compared with one another
and combined into a norm. The
entries of `g` itself cannot: a sum such as `sum(abs2, g)` adds quantities with
different units, and which entry dominates it depends on the choice of units.

The cover was computed without forming the Hessian. For this small problem we
can form it from Hessian-vector products ([`hvp`](@ref)) and confirm the bound:

```jldoctest decay
julia> E = eachcol(Matrix(1.0I, 2, 2));

julia> H = reduce(hcat, [hvp(loss, p0, e) for e in E]);

julia> all(abs.(H) .<= c.h_sum .* c.h_sum')
true
```

The same reasoning applies to the Hessian. Dividing it by the cover gives a
matrix `Hhat` whose entries lie in `[-1, 1]` and do not depend on the units:

```jldoctest decay
julia> Hhat = H ./ (c.h_sum .* c.h_sum');

julia> round.(Hhat; digits=3)
2×2 Matrix{Float64}:
 0.205  0.072
 0.072  0.073

julia> Hms = reduce(hcat, [hvp(loss_ms, q0, e) for e in E]);

julia> Hms ./ (cms.h_sum .* cms.h_sum') ≈ Hhat
true
```

[Using a cover](@ref) describes these and other uses.

## Which function to use

| Function | Bounds | Cost | Certified |
|---|---|---|---|
| [`cover_hessian`](@ref) | Hessian of a scalar function | ``O(L)`` | `h_sum` yes; `h_var` no |
| [`cover_hessian`](@ref)`(WithDiagnostics(), …)` | adds a tighter cover `h_quad` and an elementwise bound | ``O(nL)`` time, ``O(n^2)`` memory | yes |
| [`DerivativeCovers.cover_hybrid`](@ref cover_hybrid) | Hessian, between `h_sum` and `h_quad` | ``O(GL)`` for `G` groups | yes |
| [`forward_cover`](@ref) | Hessian, without recording the program | ``O(nL)`` time, ``O(n)`` memory per live variable | yes |
| [`cover_nls`](@ref) | Hessian and Gauss–Newton matrix of ``\tfrac12\lVert r(x)\rVert^2`` | ``O(GL)``, with `G = m` residual channels by default | `h`, `h_gn` yes; `h_var`, `h_var_gn` no |
| [`cover_jacobian`](@ref) | Jacobian of a vector-valued function | a few ``O(L)`` sweeps | yes |
| [`DerivativeCovers.cover_jacobian_hybrid`](@ref cover_jacobian_hybrid) | Jacobian, tighter than the default | ``O(GL)`` for `G` groups | yes |

For most uses, [`cover_hessian`](@ref) and [`cover_jacobian`](@ref) are the
place to start. The ``O(L)`` cover `h_sum` can be loose when many paths through
the program reconverge; [Computing covers](@ref) describes the options that
trade cost for tightness, and [Tightness](@ref) describes how to measure it.

## Installation

DerivativeCovers is not yet registered. Install it from its subdirectory in the
repository:

```julia
using Pkg
Pkg.add(url="https://github.com/HolyLab/DerivativeCovers.jl", subdir="DerivativeCovers")
```

## Organization of this manual

- [Computing covers](@ref) describes each kind of cover and its cost.
- [Using a cover](@ref) shows what a cover bounds and how it enters numerical
  algorithms.
- [Scale covariance](@ref) states when a cover transforms exactly with the
  units of the inputs.
- [Tightness](@ref) describes how to measure a cover's looseness and where it
  comes from.
- [Covers over a box](@ref) extends the bounds from a point to a box of
  inputs.
- [Programs and new operations](@ref) describes what a program may contain and
  how to add operations.
- The reference pages list the public API:
  [Computing and using covers](@ref), the [Extension interface](@ref), and the
  companion package [DerivativeCoverProblems](@ref). Names that are public but
  not exported, such as `equilibrate`, are called with the `DerivativeCovers.`
  prefix.
