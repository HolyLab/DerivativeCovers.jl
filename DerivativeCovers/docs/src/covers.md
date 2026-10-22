```@meta
CurrentModule = DerivativeCovers
```

# Computing covers

## How a Hessian cover is built

A program is a sequence of elementary operations, each computing a new value
`v_k = φ_k(v_{p_1}, v_{p_2}, …)` from earlier values. Differentiating twice
with the chain rule expresses the Hessian of the output as a sum of one term
per operation,

```math
H \;=\; \sum_k \bar v_k \sum_{i,j} C^{k}_{ij}\, \nabla v_{p_i} \nabla v_{p_j}^{\mathsf T},
```

where ``C^k`` holds the second partial derivatives of ``φ_k`` with respect to
its arguments, ``\nabla v_{p_i}`` is the gradient of argument `i` with respect
to the inputs (its *input gradient*), and ``\bar v_k`` is the derivative of the
output with respect to ``v_k`` (the *adjoint*). Operations that are linear in
their arguments, such as `+`, have ``C^k = 0`` and contribute nothing.

DerivativeCovers bounds each term by a rank-one matrix ``s_k s_k^{\mathsf T}``,
built from a cover of the small matrix ``|\bar v_k|\, C^k`` and from
bounds on the magnitudes of the input gradients. These *local covers* are then
combined into a cover of `H`. Two ways of combining them are provided:

- the **sum merge** `h_sum = Σ_k s_k`, which is valid because the cross terms
  of ``(\sum_k s_k)(\sum_k s_k)^{\mathsf T}`` are nonnegative;
- the **quadrature merge** `h_quad[i] = sqrt(Σ_k s_k[i]^2)`, which is valid by
  the Cauchy–Schwarz inequality and never exceeds `h_sum`.

The sum merge is linear in the local covers, so it can be accumulated in a
single reverse sweep over the recorded program, at a cost of ``O(L)`` for a
program of `L` operations. The quadrature merge needs each ``s_k`` as an
explicit vector of length `n`, at a cost of ``O(nL)``.

## Hessian covers

[`cover_hessian`](@ref) takes a function and a point, or a recorded
[`Tape`](@ref) and the index of its output (see
[Programs and new operations](@ref)). By default it returns a
[`HessianCover`](@ref), computed in ``O(L)``:

```jldoctest hess
julia> using DerivativeCovers

julia> f(x) = exp(x[1]) + x[1] * x[2]^2;

julia> x0 = [0.5, 2.0];

julia> c = cover_hessian(f, x0);

julia> round.(c.h_sum; digits=4)
2-element Vector{Float64}:
 4.1125
 2.4142

julia> round.(c.grad; digits=4)
2-element Vector{Float64}:
 5.6487
 2.0
```

Passing [`WithDiagnostics`](@ref) as the first argument returns a
[`HessianDiagnostics`](@ref), which adds the quadrature merge `h_quad` and an
explicit elementwise bound `B ≥ abs.(H)`. These cost ``O(nL)`` time and
``O(n^2)`` memory, so they are not computed by default:

```jldoctest hess
julia> d = cover_hessian(WithDiagnostics(), f, x0);

julia> round.(d.h_quad; digits=4)
2-element Vector{Float64}:
 3.1062
 1.7321

julia> round.(d.B; digits=4)
2×2 Matrix{Float64}:
 1.6487  4.0
 4.0     1.0
```

Here `B` equals `abs.(H)`, because no cancellation occurs in this program.

The default result also includes `h_var`, the root-sum-square of the same
contributions that `h_sum` adds. It is the expected size of the sum if each
contribution carried an independent random sign. **`h_var` is not a certified
cover**: it can fall below the true scale where several contributions share a
sign. It costs nothing extra, and the ratio `h_sum ./ h_var` indicates how
much the contributions to each coordinate overlap. [`cover_variance`](@ref)
returns this ratio along with both vectors.

### Between the two merges

[`cover_hybrid`](@ref) divides the local covers into `G` groups, sums within
each group, and combines the group totals in quadrature. One group gives
`h_sum`; one group per local term gives `h_quad`. The cost is ``O(GL)``, so `G`
trades cost for tightness:

```jldoctest hess
julia> round.(cover_hybrid(f, x0, 1); digits=4)    # the sum merge
2-element Vector{Float64}:
 4.1125
 2.4142

julia> round.(cover_hybrid(f, x0, 3); digits=4)    # the quadrature merge
2-element Vector{Float64}:
 3.1062
 1.7321
```

[`natoms`](@ref) reports the number of local terms on a tape, the group count
at which the quadrature merge is reached.

## Least-squares objectives

For a loss ``f(x) = \tfrac12 \sum_k r_k(x)^2``, [`cover_nls`](@ref) takes the
residual function `r` and returns an [`NLSCover`](@ref) with covers of both the
full Hessian, `h`, and the Gauss–Newton matrix ``J^{\mathsf T}J``, `h_gn`:

```jldoctest nls
julia> using DerivativeCovers

julia> r(x) = [x[1] * x[2] - 1.0, exp(x[1]) - 2.0];

julia> nc = cover_nls(r, [0.5, 2.0]);

julia> round.(nc.h; digits=4)
2-element Vector{Float64}:
 2.7014
 0.5

julia> round.(nc.h_gn; digits=4)
2-element Vector{Float64}:
 2.592
 0.5
```

By default each residual gets its own channel in the reverse sweep, for a cost
of ``O(mL)`` with `m` residuals. The `ngroups` keyword lowers the number of
channels to `G`, and the cost to ``O(GL)``, at some loss of tightness. The
uncertified scales `h_var` and `h_var_gn` are the analogs of `h_var` for
[`cover_hessian`](@ref).

## Jacobian covers

[`cover_jacobian`](@ref) returns a [`JacobianCover`](@ref) with output weights
`a` and input weights `b_sum` satisfying `abs(J[k,j]) ≤ a[k] * b_sum[j]`:

```jldoctest jac
julia> using DerivativeCovers

julia> r(x) = [x[1] * x[2], exp(x[1]) - x[3], x[2] + 2x[3]];

julia> jc = cover_jacobian(r, [0.5, 2.0, -1.0]);

julia> J = [2.0 0.5 0; exp(0.5) 0 -1; 0 1 2];

julia> all(abs.(J) .<= jc.a .* jc.b_sum')
true
```

A Jacobian cover is determined only up to a scalar factor (`a → γ*a`,
`b → b/γ`), and beyond that the choice of `a` is free. The `start` keyword
selects how `a` is chosen: `:covariant` (the default) makes the products
`a[k] * b[j]` transform exactly under rescaling of the inputs and outputs, and
`:equal` treats all outputs alike. The `rounds` keyword sets the number of
refinement rounds. Alternatively, pass a fixed `a`.

For a given `a`, the tightest valid `b` is `b_max[j] = maximum(abs(J[k,j]) / a[k])`
(with `abs(J)` replaced by a bound computed from the program). The default
computes the looser `b_sum` in ``O(L)``.
`cover_jacobian(WithDiagnostics(), …)` also returns `b_max` and an intermediate
`b_quad`, at a cost of ``O(mL)``. [`cover_jacobian_hybrid`](@ref) interpolates
between `b_sum` and `b_max` by grouping the outputs, at a cost of ``O(GL)``.

## Choosing the input-gradient channel

The local covers need bounds on the magnitudes of the input gradients. By
default,
these bounds are accumulated in absolute-value arithmetic, which loses any
cancellation between the paths from an input to an operation. This is the
[`AbsoluteTangents`](@ref) channel, and it is what allows the sum merge to be
computed in a single ``O(L)`` sweep.

The [`SignedTangents`](@ref) channel uses the exact input gradients instead. It
requires the ``O(nL)`` diagnostic mode and gives the tightest cover the package
computes:

```jldoctest channel
julia> using DerivativeCovers

julia> g(x) = (d = sin(x[1]) - x[1]; u = d * x[2]; exp(u) - u);

julia> x0 = [0.7, 1.3];

julia> round.(cover_hessian(WithDiagnostics(), g, x0).h_quad; digits=4)
2-element Vector{Float64}:
 3.1673
 0.0768

julia> round.(cover_hessian(WithDiagnostics(), g, x0; channel=SignedTangents()).h_quad; digits=4)
2-element Vector{Float64}:
 0.4854
 0.0768
```

In this program, `d = sin(x[1]) - x[1]` nearly cancels, and the absolute-value
bound on the input gradient of `d` overestimates the exact one. [Tightness](@ref) describes how to
diagnose such losses.

## Propagating a cover forward

[`forward_cover`](@ref) computes a Hessian cover without recording the
program. Each variable carries its value, its exact input gradient, and a
cover of
its own Hessian; the cover of the output is available when the program
finishes. The memory required is ``O(n)`` for each variable that is live at a
given moment, rather than for every operation of the program, so the method
suits programs too long to record. The cost is ``O(nL)`` time.

```jldoctest channel
julia> round.(forward_cover(g, x0).h_quad; digits=4)
2-element Vector{Float64}:
 2.0471
 0.2928
```

A forward pass cannot know how the output depends on the current variable, so
it bounds the adjoints in absolute-value arithmetic, while keeping the exact
input gradients. The tape-based covers do the opposite, and neither result is tighter
in general. With [`SignedTangents`](@ref), the tape-based cover is tighter than
`forward_cover`.

`forward_cover` stores its channels in arrays of the same type as `x0`. With a
`StaticVector` from StaticArrays, no operation allocates; this is suited to
programs with a handful of inputs, since the code is compiled for the specific
number of inputs.

## Reusing a recording

When several covers or products are needed at the same point, record the
program once with [`tape_function`](@ref) and pass the tape to each function:

```jldoctest tape
julia> using DerivativeCovers

julia> f(x) = exp(x[1]) + x[1] * x[2]^2;

julia> t, out = tape_function(f, [0.5, 2.0]);

julia> c = cover_hessian(t, out);

julia> round.(hvp(t, out, [1.0, 0.0]); digits=4)
2-element Vector{Float64}:
 1.6487
 4.0
```

If the adjoints of the tape have already been computed by other means,
`cover_hessian(CoverOnly(), t, vbar)` computes the cover from them; see
[`adjoints`](@ref).

## Checks and errors

All cover functions accept `check=true` (the default), which verifies each
local cover against the second partial derivatives it bounds. A program that
is not twice differentiable at the evaluation point raises an error that
identifies the offending operation:

```jldoctest errors
julia> using DerivativeCovers

julia> cover_hessian(x -> sqrt(x[1]) + x[2]^2, [0.0, 1.0])
ERROR: node 3 has non-finite second partials c=(Inf,); the program is not twice-differentiable at this point (node value 0.0, parent values (0.0,))
[...]
```
