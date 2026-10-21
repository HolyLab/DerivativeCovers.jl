```@meta
CurrentModule = DerivativeCovers
```

# Programs and new operations

## Writing a program

The functions that take a program `f` and a point `x0` call `f` on a vector of
[`TVar`](@ref)s, which record each operation on a [`Tape`](@ref) (or, for
[`forward_cover`](@ref), on a vector of [`FVar`](@ref)s, which propagate the
cover as the program runs). `f` must therefore be written generically, and use
only the supported operations:

- arithmetic: `+`, `-`, `*`, `/`, `inv`, and `^` (with a constant or variable
  exponent);
- functions: `abs`, `abs2`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `tanh`,
  and `atan`;
- array operations: `sum`, `dot`, and matrix-vector and matrix-matrix
  products, which record one operation per entry of the result rather than a
  sequence of scalar operations.

Constants, including typed literals such as `zero(eltype(x))`, are folded into
the operations that use them and take no space on the tape.

Comparisons are not defined for `TVar`. To branch on the value of a variable,
compare its underlying value, obtained with [`value`](@ref):

```jldoctest program
julia> using DerivativeCovers

julia> f(x) = DerivativeCovers.value(x[1]) > 0 ? x[1]^2 * x[2] : -x[1] * x[2];

julia> round.(cover_hessian(f, [1.0, 2.0]).h_sum; digits=4)
2-element Vector{Float64}:
 4.8284
 0.7071
```

The cover then describes the branch taken at `x0`.

## Recording a tape

[`tape_function`](@ref) records a program and returns the tape and the index
of the output node (or a vector of indices, for a vector-valued program). The
tape can also be recorded by hand:

```jldoctest program
julia> t = Tape([0.5, 2.0]);

julia> x = [TVar(t, i) for i in 1:t.ninputs];

julia> y = exp(x[1]) + x[1] * x[2]^2;

julia> DerivativeCovers.nnodes(t)
6
```

The element type of the tape, `eltype(t)`, sets the precision of the whole
computation. It defaults to `float(eltype(x0))`; use `Tape{Float32}(x0)` to
record in another precision.

## Adding an operation

A new scalar operation is added by computing its value and its first and
second derivatives and passing them to one of the recording functions
[`unarynode`](@ref) or [`binarynode`](@ref). Defining the method for
[`CoverVar`](@ref), the supertype of [`TVar`](@ref) and [`FVar`](@ref), makes
the operation available to both the tape and [`forward_cover`](@ref):

```jldoctest program
julia> function Base.log1p(x::DerivativeCovers.CoverVar)
           v = DerivativeCovers.value(x)
           return DerivativeCovers.unarynode(x, log1p(v), 1 / (1 + v), -1 / (1 + v)^2)
       end;

julia> g(x) = log1p(x[1] * x[2]);

julia> round.(cover_hessian(g, [0.5, 2.0]).h_sum; digits=4)
2-element Vector{Float64}:
 2.4142
 0.6036
```

Operations with more than two arguments and structured second derivatives
are recorded with [`pushnode!`](@ref), which takes the second derivatives as a
[`Curvature`](@ref). The docstring of [`Curvature`](@ref) lists the methods a
new form must implement. [`dotnode`](@ref) and [`linearnode`](@ref) record
inner products and linear combinations.

## Declaring a sub-map

A part of the program can be recorded as a single operation per output, with
a declared bound on its second derivatives in place of its internal
operations. [`submap`](@ref)`(F, xs)` evaluates `F` on the arguments `xs`,
computes its exact Jacobian and a cover of the Hessian of each of its outputs,
and records the result on the tape of `xs`:

```julia
outer(x) = (y = DerivativeCovers.submap(F, [x[1] + x[2], x[3]]); y[1] * y[2] + x[1])
```

[`declare_submap`](@ref) computes the declaration without recording it, and
[`submapnode`](@ref) records a declaration supplied by the caller. Covers
computed from such a tape are valid provided that the declaration is.
Because the exact second derivatives are not recorded, [`hvp`](@ref) is not
available on a tape that contains a declared sub-map.

## Combining covers of pieces

When a program is a composition `F₂ ∘ F₁` and covers of the two pieces are
available separately, [`compose_jacobian_cover`](@ref) combines Jacobian
covers of `F₁` and `F₂` into a cover of the Jacobian of the composition, and
[`compose_hessian_cover`](@ref) combines covers of the derivatives of `F` and
`g` into a cover of the Hessian of `g ∘ F`:

```jldoctest compose
julia> using DerivativeCovers

julia> F1(x) = [x[1] * x[2], x[1] + x[2]];

julia> F2(y) = [exp(y[1]) * y[2]];

julia> x0 = [0.5, 2.0];

julia> y0 = [x0[1] * x0[2], x0[1] + x0[2]];

julia> c1 = cover_jacobian(F1, x0); c2 = cover_jacobian(F2, y0);

julia> a, b = DerivativeCovers.compose_jacobian_cover(c2.a, c2.b_sum, c1.a, c1.b_sum);

julia> J = jvp(x -> F2(F1(x)), x0, [1.0, 0.0])[1], jvp(x -> F2(F1(x)), x0, [0.0, 1.0])[1];

julia> all(abs.(collect(J)) .<= a[1] .* b)
true
```

This requires no joint recording of the two pieces, so the pieces can be
covered by different means or over different regions (see
[Covers over a box](@ref)).
