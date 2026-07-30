# Cover-tightness measurements

Scripts that measure how loose the covers are, scored against a reference Hessian from
ForwardDiff. Tightness, not run time: `1/cover_ratio(H, h)` is the factor by which a cover
exceeds the largest true Hessian entry, so `1` is exact and larger is looser.

The environment is not registered. Point it at the working copies before the first run:

```julia
using Pkg
Pkg.activate("bench")
Pkg.develop([PackageSpec(path="DerivativeCovers"), PackageSpec(path=<path to MatrixCovers>)])
```

## `declaration_penalty.jl`

Compares recording an array op two ways: decomposed into scalar elementals, or as a single
[`submap`](../DerivativeCovers/src/compose.jl) node carrying an exact Jacobian and a *declared*
curvature bound. The declaration is the only route available to an op whose within-operand
curvature is a dense block larger than 2x2, so the comparison measures what the certified cover
vocabulary costs when an op falls outside it.

Two channels move in opposite directions and the script separates them. Recording the op as one
node makes its first-order partials exact, removing the path-cancellation loss that an
absolute-value sweep suffers on the decomposed form; it also collapses the op's curvature to one
rank-1 bound per output. The head determines which channel dominates: a head linear in the op's
output isolates the curvature channel, while a nonlinear head brings in the `J'∇²g J` term where
exact partials pay.

Two controls run alongside the sweep:

- a scalar-output op, where `declare_submap` reproduces the decomposed sum merge exactly, so the
  two forms must agree to roundoff;
- softmax's first-order loss, which is `1` off the diagonal and `(1+p_i)/(1-p_i)` on it.
