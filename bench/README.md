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

- a scalar-output op sitting directly on the inputs, where `declare_submap` builds its declaration
  from the same curvature the decomposed tape sweeps, so the two forms agree exactly under the sum
  merge. They do not agree under quadrature: a declared node is a single atom, so its quadrature
  merge collapses onto its own sum merge. That gap is the declaration penalty with the first-order
  channel held out;
- softmax's first-order loss, which is `1` off the diagonal and `(1+p_i)/(1-p_i)` on it.

## `scalarization_sensitivity.jl`

Records the same function twice — once through array primitives, once as explicit scalar chains —
and compares how much of the sum-to-quadrature gap
[`cover_hybrid`](../DerivativeCovers/src/merges.jl) reaches on each, at both granularities of
partition.

`cover_hybrid` sums contributions within a group, so what it recovers depends on which
contributions the partition can separate. Dealing cover atoms, as its integer argument does, is
insensitive to the recording: the curves for the two forms coincide, and both reach the atomwise
quadrature merge once the group count matches the atom count. Dealing whole nodes
(`cyclic_partition`) is sensitive to it, because an array primitive carrying curvature holds
several atoms in one node. Node, curvature-node, and atom counts are reported alongside the merges,
since those counts are exact.

A primitive that is linear carries no curvature and costs nothing either way, which is why the
softmax and layernorm heads — whose reductions are linear — lose little under node granularity
while `dot(z, A*z)/2` loses everything.
