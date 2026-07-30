# Cover-tightness measurements

Scripts that measure how loose the covers are. Tightness, not run time. The Hessian scripts score
against a reference Hessian from ForwardDiff, where `1/cover_ratio(H, h)` is the factor by which a
cover exceeds the largest true Hessian entry; the Jacobian script scores a rank-1 pair against the
surrogate it covers, as *excess* `max_kj (a_k*b_j)/Ũ_kj`. On both scales `1` is exact and larger is
looser.

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

## `equilibration_rounds.jl`

What the alternation in [`balance_cover`](../DerivativeCovers/src/jacobian.jl) costs and what it
buys. The iteration is unbounded and data-dependent — it runs until the per-round product change
falls below `rtol` and errors when it does not — so its round count belongs with any quoted cost.
Excess puts three references on one scale: the unbalanced pair (`a = ones`, one merge b-step), the
balanced pair, and the exact rank-1 optimum from `jacobian_minimax`, with
`jacobian_representability`'s 2-cycle bound alongside.

The script reports four things. A per-fixture table over all six `(merge, arule)` combinations; a
sweep of `rtol`, the only public handle on where the iteration stops; a survey of convergence
outcome and tightness over random sparse patterns at default settings; and minimal patterns for two
of the failure modes.

Balancing is an answer to *output-scale anisotropy* and to nothing else. Spreading a program's
output scales over four decades moves the unbalanced excess by three orders of magnitude and leaves
the balanced excess where it was, agreeing with the unspread fixture to 11 or more significant
digits and differing by at most one round — the covariance the docstring claims, measured. So the
whole benefit is the scaling it undoes: on the well-scaled fixture balancing buys 1.16x while
switching the sum merge for the max merge buys 4.1x, and on the spread fixture balancing buys
1220x. Against the exact optimum a balanced pair stays loose either way, 1.0x–5.1x here.

Round count and tightness gain are not aligned. The dense fixtures converge in 2–12 rounds; the
banded chain needs over 2400 for the sum merge, overrunning the `:marginal` default of 1000, so
`balance_cover` at its defaults — including the `SumMerge`/`:marginal` combination documented as
the production path — fails outright on a 15-output program. The `rtol` sweep shows that cost to be
avoidable: excess saturates decades of tolerance before `delta` does, and the curve need not be
monotone, since the alternation optimizes its fixed point rather than the excess.

`:marginal` fails on a quarter to a half of random sparse patterns at default settings and is a
median 2.6x–2.8x looser than `:geomean` where both converge. `:geomean` converged on every pattern
drawn, in a median 27 rounds; its round count does not vary with the merge, which enters only in the
closing b-step.

The marginal failures are not failures of the cover. The iterates stay valid throughout, and
tightly so — `jacobian_ratio` sits at exactly 1 on every one of them. Under drift they nonetheless
get monotonically worse: on `[1 1; 0 1]` the sum-merge excess is exactly `2*it + 2` against an
optimum of 1, so a thousand rounds return a cover a thousand times looser than the one available
after one. That is why iterating harder cannot be the remedy, and why drift is reported as drift
rather than left to exhaust the budget. What diverges is the log-spread within each side, which no
regauging touches, and sampling its growth at doubling round counts separates the two families: a
converging run's spread settles, a drifting one's grows by a constant per doubling (logarithmic
drift) or doubles (geometric drift).

That the reliable rule is `:geomean` is awkward, because it is the one that cannot be made cheap.
The tape form runs `O(L)` sweeps without forming `Ũ` only for `SumMerge`/`:marginal`; `:geomean`
needs row sums of `log Ũ`, which no absolute-value sweep produces, so it must materialize the
surrogate at `O(mL)` and then spend `O(mn)` per round. The gap widens with the output count —
2.4x the allocations of the sweep path at `m = 10`, 93x at `m = 640` — which makes `:marginal` the
only equilibration rule that can exist at production tier, and the fragile one.

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
