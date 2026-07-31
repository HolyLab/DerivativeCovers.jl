# Cover-tightness measurements

Scripts that measure how loose the covers are. Tightness, not run time. Symmetric covers are scored
against a reference matrix from ForwardDiff — the Hessian, or the Gram matrix `J'*J` — where
`1/cover_ratio(A, h)` is the factor by which a cover exceeds the largest true entry; a rank-1
Jacobian pair is scored against the matrix it covers — the surrogate `Ũ` or the exact `J` — as
*excess* `max_kj (a_k*b_j)/A_kj`. On both scales `1` is exact and larger is looser.

The environment is not registered. `[sources]` points it at the two packages in this repository;
MatrixCovers lives outside it, so supply that working copy before the first run:

```julia
using Pkg
Pkg.activate("bench")
Pkg.develop(PackageSpec(path=<path to MatrixCovers>))
Pkg.instantiate()
```

MatrixCovers is deliberately left out of `[sources]`. A URL there would let `instantiate` succeed
on its own, at the cost of silently measuring a copy other than the one under development —
against the point of these scripts. An unregistered-package error is the better failure. Note that
`[sources]` needs Julia 1.11 or later; on 1.10 it is ignored, and all three packages need the
`develop`.

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

## `factored_submap.jl`

[`submapnode`](../DerivativeCovers/src/tape.jl) holds a dense `q x p` Jacobian and a dense `q x p`
curvature bound, built by one adjoint sweep per output. A traced backend affords neither, so the
production-shaped sub-map is a vjp closure plus a factored cover `(a, b, S)` — which the package
already reaches through [`compose_jacobian_cover`](../DerivativeCovers/src/compose.jl) and
`compose_hessian_cover`, with no joint tape and no dense `J`. The script scores three routes for
one composite: the sub-map decomposed onto the outer tape, declared as `submap` nodes, and
factored.

Factoring is expensive, and the cost grows with the sub-map's width. On the Jacobian of `G(F(z))`
the factored pair is 9.6x looser than the declared one at `n = 8` and 37x at `n = 32` for softmax,
16x and 91x for layernorm. `jacobian_minimax`'s exact rank-1 optimum sits an order of magnitude
below all three routes, so what the factored route gives up is the composition rule, not the rank-1
form it produces.

The penalty has a single source, and a linear head isolates it. A head linear in the sub-map's
outputs has `∇²g = 0`, which deletes the `J'∇²g J` term from the chain rule and leaves only the
declared curvature `S` — the bound both routes carry identically, and the two covers then agree to
the last digit on every fixture. Everything else in the table is the price of bounding `J'∇²g J` by
`(a⋅hg)²*b*b'` instead of using `J`: 1.3x–43x on the sum merge, 1.1x–632x on quadrature.

Quadrature suffers more because the factored route folds the whole first-order term into a single
atom, which is what a declaration does to curvature. The two scripts therefore measure independent
steps toward a production representation, and both are paid mostly in the quadrature merge.

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

## `sign_channels.jl`

What each of a cover's two sign channels is worth. The tangents `∇v_j` carry an input's influence
forward to each node, the adjoints `vbar_k` carry the output's sensitivity back, and either can be
propagated with signs or in absolute-value arithmetic.
[`cover_lattice`](../DerivativeCovers/src/covers.jl) computes all four combinations from one tape,
so the looseness of the shipped cover splits into a tangent-side part, an adjoint-side part, and a
residue neither channel reaches. The script scores the four corners on the
`DerivativeCoverProblems` families and on array-op heads, decomposes the gap per coordinate, sweeps
the reconvergence that creates it, and isolates the residue.

**The two channels are not worth the same.** Keeping tangent signs shrinks the cover by a
per-coordinate geometric mean of 2.47x on `deep w=4` and 1.70x on a two-layer MLP, while keeping
adjoint signs buys 1.17x and 1.15x on the same programs. Adjoint blindness is the cheaper of the
two everywhere measured: the index `acheck_k/abs(vbar_k)` over curvature-carrying nodes has a
geometric mean of 1.00–2.39 across the whole fixture set.

That decides the question the lattice was built to answer. The signed-tangent, absolute-adjoint
corner is exactly the cover a tapeless forward propagation reaches, and it is *not* dominated by
the shipped cover — it wins outright on three of eleven fixtures, by 2.0x on `deep w=4`, 1.6x on
the layernorm head, and 1.2x on the MLP, and loses by 1.4x on the softmax head. Which side wins is
a property of the program, as the incomparability of the two mixed corners predicts, and the
tapeless corner wins where tangent cancellation is the larger of the two.

**Reconvergence is the whole mechanism.** A program whose graph is a tree reaches each node from
each input along one path, so no sign anywhere can cancel and all four corners coincide: at
`sharing = 0` the median looseness is 1.001 in every column. Raising `random_dag`'s sharing over
paired draws carries the adjoint gain from 1.000 to 1.224, rising at every step but one. The
tangent gain shows no such trend, and the `atoms` column says why the endpoints are not
size-matched — tree generation stops early, at a median 10 atoms against 45–49 above zero.

Within the elementwise bound the same split appears on the `sqrt` scale a rank-1 cover lives on,
and it tracks the merge closely: `within-path` reproduces the tangent gain to three digits on the
fixtures where curvature sits on one node per atom (2.465 on `deep w=4`, 1.700 on the MLP) and
exceeds it where an array primitive holds several atoms in one node (1.420 against 1.386 on
layernorm). Cross-elemental loss is the larger of the two everywhere but `deep w=4`, reaching 2.4x
by geometric mean and 12.2x on a single coordinate.

**No corner recovers cancellation between elementals.** The `cancellation` family builds programs
whose two variants share every node value, every first partial, and every adjoint, and differ only
in the sign of the constants multiplying two of each quartet's four squares. Both channels
therefore have identical work to do in both variants, and the Hessian of the cancel variant
collapses onto its rank-1 `delta` term. All four corners report the same looseness there, 91x at
one quartet, 2212x at four, 37390x at sixteen. The triangle inequality across nodes discards those
signs before any channel is chosen, so no amount of sign-keeping inside either channel touches
them.

Seven of eleven fixtures show all four corners equal. Most have nothing to cancel — tree-shaped, or
carrying their whole Hessian on one quadratic node — but `dag shared` is not among those: its
adjoint gain is 1.09x by geometric mean and 1.60x on its worst coordinate, and the corners
nonetheless coincide because the entry that binds its cover is already exact. Channel gains are
per-coordinate; looseness is the worst entry, and the two need not move together.

## `split_rules.jl`

What choosing a [`SplitRule`](../DerivativeCovers/src/covers.jl) is worth, scored in both tangent
channels. Two of the three rules are exact optima of something — `SensitivitySplit` of the sum
merge, `VectorNormSplit` of the quadrature merge — and `ValueSplit`, the default, of nothing: it is
the default because parent values are the only scale reference an `O(L)` sweep already has, where a
sensitivity costs an extra sweep and a tangent norm costs `O(n*L)`.

**A per-node optimum is not a per-cover one, and the objective it optimizes is not the score.**
`VectorNormSplit` minimizes `sum(h_quad.^2)`, and a one-coordinate weight sweep over eight decades
never beat it on any fixture. But looseness is the *worst entry*, and those are different
questions: the norm split wins the worst entry on `quadratic form` (1.80 against 2.78 for the
value split and 2.62 for the sensitivity split) and on both array heads, and loses it on
`quad entrywise` (3.91 against 2.54) and `quad matvec` while still winning the objective. A rule
that provably minimizes a sum can be beaten on a maximum, and here it is.

**The tangent channel moves more than the rule does, where it moves at all.** Keeping the tangent
signs takes `deep w=4` from 16.3 to 2.3 and the two-layer MLP from 3.39 to 1.47, and no rule
comes near that on those programs — all three agree there to the digit, the splits being pinned by
the same block structure. On the fixtures with nothing to cancel along a path (`quad entrywise`,
`quad matvec`, `quad factored`, both `dag` programs, `quadratic form`) the channel changes nothing
at all and the rule is the only handle left. The two choices are close to orthogonal on this
population.

**The value reference has one seam and the table isolates it.** At `(x[1]-1)*(x[2]-1)` with both
operands `1e-8` from zero, the split `ValueSplit` picks is set by the *ratio* of two numbers that
are both a hair from zero: over ratios from 100 to 0.01 its cover swings from `(10, 0.1)` to
`(0.1, 10)`, four orders of magnitude driven by the direction of an arbitrarily small perturbation,
and at the singular point itself it falls back to `(1, 1)`. Both derivative-based references read
`1` on each operand throughout and return `(1, 1)` at every ratio and at the point. That
discontinuity is the whole of what the other two rules buy at a vanishing product, and it is worth
`O(L)` extra for the sensitivity reference and `O(n*L)` for the norm.

## `support_granularity.jl`

[`support_components`](../DerivativeCovers/src/support.jl) calls a tape edge live where its first
partial is nonzero, which is a property of the point the tape was recorded at. A traced backend has
no values at trace time, so the most it can offer is structural support: every recorded edge live.
The script computes components both ways and scores the difference through `gramcover`, which sums
`a_k^2` over the outputs sharing a column's component — merging components enlarges those sums, so
both factors of the bound `abs(G_jk) <= s_j*s_k` inflate.

Coarsening happens exactly where a program's separation lives in the *values* of constant
coefficients rather than in the shape of the graph. Block-diagonal weights record one linear node
per output row carrying every coefficient, zeros included, and only the partials tell the blocks
apart. A banded chain and a program whose halves are written separately give identical components
under either rule. So the measurement bounds what a *dense* array primitive costs; a rule that
propagates a constant matrix's sparsity recovers the block-diagonal case, while nothing recovers
the last one below.

Against the unbalanced pair from `cover_jacobian` the price is exact rather than statistical. `a` is
all ones there, so a component's `a`-mass is its output count and the bound degrades by the factor
by which merging enlarges it: merging 2, 4 and 8 equal blocks costs 2.000, 4.000 and 8.000.

Two consequences do not appear as a looseness ratio. A dead input is found by value liveness and not
by structural, so the exact and minimal `b_j = 0` has no structural counterpart. And liveness is a
property of the point rather than of the program: two halves joined only through a product whose
other factor vanishes separate at that point and nowhere else, so a tape's component count is not
something a pattern representation can reproduce even in principle.
