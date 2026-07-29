# Rank-1 Hessian covers computed from a tape.
#
# A symmetric cover of a symmetric matrix H is a nonnegative vector h with |H[i,j]| <= h[i]h[j].
# The construction rests on the exact identity
#
#     H = sum_k vbar_k * sum_{i,j in P_k} (d²φ_k/dv_i dv_j) * ∇v_i (∇v_j)',
#
# Here vbar_k is the reverse-mode adjoint of node k and ∇v_i is an intermediate's input
# gradient. Taking absolute values gives B >= |H|. Each node contributes a rank-1 cover of its
# term in B; the merge determines how those local covers are combined.
#
# An inner-product node contributes one atom per independent parent pair. A linear node has no
# second partials and contributes no atoms.

"""
    HessianCover

The standard result returned by [`cover_hessian`](@ref).

- `grad`: gradient of the scalar function at the evaluation point.
- `h_sum`: certified Hessian cover computed with the sum merge.
"""
struct HessianCover
    grad::Vector{Float64}
    h_sum::Vector{Float64}
end

"""
    HessianDiagnostics

The extended result returned by `cover_hessian(WithDiagnostics(), ...)`.

- `grad`: gradient of the scalar function.
- `B`: explicit elementwise bound satisfying `B >= abs.(H)`.
- `h_sum`: cover computed with the sum merge.
- `h_quad`: tighter cover computed by combining cover atoms in quadrature.
- `h_sum_rev`: independent reverse-sweep computation of `h_sum`, used as a consistency check.

Forming these diagnostics costs `O(n * L)` time and `O(n^2)` memory.
"""
struct HessianDiagnostics
    grad::Vector{Float64}
    B::Matrix{Float64}
    h_sum::Vector{Float64}
    h_quad::Vector{Float64}
    h_sum_rev::Vector{Float64}
end

"""
    adjoints(t::Tape, outidx::Integer) -> vbar

Reverse-mode adjoints `vbar[k] = ∂v_outidx/∂v_k` for every node of `t`. The gradient of the
program with respect to its inputs is `vbar[1:t.ninputs]`.
"""
function adjoints(t::Tape, outidx::Integer)
    N = nnodes(t)
    vbar = zeros(N)
    vbar[outidx] = 1.0
    # Reverse topological order finalizes each adjoint before propagating it.
    for k in N:-1:1
        vk = vbar[k]
        vk == 0 && continue
        for (i, p) in enumerate(t.parents[k])
            vbar[p] += t.ds[k][i] * vk
        end
    end
    return vbar
end

"""
    adjoints(t::Tape, outidxs::AbstractVector{<:Integer}) -> Vbar

Vector-mode reverse-mode adjoints for several outputs at once: `Vbar[k, j] = ∂v_{outidxs[k]}/∂v_j`,
every output `k` against every node `j`, all propagated in one traversal. The `m × n` Jacobian of
the outputs with respect to the inputs is `Vbar[:, 1:t.ninputs]`.

Each output owns a row seeded independently, so a repeated entry in `outidxs` yields identical rows
and an input-node entry a unit row. Costs `O(m*L)` time and memory: a diagnostic path, like
[`sensitivities`](@ref), for the exact signed Jacobian; the production cover merges seed
[`abs_reverse`](@ref) at the outputs instead and stay `O(L)`/`O(G*L)`.
"""
function adjoints(t::Tape, outidxs::AbstractVector{<:Integer})
    m, N = length(outidxs), nnodes(t)
    Vbar = zeros(m, N)
    for (k, o) in enumerate(outidxs)
        Vbar[k, o] += 1.0
    end
    # Reverse topological order finalizes each adjoint column before propagation.
    for j in N:-1:1
        for (i, p) in enumerate(t.parents[j])
            d = t.ds[j][i]
            d == 0 && continue
            for k in 1:m
                Vbar[k, p] += d * Vbar[k, j]
            end
        end
    end
    return Vbar
end

"""
    sensitivities(t::Tape) -> utilde

Forward absolute sensitivities: `utilde[k][a] >= abs(∂v_k/∂x_a)`, obtained by propagating
`utilde[k] = sum_{i in P_k} abs(∂φ_k/∂v_i) * utilde[i]` from `utilde[x_a] = e_a`. The bound
follows from the triangle inequality over paths, and is tight when no path cancellation occurs.

Costs `O(n * L)` time and memory, so this is a diagnostic path: the sum merge is obtainable
without it (see [`abs_reverse`](@ref)).
"""
function sensitivities(t::Tape)
    n, N = t.ninputs, nnodes(t)
    utilde = [zeros(n) for _ in 1:N]
    for a in 1:n
        utilde[a][a] = 1.0
    end
    for k in (n + 1):N
        uk = utilde[k]
        for (i, p) in enumerate(t.parents[k])
            ad = abs(t.ds[k][i])
            ad == 0 && continue
            up = utilde[p]
            for q in 1:n
                uk[q] += ad * up[q]
            end
        end
    end
    return utilde
end

"""
    hessian_bound(t::Tape, vbar, utilde) -> B

The elementwise bound `B = sum_k abs(vbar[k]) * sum_{i,j} abs(C^k[i,j]) * utilde_i * utilde_j'`,
which satisfies `B >= abs.(H)` entrywise. Diagnostic only: `B` is `n × n`, and the two
inequalities it discards — cancellation along paths (inside each `utilde`) and cancellation
across elementals — are exactly the looseness this bound trades for its cost.

A node with an empty `C` (see [`Tape`](@ref)) contributes nothing and is skipped. A
[`PairedPartials`](@ref) node has only `2q` nonzero entries out of `(2q)^2`, so its contribution
is accumulated directly from those pairs rather than by scanning the dense `C`. A
[`CoveredPartials`](@ref) node contributes its declared bound `s[i]*s[j] >= abs(C[i, j])`, so
`B >= abs.(H)` remains valid conditional on the declaration; the bound factorizes, so the
contribution is the rank-1 term `w*w'` with `w = sum_i s[i]*utilde[ps[i]]`.
"""
function hessian_bound(t::Tape, vbar::AbstractVector, utilde::AbstractVector)
    n, N = t.ninputs, nnodes(t)
    B = zeros(n, n)
    for k in (n + 1):N
        av = abs(vbar[k])
        av == 0 && continue
        ps, C = t.parents[k], t.Cs[k]
        if C isa CoveredPartials
            w = zeros(n)
            for (i, p) in enumerate(ps)
                si = C.s[i]
                si == 0 && continue
                up = utilde[p]
                for q in 1:n
                    w[q] += si * up[q]
                end
            end
            for q in 1:n, r in 1:n
                B[q, r] += av * w[q] * w[r]
            end
            continue
        end
        isempty(C) && continue
        if C isa PairedPartials
            qh = C.q
            for j in 1:qh
                ui, uj = utilde[ps[j]], utilde[ps[qh + j]]
                for q in 1:n, r in 1:n
                    B[q, r] += av * (ui[q] * uj[r] + uj[q] * ui[r])
                end
            end
            continue
        end
        for i in eachindex(ps), j in eachindex(ps)
            cij = av * abs(C[i, j])
            cij == 0 && continue
            ui, uj = utilde[ps[i]], utilde[ps[j]]
            for q in 1:n, r in 1:n
                B[q, r] += cij * ui[q] * uj[r]
            end
        end
    end
    return B
end

"""
    balanced_split(c11, c12, c22, z1, z2) -> (rho1, rho2)

Rank-1 cover of an off-diagonal-active 2×2 block `[c11 c12; c12 c22]`, which requires
`c12^2 > c11*c22`. Returns `rho1 >= sqrt(c11)` and `rho2 >= sqrt(c22)` with
`rho1*rho2 == c12`: the cover is exactly tight on the off-diagonal, so no cover of the block is
smaller in both entries.

Those constraints leave a one-parameter family. The scale references `z1`, `z2` — one per
parent, supplied by a [`SplitRule`](@ref): the parent value magnitudes under
[`ValueSplit`](@ref), the weighted absolute sensitivities under [`SensitivitySplit`](@ref) —
pick a member, giving `rho_i` the units of `sqrt(f)/[v_{p_i}]` and making the cover
scale-covariant. The result is continuous in all five arguments except where a vanishing
scale reference meets a degenerate block; see the extended help.

# Extended help

The three constraints confine `rho1` to the segment `[sqrt(c11), c12/sqrt(c22)]` — nonempty
precisely because `c12^2 > c11*c22` — and fix `rho2 = c12/rho1`. Every returned cover lies on
this segment, which is what makes it entrywise unimprovable: a competitor smaller in both
entries would have `rho1*rho2 < c12`.

The position along the segment is the point that equalizes the value-weighted scales,
`rho1*z1 == rho2*z2`, clamped to the segment when that point falls outside. Covariance follows:
under a rescaling `v_i -> d_i*v_i` of the intermediate variables, `c_ij -> c_ij/(d_i*d_j)` and
`z_i -> d_i*z_i`, so both endpoints and the equalizing point scale as `1/d_1`; `clamp` is
positively homogeneous, so `rho_1 -> rho_1/d_1` and `rho_2 -> rho_2/d_2`.

Clamping is also what keeps the result continuous as a parent value vanishes: the equalizing
point runs off to an endpoint, and the endpoint absorbs it. The `z1 == 0` and `z2 == 0` arms
therefore take the limits of that point — the upper endpoint and the lower endpoint
respectively — rather than a separate fallback formula.

That reasoning needs the endpoint being approached to be a positive finite scale, and it is
not when the corresponding diagonal entry vanishes: `c22 == 0` puts the upper endpoint at
infinity, `c11 == 0` puts the lower endpoint at zero. Nor is there any limit to take when both
values vanish at once, since the equalizing point then depends on the ratio `z2/z1` alone. In
those cases the geometric mean of the endpoints is used, which is covariant and
permutation-equivariant; when it too is degenerate the balanced `sqrt(c12)` is returned.

So the result is continuous except at three families of inputs, where it is the ratio `z2/z1`
that carries the information and the ratio is unavailable:

  - `z1 == 0` with `c22 == 0`,
  - `z2 == 0` with `c11 == 0`,
  - `z1 == z2 == 0`, for any block.

A multiplication node with a vanishing operand has `c11 == c22 == 0` and meets all three at
once. Among the elementals of `overloads.jl` it is the only source of the first and third:
`c22 == 0` otherwise requires `x^y` at `x == 1`, and `^` rejects a nonpositive base, so it
never presents a vanishing `z1`. The second is also reached by `x^y` at a zero exponent, where
`c11 == 0` and `z2 == 0` while `c22 > 0`.

These zero-reference arms concern the value channel: under [`SensitivitySplit`](@ref) a zero
scale certifies a dead slot and [`local_cover_slots`](@ref) resolves it before this function
is reached, so this function only ever receives positive references from that rule.
"""
function balanced_split(c11, c12, c22, z1, z2)
    lo = sqrt(c11)
    hi = c22 > 0 ? c12 / sqrt(c22) : oftype(lo, Inf)
    target = if z1 > 0 && z2 > 0
        sqrt(c12 * z2 / z1)
    elseif z2 > 0
        hi
    elseif z1 > 0
        lo
    else
        oftype(lo, NaN)
    end
    # Clamping resolves zero or infinite limiting targets when the feasible interval has a finite
    # endpoint. Only an unresolved NaN or unbounded target reaches the fallback.
    rho1 = clamp(target, lo, hi)
    if !(isfinite(rho1) && rho1 > 0)
        g = sqrt(lo * hi)
        rho1 = clamp((isfinite(g) && g > 0) ? g : sqrt(c12), lo, hi)
    end
    return rho1, c12 / rho1
end

"""
    SplitRule

Policy for choosing among the valid rank-1 covers of a single tape node.

Some 2×2 second-derivative blocks admit a one-parameter family of covers. A `SplitRule` supplies
the scale used to choose one member of that family. The built-in rules are [`ValueSplit`](@ref),
the default, and [`SensitivitySplit`](@ref). Pass a rule with the `rule` keyword to cover
functions. The choice can affect tightness, but not validity.
"""
abstract type SplitRule end

"""
    ValueSplit()

Choose local covers using the magnitudes of their parent values. This is the default
[`SplitRule`](@ref).

The rule tries to equalize `rho1 * abs(v₁)` and `rho2 * abs(v₂)`, subject to the feasible range.
At nodes whose relevant parent values are zero, the cover remains valid but may vary
discontinuously with the evaluation point. Use [`SensitivitySplit`](@ref) to avoid this
value-based discontinuity.
"""
struct ValueSplit <: SplitRule end

"""
    SensitivitySplit(w=nothing)

The [`SplitRule`](@ref) whose scale references are input-weighted absolute sensitivities:
`sigma_j` is the [`abs_forward`](@ref) sweep seeded with the weights `w` at the inputs, so
`sigma_j >= sum_a w[a] * abs(∂v_j/∂x_a)` with equality when no path cancellation occurs.
`w = nothing` means unit weights; otherwise `w` must hold one strictly positive finite entry
per input.

The split equalizes `rho1*sigma_1 == rho2*sigma_2`, clamped to the feasible segment. This is
the exact minimizer of the weighted sum-merge objective `sum_a w[a]*h_sum[a]` over all
per-node splits: the sum merge is linear in the per-node weights, so the objective decouples
into one term `rho1*sigma_1 + rho2*sigma_2` per node, minimized on the feasible segment by
exactly this clamped equalization. `w` is therefore the statement of *which* cover the rule
optimizes — the weighted sum of the input bounds — not a tuning knob.

A zero scale is a certificate here, not a gap: `sigma_j == 0` implies every path from the
inputs to `v_j` carries a zero derivative product, so the total gradient of `v_j` vanishes
identically and every Hessian contribution through that slot is exactly zero. Such a *dead*
slot receives no cover weight and the surviving slot covers only its own diagonal; see
[`local_cover_slots`](@ref). This certificate is what requires `w > 0`: a zero weight could
zero the scale of a live slot and mark it dead falsely.

Covariance: under a rescaling of the intermediate variables the sigmas transform exactly as
the values do, so the covariance argument of [`balanced_split`](@ref) applies unchanged.
Under an input rescaling `x -> D*x` the covers remain exactly covariant when the weights are
rescaled alongside, `w -> D*w`; with `w` held fixed (the unit default included) the *choice*
of split — never its validity — depends on the units the inputs are expressed in.

Cost: one extra `O(L)` forward sweep per cover.
"""
struct SensitivitySplit{W} <: SplitRule
    w::W

    function SensitivitySplit{W}(w) where {W}
        w === nothing || all(wi -> isfinite(wi) && wi > 0, w) ||
            throw(
            ArgumentError(
                "SensitivitySplit weights must be strictly positive and finite: " *
                    "a zero weight could zero a live slot's scale and falsely " *
                    "certify it dead"
            )
        )
        return new{W}(w)
    end
end
SensitivitySplit(w) = SensitivitySplit{typeof(w)}(w)
SensitivitySplit{W}() where {W} = SensitivitySplit{W}(nothing)
SensitivitySplit() = SensitivitySplit(nothing)

"""
    certifies_dead(rule::SplitRule) -> Bool

Whether a zero scale reference from `rule` certifies that the slot's total gradient vanishes
identically, so the slot's Hessian contributions are zero and it needs no cover weight. A zero
*value* certifies nothing of the sort, so [`ValueSplit`](@ref) answers `false`; a zero
absolute sensitivity does, so [`SensitivitySplit`](@ref) answers `true`.

With [`split_scales`](@ref), this is the pair of methods a new `SplitRule` must define.
"""
certifies_dead(::ValueSplit) = false
certifies_dead(::SensitivitySplit) = true

# Lazy absolute-value view used by ValueSplit, avoiding an O(L) copy.
struct AbsValues <: AbstractVector{Float64}
    vals::Vector{Float64}
end
Base.size(a::AbsValues) = size(a.vals)
Base.IndexStyle(::Type{AbsValues}) = IndexLinear()
Base.getindex(a::AbsValues, i::Int) = abs(a.vals[i])

"""
    split_scales(rule::SplitRule, t::Tape) -> scales

The per-node scale references `rule` supplies to [`local_cover_slots`](@ref): a lazy view of
`abs.(t.vals)` for [`ValueSplit`](@ref), one [`abs_forward`](@ref) sweep seeded with the input
weights for [`SensitivitySplit`](@ref). Obtained once per cover sweep and shared by every node.
"""
split_scales(::ValueSplit, t::Tape) = AbsValues(t.vals)

function split_scales(rule::SensitivitySplit, t::Tape)
    theta = zeros(nnodes(t))
    w = rule.w
    if w === nothing
        fill!(view(theta, 1:t.ninputs), 1.0)
    else
        Base.require_one_based_indexing(w)
        length(w) == t.ninputs ||
            throw(DimensionMismatch("SensitivitySplit weights have length $(length(w)), expected $(t.ninputs) (one per input)"))
        copyto!(view(theta, 1:t.ninputs), w)
    end
    return abs_forward(t, theta)
end

"""
    local_covers(t::Tape, vbar; check::Bool=true) -> (rhos, theta)

Per-node rank-1 covers. For node `k`, `rhos[k][i] >= 0` satisfies
`rhos[k][i] * rhos[k][j] >= c^k[i,j]` where `c^k[i,j] = abs(vbar[k]) * abs(C^k[i,j])`; nodes
with no second-order contribution get an empty `rhos[k]`. `theta[j] = sum_k rhos[k][i]` collects
the weight each node receives as a parent, which seeds the reverse sweep of
[`abs_reverse`](@ref).

A [`PairedPartials`](@ref) node (an inner product, see [`dotnode`](@ref)) has `c = abs(vbar[k])`
on each of its `q` off-diagonal pairs and zero elsewhere, so its `2q×2q` local cover problem
separates into `q` independent 2×2 multiplication-node problems; `rhos[k]` collects the `q`
per-pair splits from [`balanced_split`](@ref), slot-aligned with the node's parents. A node with
an empty `C` (a linear n-ary node, see [`linearnode`](@ref)) contributes no second-order term
and is skipped before the arity check below.

A [`CoveredPartials`](@ref) node (a declared sub-map output, see [`submapnode`](@ref)) brings
its own rank-1 cover: `rhos[k] = sqrt(abs(vbar[k])) * s` satisfies the covering invariant
against the *declared* bound `s[i]*s[j] >= abs(C[i, j])` by construction, so the arity
restriction below does not apply to it. There is no recorded `C` to check the declaration
against, so `check` cannot audit these nodes — validity is conditional on the declaration.

`rule` selects the [`SplitRule`](@ref) that breaks the tie at off-diagonal-active blocks.
`check` verifies the covering invariant at every recorded node.
"""
function local_covers(
        t::Tape, vbar::AbstractVector; check::Bool = true,
        rule::SplitRule = ValueSplit()
    )
    N = nnodes(t)
    scales = split_scales(rule, t)
    rhos = [Float64[] for _ in 1:N]
    theta = zeros(N)
    for k in (t.ninputs + 1):N
        rho = local_cover_slots(rule, t, k, abs(vbar[k]), scales; check)
        rho === nothing && continue
        rhos[k] = rho
        for (i, p) in enumerate(t.parents[k])
            theta[p] += rho[i]
        end
    end
    return rhos, theta
end

"""
    local_cover_slots(t::Tape, k::Integer, av; check::Bool=true) -> Union{Nothing,Vector{Float64}}
    local_cover_slots(rule::SplitRule, t::Tape, k::Integer, av, scales; check::Bool=true)

The rank-1 cover of a single node, slot-aligned with its parents: nonnegative `rho` with
`rho[i]*rho[j] >= av*abs(C^k[i,j])`, where `av` is the absolute adjoint `abs(vbar[k])`. Returns
`nothing` when the node contributes no second-order term (`av == 0`, or an empty `C`). This is
the per-node kernel shared by [`local_covers`](@ref), which accumulates the results into
`(rhos, theta)`, and the [`CoverOnly`](@ref) sweep, which deposits them directly into the
absolute adjoint channel; see [`local_covers`](@ref) for the constructions per node type. `check`
verifies the covering invariant on recorded nodes.

The first form uses [`ValueSplit`](@ref). The second takes any [`SplitRule`](@ref) together
with `scales = split_scales(rule, t)`, computed once by the caller and shared across nodes.

Under a rule for which a zero scale certifies a dead slot ([`SensitivitySplit`](@ref)), a slot
with `scales[p_i] == 0` gets `rho[i] = 0` and its partner covers only its own diagonal. The
covering invariant deliberately does not hold slotwise there: every entry it would cover is
multiplied by a total gradient that vanishes identically, so the bound
`B <= sum_k s_k s_k'` of [`cover_hessian`](@ref) is unaffected.

A node whose covering entries are not finite is reported by
[`check_finite_partials`](@ref) rather than by the invariant test.
"""
local_cover_slots(t::Tape, k::Integer, av::Real; check::Bool = true) =
    local_cover_slots(ValueSplit(), t, k, av, split_scales(ValueSplit(), t); check)

"""
    check_finite_partials(t::Tape, k::Integer, av, c::Tuple)

Throw unless the absolute adjoint `av` of node `k` and its covering entries `c` are all finite,
reporting the node's own value and its parents' values.

The covering invariant is a set of comparisons against `c`, so it cannot fire meaningfully when
`c` is not finite: every comparison against `NaN` is false and no finite `rho` covers an infinite
entry, so the test would reject a split that is not at fault. A non-finite `c` says instead that
the program is not twice-differentiable where it was recorded — a division by zero, or a power,
logarithm or root at its singularity. A non-finite `av` says the same of a node downstream, whose
partials the adjoint has already passed through.
"""
function check_finite_partials(t::Tape, k::Integer, av::Real, c::Tuple)
    (isfinite(av) && all(isfinite, c)) && return nothing
    cause = isfinite(av) ? "node $k has non-finite second partials c=$c" :
        "the absolute adjoint reaching node $k is $av, from a node downstream"
    error(
        "$cause; the program is not twice-differentiable at this point (node value " *
            "$(t.vals[k]), parent values $(Tuple(t.vals[p] for p in t.parents[k])))"
    )
end

function local_cover_slots(
        rule::SplitRule, t::Tape, k::Integer, av::Real,
        scales::AbstractVector; check::Bool = true
    )
    ps, C = t.parents[k], t.Cs[k]
    if C isa PairedPartials
        c = av
        c == 0 && return nothing
        q = C.q
        rho = zeros(2q)
        for j in 1:q
            s1 = scales[ps[j]]
            s2 = scales[ps[q + j]]
            if certifies_dead(rule) && (s1 == 0 || s2 == 0)
                continue   # dead pair: both slots stay zero; see the docstring
            end
            a, b = balanced_split(0.0, c, 0.0, s1, s2)
            if check
                check_finite_partials(t, k, av, (c,))
                # Allow for roundoff accumulated across the tape, not just this comparison.
                rtol = sqrt(eps(float(real(eltype(t.vals)))))
                a * b >= c * (1 - rtol) ||
                    error("local cover invariant violated at node $k, pair $j: rho=($a,$b), c=$c")
            end
            rho[j] = a
            rho[q + j] = b
        end
        return rho
    elseif C isa CoveredPartials
        av == 0 && return nothing
        return sqrt(av) .* C.s
    elseif isempty(C)
        return nothing
    elseif length(ps) == 1
        c11 = av * abs(C[1, 1])
        c11 == 0 && return nothing
        return [sqrt(c11)]
    elseif length(ps) == 2
        c11 = av * abs(C[1, 1])
        c12 = av * abs(C[1, 2])
        c22 = av * abs(C[2, 2])
        (c11 == 0 && c12 == 0 && c22 == 0) && return nothing
        s1, s2 = scales[ps[1]], scales[ps[2]]
        if certifies_dead(rule) && (s1 == 0 || s2 == 0)
            # A dead slot contributes no diagonal or cross term; only its live partner needs
            # cover weight.
            return [s1 == 0 ? 0.0 : sqrt(c11), s2 == 0 ? 0.0 : sqrt(c22)]
        end
        if c12^2 <= c11 * c22
            rho1, rho2 = sqrt(c11), sqrt(c22)
        else
            rho1, rho2 = balanced_split(c11, c12, c22, s1, s2)
        end
        if check
            check_finite_partials(t, k, av, (c11, c12, c22))
            rtol = sqrt(eps(float(real(eltype(t.vals)))))
            (
                rho1 >= sqrt(c11) * (1 - rtol) && rho2 >= sqrt(c22) * (1 - rtol) &&
                    rho1 * rho2 >= c12 * (1 - rtol)
            ) ||
                error("local cover invariant violated at node $k: rho=($rho1,$rho2), c=($c11,$c12,$c22)")
        end
        return [rho1, rho2]
    else
        error("node $k has $(length(ps)) parents; only unary and binary elementals are supported")
    end
end

"""
    abs_reverse(t::Tape, theta) -> abar

One reverse sweep in absolute-value arithmetic:
`abar[j] = theta[j] + sum_{k : j in P_k} abs(∂φ_k/∂v_j) * abar[k]`.

With `theta` from [`local_covers`](@ref), `abar[1:t.ninputs]` is the sum-merge cover — the whole
merge in a single gradient-cost sweep, because `sum_k s_k = sum_j theta[j] * utilde_j` is linear
in the seeds.
"""
function abs_reverse(t::Tape, theta::AbstractVector)
    abar = collect(float.(theta))
    for k in nnodes(t):-1:1
        ak = abar[k]
        ak == 0 && continue
        for (i, p) in enumerate(t.parents[k])
            abar[p] += abs(t.ds[k][i]) * ak
        end
    end
    return abar
end

"""
    atom_slots(C, nslots) -> atoms

The groups of a node's parent slots that the quadrature merge of [`cover_hessian`](@ref)
combines separately. A dense `C` (unary or binary) forms one atom spanning every slot. A
[`PairedPartials`](@ref)`(q)` forms `q` atoms, one per pair `(j, q+j)`, matching the
block-diagonal structure of its second partials — the atoms it contributes are independent
rank-1 terms, not one combined term for the whole node.
"""
atom_slots(C, nslots) = (1:nslots,)
atom_slots(C::PairedPartials, nslots) = ((j, C.q + j) for j in 1:C.q)

"""
    CoverOnly()

Return only the gradient and sum-merge cover from [`cover_hessian`](@ref). This is the default
mode and costs `O(L)`.
"""
struct CoverOnly end

"""
    WithDiagnostics()

Return a [`HessianDiagnostics`](@ref), including the elementwise bound, quadrature merge, and an
independent consistency check. This mode costs `O(n * L)` time and `O(n^2)` memory.
"""
struct WithDiagnostics end

const CoverMode = Union{CoverOnly, WithDiagnostics}

"""
    cover_hessian([mode], t::Tape, outidx::Integer; check::Bool=true, rule=ValueSplit())
    cover_hessian([mode], f, x0::AbstractVector; check::Bool=true, rule=ValueSplit())

Compute a certified rank-1 cover of a scalar function's Hessian.

Pass either a tape and scalar output node, or a function and evaluation point. The default
[`CoverOnly`](@ref) mode returns a [`HessianCover`](@ref) containing the gradient and the
sum-merge cover. [`WithDiagnostics`](@ref) additionally computes an explicit elementwise bound
and a tighter quadrature cover.

The `rule` keyword chooses how local covers are scaled; see [`SplitRule`](@ref). Every rule
produces a valid cover. `check=true` verifies the local covering conditions.

Two merges are available:

- **sum merge**, `h_sum = sum_k s_k`: the cross terms of `(sum_k s_k)(sum_k s_k)'` are
  nonnegative slack;
- **quadrature merge**, `h_quad[i] = sqrt(sum_atoms s_atom[i]^2)`: valid by Cauchy–Schwarz, and
  the minimal cover of the no-cancellation surrogate `M = sum_atoms s_atom s_atom'`. An atom (see
  [`atom_slots`](@ref)) is a whole node for unary and binary elementals and one parent pair for
  an inner-product node, so an inner-product node contributes `q` separate terms to the
  quadrature merge rather than one.

Both satisfy `abs.(H) <= h * h'`.

# `CoverOnly()`

This mode takes one forward pass and one reverse sweep, or `O(L)` time and memory. The reverse
sweep carries the ordinary adjoint and the absolute cover channel together:

    vbar[p_i] += d_i * vbar[k]                       # signed channel: reverse-mode AD
    abar[p_i] += abs(d_i) * abar[k] + rho_k[i]       # absolute channel: the cover

with `rho_k` from [`local_cover_slots`](@ref). Both channels of node `k` are final when the
sweep reaches it: adjoint flow and `theta` seeds come only from `k`'s consumers, which follow it
on the tape and are processed first. That makes `rho_k` — which needs the finished `vbar[k]` —
computable in place, and `abar[1:ninputs]` equal to `abs_reverse(t, theta)[1:ninputs]` up to
roundoff.

[`SensitivitySplit`](@ref) requires one additional forward sweep.

# `WithDiagnostics()`

This mode returns a [`HessianDiagnostics`](@ref) and costs `O(n * L)` time and `O(n^2)` memory.
With `check=true`, it also verifies that the diagnostic and fused computations agree.

# Examples

```jldoctest
julia> f(x) = exp(x[1]) + x[1] * x[2]^2;

julia> c = cover_hessian(f, [0.5, 2.0]);

julia> round.(c.h_sum; digits=4)
2-element Vector{Float64}:
 4.1125
 2.4142

julia> round.(c.grad; digits=4)
2-element Vector{Float64}:
 5.6487
 2.0

julia> H = [exp(0.5) 4.0; 4.0 1.0];   # the Hessian of f at [0.5, 2.0]

julia> all(abs.(H) .<= c.h_sum * c.h_sum')
true
```
"""
function cover_hessian(
        ::WithDiagnostics, t::Tape, outidx::Integer; check::Bool = true,
        rule::SplitRule = ValueSplit()
    )
    n, N = t.ninputs, nnodes(t)
    vbar = adjoints(t, outidx)
    utilde = sensitivities(t)
    B = hessian_bound(t, vbar, utilde)
    rhos, theta = local_covers(t, vbar; check, rule)

    ssum = zeros(n)             # sum_k s_k
    ssq = zeros(n)              # sum over atoms of s_atom.^2
    sk = zeros(n)
    for k in (n + 1):N
        rho = rhos[k]
        isempty(rho) && continue
        ps = t.parents[k]
        for atom in atom_slots(t.Cs[k], length(ps))
            fill!(sk, 0.0)
            for i in atom
                up = utilde[ps[i]]
                for q in 1:n
                    sk[q] += rho[i] * up[q]
                end
            end
            for q in 1:n
                ssum[q] += sk[q]
                ssq[q] += sk[q]^2
            end
        end
    end
    h_sum = ssum
    h_quad = sqrt.(ssq)
    h_sum_rev = abs_reverse(t, theta)[1:n]

    if check
        # These two computations share most operations, so their roundoff errors are correlated.
        rtol = eps(float(real(eltype(t.vals))))^(3 // 4)
        for q in 1:n
            denom = max(abs(h_sum[q]), 1.0)
            abs(h_sum[q] - h_sum_rev[q]) <= rtol * denom ||
                error("reverse-sweep h_sum mismatch at input $q: forward=$(h_sum[q]) reverse=$(h_sum_rev[q])")
        end
        # local_covers already checked the node invariants.
        pr = cover_hessian(CoverOnly(), t, outidx; check = false, rule)
        for q in 1:n
            gdenom = max(abs(vbar[q]), 1.0)
            abs(vbar[q] - pr.grad[q]) <= rtol * gdenom ||
                error("paired-sweep gradient mismatch at input $q: adjoint=$(vbar[q]) paired=$(pr.grad[q])")
            hdenom = max(abs(h_sum[q]), 1.0)
            abs(h_sum[q] - pr.h_sum[q]) <= rtol * hdenom ||
                error("paired-sweep h_sum mismatch at input $q: forward=$(h_sum[q]) paired=$(pr.h_sum[q])")
        end
    end

    return HessianDiagnostics(vbar[1:n], B, h_sum, h_quad, h_sum_rev)
end

function cover_hessian(
        ::CoverOnly, t::Tape, outidx::Integer; check::Bool = true,
        rule::SplitRule = ValueSplit()
    )
    n, N = t.ninputs, nnodes(t)
    scales = split_scales(rule, t)
    vbar = zeros(N)
    abar = zeros(N)
    vbar[outidx] = 1.0
    for k in N:-1:1
        vk, ak = vbar[k], abar[k]
        ps = t.parents[k]
        rho = local_cover_slots(rule, t, k, abs(vk), scales; check)
        if rho !== nothing
            for (i, p) in enumerate(ps)
                abar[p] += rho[i]
            end
        end
        (vk == 0 && ak == 0) && continue
        ds = t.ds[k]
        for (i, p) in enumerate(ps)
            vbar[p] += ds[i] * vk
            abar[p] += abs(ds[i]) * ak
        end
    end
    return HessianCover(vbar[1:n], abar[1:n])
end

function cover_hessian(
        mode::CoverMode, f, x0::AbstractVector; check::Bool = true,
        rule::SplitRule = ValueSplit()
    )
    t, outidx = tape_function(f, x0)
    return cover_hessian(mode, t, outidx; check, rule)
end

cover_hessian(t::Tape, outidx::Integer; check::Bool = true, rule::SplitRule = ValueSplit()) =
    cover_hessian(CoverOnly(), t, outidx; check, rule)
cover_hessian(f, x0::AbstractVector; check::Bool = true, rule::SplitRule = ValueSplit()) =
    cover_hessian(CoverOnly(), f, x0; check, rule)
