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
    HessianCover{T, M}

The standard result returned by [`cover_hessian`](@ref). `T` is the element type of the
[`Tape`](@ref) it was computed from, and `M =` [`magtype`](@ref)`(T)` the type of the cover.
`HessianCover{T}` stands for `HessianCover{T, magtype(T)}`.

- `grad::Vector{T}`: gradient of the scalar function at the evaluation point.
- `h_sum::Vector{M}`: certified Hessian cover computed with the sum merge.
- `h_var::Vector{M}`: the variance channel, the root-sum-square of the path contributions that
  `h_sum` adds, which models one independent sign per parent slot. This is a scale, **not a
  certified cover**; see [`cover_variance`](@ref).
"""
struct HessianCover{T, M}
    grad::Vector{T}
    h_sum::Vector{M}
    h_var::Vector{M}
end
HessianCover{T}(grad, h_sum, h_var) where {T} = HessianCover{T, magtype(T)}(grad, h_sum, h_var)

"""
    HessianDiagnostics{T, M}

The extended result returned by `cover_hessian(WithDiagnostics(), ...)`. `T` is the element type
of the [`Tape`](@ref) it was computed from, which `grad` holds, and `M =` [`magtype`](@ref)`(T)`
the type of the bound and the covers. `HessianDiagnostics{T}` stands for
`HessianDiagnostics{T, magtype(T)}`.

- `grad`: gradient of the scalar function.
- `B`: explicit elementwise bound satisfying `B >= abs.(H)`.
- `h_sum`: cover computed with the sum merge.
- `h_quad`: tighter cover computed by combining cover atoms in quadrature.
- `h_sum_rev`: the absolute-value reverse sweep's own route to the sum merge, a consistency check.

`B`, `h_sum` and `h_quad` are formed over the [`TangentChannel`](@ref) the caller asked for, so
under [`SignedTangents`](@ref) all three are the signed-tangent quantities. `h_sum_rev` is always
the absolute-value route, which is what makes it independent; it equals `h_sum` in the default
channel and dominates it in the signed one.

Forming these diagnostics costs `O(n * L)` time and `O(n^2)` memory.
"""
struct HessianDiagnostics{T, M}
    grad::Vector{T}
    B::Matrix{M}
    h_sum::Vector{M}
    h_quad::Vector{M}
    h_sum_rev::Vector{M}
end
HessianDiagnostics{T}(grad, B, h_sum, h_quad, h_sum_rev) where {T} =
    HessianDiagnostics{T, magtype(T)}(grad, B, h_sum, h_quad, h_sum_rev)

"""
    CoverLattice{T, M}

Covers of one Hessian computed with every combination of the two sign choices available to a
cover construction, returned by [`cover_lattice`](@ref). `grad` is in the element type `T` of the
[`Tape`](@ref) the lattice was computed from; the bounds and covers are in
`M =` [`magtype`](@ref)`(T)`. `CoverLattice{T}` stands for `CoverLattice{T, magtype(T)}`.

Two channels feed the identity `H = sum_k vbar_k * sum_{i,j in P_k} C^k[i,j] * ∇v_i (∇v_j)'`:
the tangents `∇v_j`, which carry an input's influence forward to each node, and the adjoints
`vbar_k`, which carry the output's sensitivity back. Either can be propagated with signs or in
absolute-value arithmetic. The absolute form is what a construction unable to see those signs
must accept, and it costs tightness wherever the terms it merges would have cancelled.

| tangent \\ adjoint        | signed `abs.(vbar)`             | absolute `acheck`                 |
|:--------------------------|:--------------------------------|:----------------------------------|
| absolute `utilde`         | `h_sum`, `h_quad`               | `h_sum_absadj`, `h_quad_absadj`   |
| signed `abs.(∇v)`         | `h_sum_signed`, `h_quad_signed` | `h_sum_forward`, `h_quad_forward` |

All eight vectors satisfy `abs.(H) <= h * h'`. The absolute-tangent, signed-adjoint corner is
what [`cover_hessian`](@ref) computes. The signed-tangent, absolute-adjoint corner keeps every
sign a forward pass can see and discards exactly those only a reverse sweep supplies, so it is
the cover a single tapeless forward propagation reaches.

- `grad`: gradient, read off the signed forward sweep at the output node.
- `B`, `B_signed`: elementwise bounds satisfying `>= abs.(H)`, built from `utilde` and from
  `abs.(∇v)` respectively. `B >= B_signed`, and the gap is the path cancellation the absolute
  tangent channel discards.
- `h_sum`, `h_quad`: sum and quadrature merges over the four channel pairings, named per the
  table above.

Forming these costs `O(n * L)` time and `O(n^2)` memory.
"""
struct CoverLattice{T, M}
    grad::Vector{T}
    B::Matrix{M}
    B_signed::Matrix{M}
    h_sum::Vector{M}
    h_quad::Vector{M}
    h_sum_absadj::Vector{M}
    h_quad_absadj::Vector{M}
    h_sum_signed::Vector{M}
    h_quad_signed::Vector{M}
    h_sum_forward::Vector{M}
    h_quad_forward::Vector{M}
end
CoverLattice{T}(grad, B, rest...) where {T} = CoverLattice{T, magtype(T)}(grad, B, rest...)

"""
    adjoints(t::Tape, outidx::Integer) -> vbar

Reverse-mode adjoints `vbar[k] = ∂v_outidx/∂v_k` for every node of `t`. The gradient of the
program with respect to its inputs is `vbar[1:t.ninputs]`, and
[`cover_hessian`](@ref)`(CoverOnly(), t, vbar)` computes the Hessian cover from these adjoints
without propagating them again.
"""
function adjoints(t::Tape{T}, outidx::Integer) where {T}
    N = nnodes(t)
    vbar = zeros(T, N)
    vbar[outidx] = 1
    # Reverse topological order finalizes each adjoint before propagating it.
    for k in N:-1:1
        vk = vbar[k]
        iszero(magnitude(vk)) && continue
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
function adjoints(t::Tape{T}, outidxs::AbstractVector{<:Integer}) where {T}
    m, N = length(outidxs), nnodes(t)
    Vbar = zeros(T, m, N)
    for (k, o) in enumerate(outidxs)
        Vbar[k, o] += 1
    end
    # Reverse topological order finalizes each adjoint column before propagation.
    for j in N:-1:1
        for (i, p) in enumerate(t.parents[j])
            d = t.ds[j][i]
            iszero(magnitude(d)) && continue
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
`utilde[k] = sum_{i in P_k} magnitude(∂φ_k/∂v_i) * utilde[i]` from `utilde[x_a] = e_a`. The bound
follows from the triangle inequality over paths, and is tight when no path cancellation occurs.
The entries are in [`boundtype`](@ref) of the tape's element type.

Costs `O(n * L)` time and memory, so this is a diagnostic path: the sum merge is obtainable
without it (see [`abs_reverse`](@ref)).
"""
function sensitivities(t::Tape{T}) where {T}
    n, N = t.ninputs, nnodes(t)
    utilde = [zeros(boundtype(T), n) for _ in 1:N]
    for a in 1:n
        utilde[a][a] = 1
    end
    for k in (n + 1):N
        uk = utilde[k]
        for (i, p) in enumerate(t.parents[k])
            ad = magnitude(t.ds[k][i])
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
    tangents(t::Tape) -> u

Forward signed tangents: `u[k][a] = ∂v_k/∂x_a`, obtained by propagating
`u[k] = sum_{i in P_k} (∂φ_k/∂v_i) * u[i]` from `u[x_a] = e_a`.

The signed counterpart of [`sensitivities`](@ref), which runs the same recursion in absolute-value
arithmetic and so returns a bound `utilde >= abs.(u)` rather than the derivatives themselves. The
gap between the two is the cancellation that occurs along paths from an input to a node.

Costs `O(n * L)` time and memory. Only first partials enter, so a declared sub-map node (see
[`submapnode`](@ref)) propagates exactly like any other.
"""
function tangents(t::Tape{T}) where {T}
    n, N = t.ninputs, nnodes(t)
    u = [zeros(T, n) for _ in 1:N]
    for a in 1:n
        u[a][a] = 1
    end
    for k in (n + 1):N
        uk = u[k]
        for (i, p) in enumerate(t.parents[k])
            d = t.ds[k][i]
            iszero(magnitude(d)) && continue
            up = u[p]
            for q in 1:n
                uk[q] += d * up[q]
            end
        end
    end
    return u
end

"""
    TangentChannel

Which tangents a cover construction merges over: the exact ones, or an absolute-value bound on
them. The built-in channels are [`AbsoluteTangents`](@ref), the default, and
[`SignedTangents`](@ref). Pass one with the `channel` keyword to [`cover_hessian`](@ref).

A channel defines [`tangent_channel`](@ref). The choice affects tightness and cost, but not
validity: every channel dominates `abs.(∇v)` entrywise, which is all the merges require.
"""
abstract type TangentChannel end

"""
    AbsoluteTangents()

The [`TangentChannel`](@ref) that merges over [`sensitivities`](@ref), the absolute-value bound
`utilde >= abs.(∇v)`. This is the default, and the only channel whose sum merge collapses into a
single `O(L)` reverse sweep — the bound is linear in the seeds, so [`abs_reverse`](@ref) can carry
the whole merge (see [`CoverOnly`](@ref)).

What it gives up is cancellation along the paths from an input to a node, which
[`SignedTangents`](@ref) keeps.
"""
struct AbsoluteTangents <: TangentChannel end

"""
    SignedTangents()

The [`TangentChannel`](@ref) that merges over `abs.(∇v)` from [`tangents`](@ref), the exact
gradient magnitudes.

Paired with the signed adjoints every cover already uses, this is the tightest cover of the
`O(n*L)` tier: `abs.(∇v) <= utilde` entrywise, so it dominates the default channel at every
coordinate, and the only cancellation it still discards is the cancellation *between* elementals,
which the triangle inequality over nodes throws away before any channel is chosen.

It costs `O(n*L)` time and memory, and cannot be cheaper: the merge takes magnitudes of each
tangent *inside* the sum, where nothing is linear in the seeds and no reverse sweep collapses it.
So [`CoverOnly`](@ref) rejects this channel rather than quietly becoming `O(n*L)`.
"""
struct SignedTangents <: TangentChannel end

"""
    tangent_channel(ch::TangentChannel, t::Tape) -> u

The per-node tangent channel `ch` supplies to [`merge_atoms`](@ref) and [`hessian_bound`](@ref):
nonnegative vectors satisfying `u[k] >= abs.(∇v_k)` entrywise, which is what makes the merges
cover. [`sensitivities`](@ref) for [`AbsoluteTangents`](@ref), the magnitudes of
[`tangents`](@ref) for [`SignedTangents`](@ref).
"""
tangent_channel(::AbsoluteTangents, t::Tape) = sensitivities(t)
tangent_channel(::SignedTangents, t::Tape) = signed_magnitudes(tangents(t))

# Magnitudes of signed tangents, in [`boundtype`](@ref). Only magnitudes enter the elementwise
# bound and the atoms, so where they share the tangents' type they are reduced in place rather than
# copied, which holds the signed channel to a single `n * L` array.
function signed_magnitudes(u::AbstractVector{<:AbstractVector{T}}) where {T}
    boundtype(T) === T || return [map(x -> tobound(T, magnitude(x)), g) for g in u]
    for g in u
        map!(magnitude, g, g)
    end
    return u
end

"""
    merge_atoms(t::Tape, rhos, u) -> (h_sum, h_quad)

Both merges of the cover atoms, from per-node cover slots `rhos` and a tangent channel `u`. An
atom's cover vector is `s_atom = sum_{i in atom} rhos[k][i] * u[P_k[i]]`, and the merges are
`h_sum = sum_atoms s_atom` and `h_quad[i] = sqrt(sum_atoms s_atom[i]^2)`.

`u[j]` must be nonnegative and dominate `abs.(∇v_j)` entrywise for the results to cover; both
[`sensitivities`](@ref) and the magnitudes of [`tangents`](@ref) qualify. The merges are in
[`boundtype`](@ref) of the tape's element type. Which slots share an atom comes from
[`atom_slots`](@ref), and the loop depends on its partitioning the slots: a slot in no atom would
lose its contribution, one named twice would be counted twice.

Costs `O(n * L)` time and `O(n)` memory beyond `u`.
"""
function merge_atoms(t::Tape{T}, rhos::AbstractVector, u::AbstractVector) where {T}
    n, N = t.ninputs, nnodes(t)
    M = boundtype(T)
    ssum = zeros(M, n)          # sum_k s_k
    ssq = zeros(M, n)           # sum over atoms of s_atom.^2
    sk = zeros(M, n)
    for k in (n + 1):N
        rho = rhos[k]
        isempty(rho) && continue
        ps = t.parents[k]
        for atom in atom_slots(t.Cs[k], length(ps))
            fill!(sk, 0)
            for i in atom
                up = u[ps[i]]
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
    return ssum, sqrt.(ssq)
end

"""
    hessian_bound(t::Tape, vbar, utilde) -> B

The elementwise bound
`B = sum_k magnitude(vbar[k]) * sum_{i,j} magnitude(C^k[i,j]) * utilde_i * utilde_j'`, which
satisfies `B >= abs.(H)` entrywise, in [`boundtype`](@ref) of the tape's element type.
Diagnostic only: `B` is `n × n`, and the two inequalities it discards — cancellation along paths
(inside each `utilde`) and cancellation across elementals — are exactly the looseness this bound
trades for its cost.

Each node's term is accumulated by [`bound_contribution!`](@ref), which every [`Curvature`](@ref)
supplies in the form its structure allows. A [`DeclaredCurvature`](@ref) node contributes its
declared bound `s[i]*s[j] >= abs(C[i, j])`, so `B >= abs.(H)` remains valid conditional on the
declaration.
"""
function hessian_bound(t::Tape{T}, vbar::AbstractVector, utilde::AbstractVector) where {T}
    n, N = t.ninputs, nnodes(t)
    B = zeros(boundtype(T), n, n)
    for k in (n + 1):N
        av = tobound(T, magnitude(vbar[k]))
        av == 0 && continue
        ps = t.parents[k]
        dispatchcurvature(t.Cs[k], T) do C
            bound_contribution!(B, C, av, ps, utilde)
        end
    end
    return B
end

"""
    offdiag_active(c11, c12, c22) -> Bool

Whether the diagonal cover `(sqrt(c11), sqrt(c22))` of the nonnegative block `[c11 c12; c12 c22]`
falls short of its off-diagonal, i.e. whether `c12^2 > c11*c22`, in which case
[`balanced_split`](@ref) is needed.

The test is `c12 > sqrt(c11)*sqrt(c22)`, never the squared form: `c12^2` underflows to zero for
`c12` below the square root of the smallest subnormal (about `1e-162` in `Float64`), which would
misread a block with `c11 == c22 == 0` and a tiny but nonzero cross term as diagonal and cover it
with `(0, 0)`. A long running product of small factors produces exactly such blocks. Square roots
keep every representable entry inside the representable range, so this form neither underflows
nor overflows.
"""
offdiag_active(c11, c12, c22) = c12 > sqrt(c11) * sqrt(c22)

"""
    balanced_split(c11, c12, c22, z1, z2) -> (rho1, rho2)

Rank-1 cover of an off-diagonal-active 2×2 block `[c11 c12; c12 c22]`, which requires
`c12^2 > c11*c22` (see [`offdiag_active`](@ref)). Returns `rho1 >= sqrt(c11)` and
`rho2 >= sqrt(c22)` with `rho1*rho2 == c12`: the cover is exactly tight on the off-diagonal, so
no cover of the block is smaller in both entries.

Those constraints leave a one-parameter family. The scale references `z1`, `z2` — one per parent,
supplied by [`split_block`](@ref): the parent value magnitudes wherever they carry a scale, the
weighted absolute sensitivities where they do not — pick a member, giving `rho_i` the units of
`sqrt(f)/[v_{p_i}]` and making the cover scale-covariant. The result is continuous in all five
arguments except where a vanishing scale reference meets a degenerate block; see the extended
help.

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
point runs off to an endpoint, and the endpoint absorbs it. Nothing computes those limits
separately, because floating-point arithmetic already carries them. `sqrt(c12*z2/z1)` is `Inf`
at `z1 == 0` and `0` at `z2 == 0`, which `clamp` resolves to the upper and the lower endpoint;
`c12/sqrt(c22)` is `Inf` at `c22 == 0`, placing the upper endpoint there. Only `z1 == z2 == 0`
leaves a `NaN` that no clamp resolves, and that is what reaches the fallback.

That reasoning needs the endpoint being approached to be a positive finite scale, and it is
not when the corresponding diagonal entry vanishes: `c22 == 0` puts the upper endpoint at
infinity, `c11 == 0` puts the lower endpoint at zero. Nor is there any limit to take when both
values vanish at once, since the equalizing point then depends on the ratio `z2/z1` alone. In
those cases the geometric mean of the endpoints is used, which is covariant and
permutation-equivariant.

So the result is continuous except at three families of inputs, where it is the ratio `z2/z1`
that carries the information and the ratio is unavailable:

  - `z1 == 0` with `c22 == 0`,
  - `z2 == 0` with `c11 == 0`,
  - `z1 == z2 == 0`, for any block.

A multiplication node with a vanishing operand has `c11 == c22 == 0` and meets all three at
once. Among the other elementals of `overloads.jl`, a quotient `u/w` meets the first at `u == 0`,
since `c22 = 2u/w^3` vanishes with the numerator, and `x^y` meets the second at a zero exponent,
where `c11 == 0` and `z2 == 0` while `c22 > 0`.

The continuity above is continuity in the values with the block held fixed. At those two
elementals the block moves with the value: the diagonal entry that bounds the feasible segment is
proportional to the vanishing value, so the endpoint the clamp settles on moves too, and the seed
of the slot whose value vanishes grows like `|value|^(-1/2)`. The split stays covariant and
valid, but the cover grows without bound as the value approaches zero while the derivatives it
covers stay finite, and at the zero itself the block falls into the families above and the
cover jumps to a finite one.

These zero-reference cases concern the value channel, and they are exactly the ones
[`value_bounds_split`](@ref) detects: [`split_block`](@ref) substitutes the declared scales there,
and a declared scale vanishes only at a slot with no tangent, which [`cover_slots!`](@ref) covers
diagonally instead of splitting. So a split reaches this function with a pair that is not
degenerate, and a pair carrying a ratio reaches the geometric mean only when `c12*z2/z1` leaves
the range of the element type at a block whose feasible segment has a degenerate endpoint. There
the equalizing point is recomputed from the separated exponents, which represents it whenever it
is representable at all; see [`split_fallback`](@ref). A split that is genuinely out of range
throws, since every pair returned here is a cover. Entries that are already non-finite are
returned as IEEE arithmetic gives them, which is what `check=false` asks for.
"""
function balanced_split(c11, c12, c22, z1, z2)
    lo = sqrt(c11)
    hi = c12 / sqrt(c22)
    # The equalizing point carries the vanishing-reference limits itself, and clamping resolves
    # them when the feasible interval has a finite endpoint. Only an unresolved NaN or unbounded
    # target reaches the fallback.
    rho1 = clamp(sqrt(c12 * z2 / z1), lo, hi)
    if !(isfinite(rho1) && rho1 > 0)
        rho1 = clamp(split_fallback(lo, hi, c12, z1, z2), lo, hi)
    end
    rho2 = c12 / rho1
    if !(isfinite(rho1) && rho1 > 0 && isfinite(rho2) && rho2 > 0)
        all(isfinite, (c11, c12, c22, z1, z2)) || return rho1, rho2
        throw(
            ArgumentError(
                "the balanced split of the block [$c11 $c12; $c12 $c22] against the scale " *
                    "references ($z1, $z2) is outside the floating-point range of " *
                    "$(typeof(rho1)): it asks for rho = ($rho1, $rho2)"
            )
        )
    end
    return rho1, rho2
end

"""
    split_fallback(lo, hi, c12, z1, z2) -> rho1

The equalizing point [`balanced_split`](@ref) aims at, for blocks where `sqrt(c12*z2/z1)` misses it.

A pair with a ratio to read misses it only by overflow or underflow of the product under the root,
and `frexp` recovers the point from the separated exponents: the exponent arithmetic is exact, and
a rescaling by powers of two shifts `e` by an even amount, so this form is covariant to the bit
exactly as the product form is. A pair with no ratio — a vanishing or non-finite reference — gets
the geometric mean of the endpoints, the covariant, permutation-equivariant point of the feasible
segment.
"""
function split_fallback(lo, hi, c12::AbstractFloat, z1::AbstractFloat, z2::AbstractFloat)
    (isfinite(c12) && c12 > 0 && isfinite(z1) && z1 > 0 && isfinite(z2) && z2 > 0) ||
        return sqrt(lo * hi)
    m12, e12 = frexp(c12)
    m2, e2 = frexp(z2)
    m1, e1 = frexp(z1)
    m = m12 * m2 / m1
    e = e12 + e2 - e1
    if isodd(e)
        # Only an even exponent halves exactly; the odd factor rides along in the mantissa.
        m *= 2
        e -= 1
    end
    return ldexp(sqrt(m), e >> 1)
end

# An element type with no exponent to separate has the geometric mean and nothing else.
split_fallback(lo, hi, c12, z1, z2) = sqrt(lo * hi)

"""
    value_bounds_split(c11, c12, c22, z1, z2, ratio) -> Bool

Whether the value pair `(z1, z2)` fixes the split of the block `[c11 c12; c12 c22]`.

[`balanced_split`](@ref) aims at `sqrt(c12*z2/z1)` and clamps it to the feasible segment
`[sqrt(c11), c12/sqrt(c22)]`, so a pair carrying no scale still yields a covariant split wherever
the endpoint it runs toward is positive and finite: the clamp supplies the answer, and both
endpoints are covariant. What is left is the pair of cases where that endpoint is degenerate as
well — `c22 == 0` puts the upper endpoint at infinity, `c11 == 0` puts the lower one at zero —
together with two vanishing values, which leave the ratio `z2/z1` undefined whatever the block.
`ratio` is how lopsided a pair must be to count as carrying no scale.

The test compares `z1` and `z2` directly, which is not invariant when the two parents carry
different units, so it is applied only where no covariant split exists. In particular a block
whose bounding diagonal entry is proportional to the vanishing value — a quotient's numerator, a
power's exponent — has its split fixed by the values until the value is exactly zero, and that
split grows without bound on the approach; see [`balanced_split`](@ref).
"""
function value_bounds_split(c11, c12, c22, z1, z2, ratio)
    (z1 == 0 && z2 == 0) && return false
    z1 <= ratio * z2 && return c22 > 0
    z2 <= ratio * z1 && return c11 > 0
    return true
end

"""
    split_ratio(ratio, T::Type) -> r

The degeneracy threshold a cover over element type `T` runs at, from the `ratio` keyword its entry
point was given, in `M =` [`magtype`](@ref)`(T)`. `nothing` means `sqrt(eps(M))`.

`ratio` is how lopsided a value pair must be before [`value_bounds_split`](@ref) counts it as
carrying no scale, and it must lie in `[0, 1)`: at `1` every pair is condemned. It caps the split
asymmetry the value channel can express, since a multiplication block aims at
`rho1 = sqrt(c12*z2/z1)` and so at a ratio of `sqrt(ratio)` between the two references.

The default resolves against `M` rather than being stored as a constant because the quantity it
bounds is representability. `eps(M)` is the floor below which one parent value is
indistinguishable from zero at the scale of the other; `sqrt(eps(M))` is the half-the-digits
margin above that floor, capping the asymmetry at `eps(M)^(-1/4)`. A constant chosen at one
precision is silently wrong at another — `1e-8` sits below `eps(Float32)`, which would collapse
the guard to an exactly-vanishing test on a `Float32` program, and that test does not survive an
ordinary perturbation of the evaluation point.
"""
function split_ratio(ratio, ::Type{T}) where {T}
    F = magtype(T)
    ratio === nothing && return sqrt(eps(F))
    r = convert(F, ratio)
    isfinite(r) && 0 <= r < 1 || throw(
        ArgumentError(
            "the degeneracy threshold `ratio` must lie in [0, 1): it is the value ratio below " *
                "which a pair counts as carrying no scale, and 1 would condemn every pair " *
                "(got $ratio)"
        )
    )
    return r
end

"""
    check_input_scales(w, ninputs::Integer) -> w

Validate declared input scales: `nothing` for unit scales, or one strictly positive finite entry
per input.

Positivity is not a formality. The declared scales stand in for the parent values at blocks whose
values carry none, so a zero entry would blank a live slot's scale, leaving its split with no
ratio to read and, at a block whose feasible segment is degenerate too, no cover to return.
"""
function check_input_scales(w, ninputs::Integer)
    w === nothing && return w
    Base.require_one_based_indexing(w)
    length(w) == ninputs || throw(
        DimensionMismatch(
            "declared input scales have length $(length(w)), expected $ninputs (one per input)"
        )
    )
    all(wi -> isfinite(wi) && wi > 0, w) || throw(
        ArgumentError(
            "declared input scales must be strictly positive and finite: a zero entry would " *
                "blank a live slot's scale where the guard falls back to them"
        )
    )
    return w
end

# Lazy magnitude view of the value channel, avoiding an O(L) copy.
struct AbsValues{M, T} <: AbstractVector{M}
    vals::Vector{T}
end
AbsValues(vals::Vector{T}) where {T} = AbsValues{magtype(T), T}(vals)
Base.size(a::AbsValues) = size(a.vals)
Base.IndexStyle(::Type{<:AbsValues}) = IndexLinear()
Base.getindex(a::AbsValues, i::Int) = magnitude(a.vals[i])

# Per-node scales, gathered into slot order for `cover_slots`. Lazy, so the tape sweeps hand over
# a node's scale references without copying them once per node.
struct SlotScales{T, S <: AbstractVector{T}, P <: AbstractVector{<:Integer}} <: AbstractVector{T}
    scales::S
    parents::P
end
Base.size(s::SlotScales) = size(s.parents)
Base.IndexStyle(::Type{<:SlotScales}) = IndexLinear()
Base.getindex(s::SlotScales, i::Int) = s.scales[s.parents[i]]

"""
    DeclaredScales(w, t)

The two node channels a block may ask for beyond the values: [`sensitivity_scales`](@ref) seeded
with the input scales `w`, and the [`live_nodes`](@ref) flags. Each is built by the first block
that asks for it and shared by every block after.

Deferring the sweeps is what makes them free on a program that never needs them. Deciding in
advance would take its own `O(L)` pass over the recorded blocks, which on a tape whose blocks all
keep a usable value pair costs about as much as the cover sweep it precedes — so every program
would pay for the possibility of a fallback none of its blocks reach.
"""
mutable struct DeclaredScales{M, W, TT <: Tape}
    const w::W
    const tape::TT
    sigmas::Vector{M}
    lives::Vector{Bool}
end
DeclaredScales(w, t::Tape{T}) where {T} = DeclaredScales(w, t, magtype(T)[], Bool[])

function declared_scale(d::DeclaredScales, i::Integer)
    isempty(d.sigmas) && (d.sigmas = sensitivity_scales(d.w, d.tape))
    return d.sigmas[i]
end

function dead_slot(d::DeclaredScales, i::Integer)
    isempty(d.lives) && (d.lives = live_nodes(d.tape))
    return !d.lives[i]
end

"""
    live_nodes(t::Tape) -> live

Which nodes of `t` have a tangent that is not identically zero. `live[k]` is `false` exactly when
every path from the inputs to node `k` passes through a recorded first partial that is exactly
zero (whose [`magnitude`](@ref) is zero), and then `∇v_k` is the zero vector in any arithmetic: every term of every path product
carries an exact zero factor.

The test is on the recorded partials rather than on a propagated magnitude, so it neither
underflows a live node to dead nor depends on the declared input scales.

Costs one `O(L)` forward pass, paid once per cover and only by a program carrying a block whose
off-diagonal is active.
"""
function live_nodes(t::Tape)
    live = fill(false, nnodes(t))
    for k in 1:t.ninputs
        live[k] = true
    end
    for k in (t.ninputs + 1):nnodes(t)
        d = t.ds[k]
        for (i, p) in enumerate(t.parents[k])
            if live[p] && !iszero(magnitude(d[i]))
                live[k] = true
                break
            end
        end
    end
    return live
end

"""
    SplitScale(value, declared)

Both scale references [`split_block`](@ref) may read at one slot: the parent value, and the means
of reaching the declared scale that stands in for it. They travel together because the choice
between them belongs to a *block*, not to a node — a variable whose value carries scale in one
product can be the vanishing operand of another — so a single scale per node cannot express the
split.

`declared` is whatever [`declared_scale`](@ref) resolves: the number itself where a backend
already has it, and a [`DeferredScale`](@ref) where the sweep that produces it should run only if
some block asks. [`dead_slot`](@ref) reads the same field, since a backend that can name a slot's
declared scale can say whether the slot has a tangent at all.
"""
struct SplitScale{T, D}
    value::T
    declared::D
end

"""
    dead_slot(s::SplitScale) -> Bool

Whether the slot's tangent `∇v` is identically zero.

Such a slot contributes no outer product to the Hessian
`H = sum_k vbar_k * sum_{i,j} C^k_{ij} * ∇v_{p_i} * ∇v_{p_j}'`, so every entry of its block that
involves it is multiplied away and needs no cover. [`cover_slots!`](@ref) drops those entries,
which is what keeps the cover from importing a live slot's scale into a term that vanishes.

The test is exact: a tolerance would either declare a live slot dead, leaving a real Hessian entry
uncovered, or compare two quantities with different units.

A slot reported by a bare number carries no liveness of its own, and the number is read as the
whole answer — which a declared scale is only while it cannot underflow. The backends here supply
a reference that reports liveness separately: [`DeferredScale`](@ref) over a tape,
[`ForwardScale`](@ref) in a propagation.
"""
dead_slot(s::SplitScale) = dead_slot(s.declared)
dead_slot(sigma::Real) = sigma == 0

"""
    declared_scale(x) -> sigma

The declared scale behind one slot's reference: the absolute sensitivity that stands in for a
parent value carrying no scale of its own.
"""
declared_scale(s::SplitScale) = declared_scale(s.declared)
declared_scale(sigma::Real) = sigma

"""
    DeferredScale(store, index)

A declared scale named but not yet computed: slot `index` of `store`, resolved by
[`declared_scale`](@ref) when a block first reaches the fallback.
"""
struct DeferredScale{D}
    store::D
    index::Int
end
declared_scale(d::DeferredScale) = declared_scale(d.store, d.index)
dead_slot(d::DeferredScale) = dead_slot(d.store, d.index)

# The scale references over a whole tape, gathered per slot on demand.
struct SplitScales{T, V <: AbstractVector{T}, D} <:
    AbstractVector{SplitScale{T, DeferredScale{D}}}
    values::V
    store::D
end
Base.size(g::SplitScales) = size(g.values)
Base.IndexStyle(::Type{<:SplitScales}) = IndexLinear()
Base.getindex(g::SplitScales, i::Int) =
    SplitScale(g.values[i], DeferredScale(g.store, i))

"""
    split_scales(w, t::Tape) -> scales

The per-node scale references [`local_cover_slots`](@ref) reads, one [`SplitScale`](@ref) per
node: a lazy view of `magnitude.(t.vals)` for the value channel, paired with the declared scales `w`
propagates, which are built only if some block asks for them. Obtained once per cover sweep and
shared by every node.
"""
split_scales(w, t::Tape) =
    SplitScales(AbsValues(t.vals), DeclaredScales(check_input_scales(w, t.ninputs), t))

"""
    sensitivity_scales(w, t::Tape) -> sigma

One [`abs_forward`](@ref) sweep seeded with the declared input scales `w`, so
`sigma_j >= sum_a w[a] * abs(∂v_j/∂x_a)` with equality when no path cancellation occurs.
`w = nothing` means unit scales, which asserts that the inputs share units.

These are the references the split falls back on where a block's parent values carry no scale.
They transform exactly as the values do under a rescaling of the intermediate variables, so the
covariance argument of [`balanced_split`](@ref) applies to them unchanged; under an input
rescaling `x -> D*x` the cover stays exactly covariant when the scales are rescaled alongside,
`w -> D*w`.

Cost: one `O(L)` forward sweep, paid once per cover and only by a program that reaches the
fallback.
"""
function sensitivity_scales(w, t::Tape{T}) where {T}
    theta = zeros(magtype(T), nnodes(t))
    if w === nothing
        fill!(view(theta, 1:t.ninputs), 1)
    else
        copyto!(view(theta, 1:t.ninputs), check_input_scales(w, t.ninputs))
    end
    return abs_forward(t, theta)
end

"""
    guards_any_block(t::Tape; ratio=nothing) -> Bool

Whether any block of `t` could reach the declared-scale fallback, decided by the same
[`value_bounds_split`](@ref) test the split itself applies and under the same conditions in which
[`cover_slots!`](@ref) consults a scale at all.

This is how a caller learns whether the unit-scale default is load-bearing for their program. On a
tape that answers `false` the cover reads parent values everywhere, is exactly covariant under an
input rescaling, and pays no extra sweep. On one that answers `true` the cover imports the units
its inputs are expressed in at the blocks where no covariant local choice exists, and stays
covariant only when the declared scales are rescaled alongside.

The test reads recorded values and second partials, never an adjoint and never a tangent. The
adjoint scales a block's three entries together, so it changes neither the value ratio nor which
diagonal entries are nonzero, except where it underflows one of them to zero. Reading it that way
overcounts — a block whose adjoint vanishes contributes no cover term for the fallback to reach —
and, in that underflow corner alone, can undercount. Ignoring tangents overcounts too: a block
with a [`dead_slot`](@ref) is covered diagonally and never consults a scale at all.

So this predicts which programs will ask for the declared scales without running a cover; the
covers themselves build those scales on demand and never call it, and they are the authority on
what a given tape does.
"""
function guards_any_block(t::Tape{T}; ratio = nothing) where {T}
    M = magtype(T)
    r = split_ratio(ratio, T)
    z = AbsValues(t.vals)
    for k in (t.ninputs + 1):nnodes(t)
        C, ps = t.Cs[k], t.parents[k]
        if C isa BinaryCurvature
            c11, c12, c22 = magnitude(C.c11), magnitude(C.c12), magnitude(C.c22)
            offdiag_active(c11, c12, c22) || continue   # the diagonal cover consults no scale
            value_bounds_split(c11, c12, c22, z[ps[1]], z[ps[2]], r) || return true
        elseif C isa PairedCurvature
            for j in 1:C.q
                # Every pair of an inner product is the multiplication block `[0 c; c 0]`.
                value_bounds_split(zero(M), one(M), zero(M), z[ps[j]], z[ps[C.q + j]], r) ||
                    return true
            end
        end
    end
    return false
end

"""
    split_block(ratio, c11, c12, c22, s1::SplitScale, s2::SplitScale) -> (rho1, rho2)

The member of the block's cover family the split selects, from the two channels each slot carries.

Parent values fix the split covariantly except at blocks where a vanishing value meets a degenerate
feasible segment, and there no covariant local choice exists at all: the equalizing point runs to
an endpoint that is itself zero or infinite. So the values are read wherever they carry a usable
scale and the declared input scales stand in wherever they do not, which keeps covariance where it
is attainable and asks for units only where it is not.
[`value_bounds_split`](@ref) is the test and `ratio` its threshold; see [`split_ratio`](@ref).

The guard compares two differently scaled quantities and is therefore not itself invariant: a
rescaling moves a block's value ratio, and one sitting near `ratio` can cross it. Nothing narrower
is available — the one degeneracy test a rescaling leaves alone is an exactly vanishing pair, and
a value exactly zero at one point is of order the displacement at a nearby one, so that test
catches a block at one evaluation point and not at its neighbors.

Cost: the guard is three comparisons on values already recorded, and the declared scales are built
by the first block that asks for them rather than in advance. A program whose blocks all keep a
usable value pair therefore pays nothing for the fallback; one that reaches a guarded block pays
one extra `O(L)` forward sweep, once. [`guards_any_block`](@ref) answers which case a tape is in
without running a cover.
"""
function split_block(ratio, c11, c12, c22, s1::SplitScale, s2::SplitScale)
    # The split only picks a member of the family; an upward-rounded bound need not reach it, and
    # the cover is made to hold afterwards (see `cover_slots!`).
    c11, c12, c22 = plainvalue(c11), plainvalue(c12), plainvalue(c22)
    z1, z2 = plainvalue(s1.value), plainvalue(s2.value)
    value_bounds_split(c11, c12, c22, z1, z2, plainvalue(ratio)) &&
        return balanced_split(c11, c12, c22, z1, z2)
    return balanced_split(
        c11, c12, c22, plainvalue(declared_scale(s1)), plainvalue(declared_scale(s2))
    )
end

"""
    local_covers(t::Tape, vbar; check::Bool=true) -> (rhos, theta)

Per-node rank-1 covers. For node `k`, `rhos[k][i] >= 0` satisfies
`rhos[k][i] * rhos[k][j] >= c^k[i,j]` where `c^k[i,j] = magnitude(vbar[k]) * magnitude(C^k[i,j])`,
except at
entries a [`dead_slot`](@ref) empties; nodes with no second-order contribution get an empty
`rhos[k]`. `theta[j] = sum_k rhos[k][i]` collects the weight each node receives as a parent, which
seeds the reverse sweep of [`abs_reverse`](@ref); it is in [`boundtype`](@ref) of the tape's
element type, so the sweep it seeds rounds as the tape asks.

Each node's cover comes from [`cover_slots`](@ref), dispatching on its [`Curvature`](@ref). A
[`DeclaredCurvature`](@ref) node (a declared sub-map output, see [`submapnode`](@ref)) brings its
own rank-1 cover, and there is no recorded `C` to check it against, so `check` cannot audit those
nodes — validity is conditional on the declaration.

`scales` declares the input scales the split falls back on and `ratio` the threshold at which it
does so; see [`split_block`](@ref) and [`split_ratio`](@ref). `check` verifies the covering
invariant at every recorded node.
"""
function local_covers(
        t::Tape{T}, vbar::AbstractVector; check::Bool = true,
        scales = nothing, ratio = nothing
    ) where {T}
    N = nnodes(t)
    r = split_ratio(ratio, T)
    sc = split_scales(scales, t)
    rhos = [magtype(T)[] for _ in 1:N]
    theta = zeros(boundtype(T), N)
    for k in (t.ninputs + 1):N
        rho = local_cover_slots(r, t, k, magnitude(vbar[k]), sc; check)
        rho === nothing && continue
        rhos[k] = rho
        for (i, p) in enumerate(t.parents[k])
            theta[p] += rho[i]
        end
    end
    return rhos, theta
end

"""
    local_cover_slots(t::Tape{T}, k::Integer, av; check::Bool=true) -> Union{Nothing,Vector{M}}
    local_cover_slots(ratio, t::Tape, k::Integer, av, scales; check::Bool=true)

The rank-1 cover of a single node, slot-aligned with its parents: nonnegative `rho` with
`rho[i]*rho[j] >= av*magnitude(C^k[i,j])` wherever both slots have a tangent (see
[`dead_slot`](@ref)), and `av` the absolute adjoint `magnitude(vbar[k])`. `rho` and `av` are in
`M =` [`magtype`](@ref)`(T)`. Returns `nothing` when the node contributes no
second-order term (`av == 0`, or a [`NoCurvature`](@ref) node).
This is the per-node kernel shared by [`local_covers`](@ref), which
accumulates the results into `(rhos, theta)`, and the [`CoverOnly`](@ref) sweep, which deposits
them directly into the absolute adjoint channel. It reads the node's [`Curvature`](@ref) off the
tape and forwards to [`cover_slots`](@ref), which holds the construction for each form. `check`
verifies the covering invariant on recorded nodes.

The first form takes the default split, resolving the threshold against the tape's element type.
The second takes a resolved `ratio` together with `scales = split_scales(w, t)`, both computed
once by the caller and shared across nodes.

This is the tape's adapter onto [`cover_slots`](@ref): it names the node with a
[`TapeSite`](@ref) and gathers `scales[t.parents[k]]` into slot order, both lazily.

A node whose covering entries are not finite is reported by
[`check_finite_partials`](@ref) rather than by the invariant test.
"""
local_cover_slots(t::Tape{T}, k::Integer, av::Real; check::Bool = true) where {T} =
    local_cover_slots(split_ratio(nothing, T), t, k, av, split_scales(nothing, t); check)

"""
    check_finite_partials(site::CoverSite, av, c::Tuple)

Throw unless the adjoint weight `av` of the elemental at `site` and its covering entries `c` are
all finite, reporting the values around it.

The covering invariant is a set of comparisons against `c`, so it cannot fire meaningfully when
`c` is not finite: every comparison against `NaN` is false and no finite `rho` covers an infinite
entry, so the test would reject a split that is not at fault. A non-finite `c` says instead that
the program is not twice-differentiable where it was evaluated — a division by zero, or a power,
logarithm or root at its singularity. A non-finite `av` says the same of a node downstream, whose
partials the adjoint has already passed through.
"""
function check_finite_partials(site::CoverSite, av::Real, c::Tuple)
    (isfinite(av) && all(isfinite, c)) && return nothing
    cause = isfinite(av) ? "$(site_label(site)) has non-finite second partials c=$c" :
        "the absolute adjoint reaching $(site_label(site)) is $av, from a node downstream"
    error(
        "$cause; the program is not twice-differentiable at this point ($(site_values(site)))"
    )
end

"""
    worst_severity(severity::AbstractArray) -> (s, idx)

Largest entry of `severity` and where it occurs, treating `NaN` as passing.

Checks over a whole array score every entry and reduce, rather than testing entries in turn and
raising at the first failure: the score is an array computation and only its reduction is
inspected, so the decision to raise is taken once, by the caller, on a single scalar. Each caller
normalizes its `severity` so that entries `<= 1` pass, which also makes the reported number the
factor by which the worst entry misses. A `NaN` entry comes from a vacuous comparison — zero
against zero — and scores as `0`.
"""
function worst_severity(severity::AbstractArray)
    finite = map(s -> ifelse(isnan(s), zero(s), s), severity)
    idx = argmax(finite)
    return finite[idx], idx
end

function local_cover_slots(
        ratio, t::Tape{T}, k::Integer, av::Real,
        scales::AbstractVector; check::Bool = true
    ) where {T}
    site = TapeSite(t, Int(k))
    sl = SlotScales(scales, t.parents[k])
    # The assertion is the return type [`cover_slots`](@ref) promises. Without it inference gives
    # up at the open fallback inside `dispatchcurvature` and the sweeps, which call this once per
    # node, box every result.
    return dispatchcurvature(t.Cs[k], T) do C
        cover_slots(C, ratio, site, av, sl; check)
    end::Union{Nothing, Vector{magtype(T)}}
end

"""
    abs_reverse(t::Tape, theta) -> abar

One reverse sweep in absolute-value arithmetic:
`abar[j] = theta[j] + sum_{k : j in P_k} magnitude(∂φ_k/∂v_j) * abar[k]`.

With `theta` from [`local_covers`](@ref), `abar[1:t.ninputs]` is the sum-merge cover — the whole
merge in a single gradient-cost sweep, because `sum_k s_k = sum_j theta[j] * utilde_j` is linear
in the seeds.

The sweep computes in the element type of `theta`, so seeds in [`boundtype`](@ref) give a sweep
that rounds as the tape asks.
"""
abs_reverse(t::Tape, theta::AbstractVector) = _reverse_sweep(_absslot, t, theta)

# Reverse sweep with a pluggable per-slot weight. `w(d, k, i, p)` sees the slot's first partial and
# both of its endpoints, so `_absslot` gives [`abs_reverse`](@ref) and the flow weightings of
# `covariant.jl` divide by a node's degree. A zero weight is skipped, which also keeps an infinite
# accumulator out of a slot that transmits nothing.
function _reverse_sweep(w::F, t::Tape, theta::AbstractVector) where {F}
    abar = collect(float.(theta))
    for k in nnodes(t):-1:1
        ak = abar[k]
        ak == 0 && continue
        for (i, p) in enumerate(t.parents[k])
            wd = w(t.ds[k][i], k, i, p)
            wd == 0 && continue
            abar[p] += wd * ak
        end
    end
    return abar
end

# The magnitude weighting, as a plain function so that the sweeps specialize on it.
_absslot(d, k, i, p) = magnitude(d)
# The squared magnitude, which carries squared seeds along each path.
_sqslot(d, k, i, p) = magnitude(d)^2

"""
    CoverOnly()

Return only the covers a sweep-tier computation yields: the gradient, sum-merge cover, and
variance channel from [`cover_hessian`](@ref) at `O(L)`, or the gradient and the two merges from [`cover_nls`](@ref) at
`O(G * L)`. This is the default mode of both.
"""
struct CoverOnly end

"""
    WithDiagnostics()

Return the extended result, adding the quantities that must be materialized per input or per
output: a [`HessianDiagnostics`](@ref) from [`cover_hessian`](@ref), with the elementwise bound,
quadrature merge, and an independent consistency check, at `O(n * L)` time and `O(n^2)` memory; an
[`NLSDiagnostics`](@ref) from [`cover_nls`](@ref), with the residual-sensitivity surrogate, at
`O(m * L)` time and `O(m * n)` memory.
"""
struct WithDiagnostics end

const CoverMode = Union{CoverOnly, WithDiagnostics}

"""
    cover_hessian([mode], t::Tape, outidx::Integer; check=true, scales=nothing, ratio=nothing, channel=AbsoluteTangents())
    cover_hessian([mode], f, x0::AbstractVector; check=true, scales=nothing, ratio=nothing, channel=AbsoluteTangents())
    cover_hessian(CoverOnly(), t::Tape, vbar::AbstractVector; check=true, scales=nothing, ratio=nothing)

Compute a certified rank-1 cover of a scalar function's Hessian.

Pass either a tape and scalar output node, or a function and evaluation point. The default
[`CoverOnly`](@ref) mode returns a [`HessianCover`](@ref) containing the gradient, the
sum-merge cover, and the uncertified variance channel. [`WithDiagnostics`](@ref) additionally computes an explicit elementwise bound
and a tighter quadrature cover.

The `scales` keyword declares the input scales the split falls back on where a block's parent
values carry none, and `ratio` the threshold at which it does so; see [`split_block`](@ref) and
[`split_ratio`](@ref). Neither affects validity, only which of the valid covers is chosen, and
[`guards_any_block`](@ref) says whether a program consults them at all. `check=true` verifies the
local covering conditions.

The `channel` keyword chooses which tangents the merges run over; see [`TangentChannel`](@ref).
The default [`AbsoluteTangents`](@ref) is what makes the `O(L)` sweep possible.
[`SignedTangents`](@ref) keeps the tangent signs and gives the tightest cover available at
`O(n*L)`, tighter at every coordinate than the default; it is available under
[`WithDiagnostics`](@ref) only.

Two merges are available:

- **sum merge**, `h_sum = sum_k s_k`: the cross terms of `(sum_k s_k)(sum_k s_k)'` are
  nonnegative slack;
- **quadrature merge**, `h_quad[i] = sqrt(sum_atoms s_atom[i]^2)`: valid by Cauchy–Schwarz, and
  the minimal cover of the no-cancellation surrogate `M = sum_atoms s_atom s_atom'`. An atom (see
  [`atom_slots`](@ref)) is a whole node for unary and binary elementals and one parent pair for
  an inner-product node, so an inner-product node contributes `q` separate terms to the
  quadrature merge rather than one.

Both satisfy `abs.(H) <= h * h'`.

The gradient is in the tape's element type `T` and the covers in [`magtype`](@ref)`(T)`. `T` may
be a type whose values stand for sets, such as `Interval{Float64}` from IntervalArithmetic.jl:
recording `f` over a box `x0 = interval.(lo, hi)` then gives covers (here `Vector{Float64}`) that
hold at every point of the box, and a gradient whose entries enclose the gradient over the box.
The sweeps over such a tape round every sum, product and square root upward (see
[`boundtype`](@ref)), so the covers hold exactly; on a tape of points they hold up to the rounding
of the sweeps.
Control flow in `f` that the box does not decide, such as a comparison between overlapping
intervals, raises an error, as does an elemental evaluated outside its domain somewhere on the box
(see [`check_values`](@ref)).

# `CoverOnly()`

This mode takes one forward pass and one reverse sweep, or `O(L)` time and memory. The reverse
sweep carries the ordinary adjoint, the absolute cover channel, and the variance channel
together:

    vbar[p_i] += d_i * vbar[k]                              # signed channel: reverse-mode AD
    abar[p_i] += magnitude(d_i) * abar[k] + rho_k[i]        # absolute channel: the cover
    qbar[p_i] += magnitude(d_i)^2 * qbar[k] + rho_k[i]^2    # variance channel

with `rho_k` from [`local_cover_slots`](@ref). Every channel of node `k` is final when the
sweep reaches it: adjoint flow and `theta` seeds come only from `k`'s consumers, which follow it
on the tape and are processed first. That makes `rho_k` — which needs the finished `vbar[k]` —
computable in place, and `abar[1:ninputs]` equal to `abs_reverse(t, theta)[1:ninputs]` up to
roundoff.

`abar` sums, over every path from a parent slot of a node to an input, the slot's seed times the
product of the absolute first partials along the path. `qbar` sums the squares of the same path
contributions, and `h_var = sqrt.(qbar[1:ninputs])` is the variance channel of
[`cover_variance`](@ref): a scale, not a cover.

A program that reaches the declared-scale fallback requires one additional forward sweep.

Passing the node adjoints `vbar = `[`adjoints`](@ref)`(t, outidx)` in place of `outidx` runs the
same sweep without the signed channel, for a caller that already holds the adjoints of a reverse
pass. It returns the same [`HessianCover`](@ref) bit for bit, with `grad = vbar[1:ninputs]`.

# `WithDiagnostics()`

This mode returns a [`HessianDiagnostics`](@ref) and costs `O(n * L)` time and `O(n^2)` memory.
With `check=true`, it also verifies that the diagnostic and fused computations agree — as an
equality under [`AbsoluteTangents`](@ref), where both routes compute the same merge, and as a
domination under [`SignedTangents`](@ref), where the absolute route bounds the signed one.

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
        ::WithDiagnostics, t::Tape{T}, outidx::Integer; check::Bool = true,
        scales = nothing, ratio = nothing, channel::TangentChannel = AbsoluteTangents()
    ) where {T}
    n = t.ninputs
    vbar = adjoints(t, outidx)
    u = tangent_channel(channel, t)
    B = tomag(T, hessian_bound(t, vbar, u))
    rhos, theta = local_covers(t, vbar; check, scales, ratio)

    h_sum, h_quad = tomag.(T, merge_atoms(t, rhos, u))
    h_sum_rev = tomag(T, abs_reverse(t, theta)[1:n])

    if check
        # These computations share most operations, so their roundoff errors are correlated.
        M = magtype(T)
        rtol = eps(M)^(3 // 4)
        # Relative disagreement in units of the tolerance, so `> 1` is a mismatch.
        disagreement(x, y) = @. separation(x, y) / (rtol * max(magnitude(x), one(M)))
        s, q = worst_severity(channel_severity(channel, h_sum, h_sum_rev, rtol, M))
        s > 1 &&
            error("reverse-sweep h_sum mismatch at input $q: forward=$(h_sum[q]) reverse=$(h_sum_rev[q])")
        # local_covers already checked the node invariants.
        pr = cover_hessian(CoverOnly(), t, outidx; check = false, scales, ratio)
        s, q = worst_severity(disagreement(view(vbar, 1:n), pr.grad))
        s > 1 &&
            error("paired-sweep gradient mismatch at input $q: adjoint=$(vbar[q]) paired=$(pr.grad[q])")
        s, q = worst_severity(channel_severity(channel, h_sum, pr.h_sum, rtol, M))
        s > 1 &&
            error("paired-sweep h_sum mismatch at input $q: forward=$(h_sum[q]) paired=$(pr.h_sum[q])")
    end

    return HessianDiagnostics(vbar[1:n], B, h_sum, h_quad, h_sum_rev)
end

# How a merge over `channel` must compare against the absolute-value reverse route, in units of
# `rtol` so that `> 1` fails. `M` is the type the covers are computed in. The routes compute the
# same number in the absolute channel; in the signed channel `abs.(∇v) <= utilde` entrywise, so
# the reverse route dominates instead.
channel_severity(::AbsoluteTangents, h, h_rev, rtol, ::Type{M}) where {M} =
    @. abs(h - h_rev) / (rtol * max(abs(h), one(M)))
channel_severity(::SignedTangents, h, h_rev, rtol, ::Type{M}) where {M} =
    @. (h - h_rev) / (rtol * max(abs(h), one(M)))

function cover_hessian(
        ::CoverOnly, t::Tape{T}, outidx::Integer; check::Bool = true,
        scales = nothing, ratio = nothing, channel::TangentChannel = AbsoluteTangents()
    ) where {T}
    check_cover_only_channel(channel)
    n, N = t.ninputs, nnodes(t)
    vbar = zeros(T, N)
    vbar[outidx] = 1
    abar, qbar = cover_only_sweep!(Val(true), vbar, t, split_ratio(ratio, T), split_scales(scales, t), check)
    return HessianCover(vbar[1:n], tomag(T, abar[1:n]), sqrt.(qbar[1:n]))
end

function cover_hessian(
        ::CoverOnly, t::Tape{T}, vbar::AbstractVector; check::Bool = true,
        scales = nothing, ratio = nothing, channel::TangentChannel = AbsoluteTangents()
    ) where {T}
    check_cover_only_channel(channel)
    n, N = t.ninputs, nnodes(t)
    axes(vbar) == (Base.OneTo(N),) || throw(
        DimensionMismatch("vbar must hold one adjoint per tape node, indices $(Base.OneTo(N)); got $(axes(vbar))")
    )
    abar, qbar = cover_only_sweep!(Val(false), vbar, t, split_ratio(ratio, T), split_scales(scales, t), check)
    return HessianCover{T}(vbar[1:n], tomag(T, abar[1:n]), sqrt.(qbar[1:n]))
end

check_cover_only_channel(channel::TangentChannel) = channel isa AbsoluteTangents || throw(
    ArgumentError(
        "CoverOnly cannot merge over $(nameof(typeof(channel))): its single sweep exists " *
            "because the absolute-tangent sum merge is linear in its seeds and collapses " *
            "into one reverse sweep, and a signed channel takes magnitudes inside the merge, " *
            "where nothing collapses. That channel costs O(n*L); ask for WithDiagnostics()"
    )
)

# The reverse sweep of `CoverOnly`, returning the absolute and variance channels `(abar, qbar)`.
# With `Val(true)`, `vbar` holds only the output seed on entry and the sweep accumulates the
# adjoints into it alongside the two channels. With `Val(false)`, `vbar` holds finished adjoints
# and is only read. Both forms perform the same channel operations in the same order, so they
# return identical channels whenever the adjoints agree.
function cover_only_sweep!(::Val{propagate}, vbar, t::Tape{T}, r, sc, check::Bool) where {propagate, T}
    N = nnodes(t)
    abar = zeros(boundtype(T), N)
    qbar = zeros(magtype(T), N)
    for k in N:-1:1
        vk, ak, qk = vbar[k], abar[k], qbar[k]
        ps = t.parents[k]
        avk = magnitude(vk)
        rho = local_cover_slots(r, t, k, avk, sc; check)
        if rho !== nothing
            for (i, p) in enumerate(ps)
                abar[p] += rho[i]
                qbar[p] += rho[i]^2
            end
        end
        # Without adjoint propagation, a node whose channels are both zero has nothing to send.
        ((!propagate || iszero(avk)) && ak == 0 && qk == 0) && continue
        ds = t.ds[k]
        for (i, p) in enumerate(ps)
            propagate && (vbar[p] += ds[i] * vk)
            d = magnitude(ds[i])
            abar[p] += d * ak
            qbar[p] += d^2 * qk
        end
    end
    return abar, qbar
end

function cover_hessian(
        mode::CoverMode, f, x0::AbstractVector; check::Bool = true,
        scales = nothing, ratio = nothing, channel::TangentChannel = AbsoluteTangents()
    )
    t, outidx = tape_function(f, x0)
    return cover_hessian(mode, t, outidx; check, scales, ratio, channel)
end

cover_hessian(t::Tape, outidx::Integer; check::Bool = true, scales = nothing, ratio = nothing) =
    cover_hessian(CoverOnly(), t, outidx; check, scales, ratio)
cover_hessian(f, x0::AbstractVector; check::Bool = true, scales = nothing, ratio = nothing) =
    cover_hessian(CoverOnly(), f, x0; check, scales, ratio)

"""
    cover_lattice(t::Tape, outidx::Integer; check::Bool=true, scales=nothing, ratio=nothing)
    cover_lattice(f, x0::AbstractVector; check::Bool=true, scales=nothing, ratio=nothing)

Compute the same Hessian cover four ways, once for each combination of a signed or absolute
tangent channel with a signed or absolute adjoint channel, and return them in a
[`CoverLattice`](@ref).

Each channel is a place where the construction may discard signs. The tangent channel is either
the exact derivatives from [`tangents`](@ref) or their absolute-value bound from
[`sensitivities`](@ref); the adjoint channel is either `abs.(vbar)` from [`adjoints`](@ref) or
the unsigned `acheck = abs_reverse(t, e_outidx)`, whose entry `acheck[k]` sums `prod(abs(∂φ/∂v))`
over every path from `k` to the output and therefore satisfies `acheck >= abs.(vbar)`.

An absolute channel bounds the exact one wherever it cancels, so replacing a signed channel by
its absolute counterpart can only inflate the cover. That orders the corners: `h_sum_signed` and
`h_quad_signed` are the smallest, `h_sum_absadj` and `h_quad_absadj` the largest, and the two
mixed corners fall between them but are incomparable with each other. The ordering in the adjoint
channel relies on the split scaling with the local matrix, which it does: both of its scale
references are properties of the parent nodes, and the adjoint enters every entry of the block
alike.

Cost is `O(n * L)` time and `O(n^2)` memory: two forward channels, two elementwise bounds, and
four merges, each of that order. This is a diagnostic — it measures what each sign channel is
worth on a given program, which the `O(L)` production sweeps cannot report.

`scales` and `ratio` govern the split, as in [`cover_hessian`](@ref); `check` verifies the local
covering conditions and that
the forward and reverse sweeps agree on the gradient and on the sum merge.

# Examples

```jldoctest
julia> f(x) = (d = sin(x[1]) - x[1]; u = d * x[2]; exp(u) - u);

julia> lat = cover_lattice(f, [0.7, 1.3]);

julia> round.(lat.h_quad; digits=4)          # the corner `cover_hessian` computes
2-element Vector{Float64}:
 3.1673
 0.0768

julia> round.(lat.h_quad_signed; digits=4)   # keeping the tangent signs as well
2-element Vector{Float64}:
 0.4854
 0.0768

julia> all(lat.h_quad_signed .<= lat.h_quad)
true
```
"""
function cover_lattice(
        t::Tape{T}, outidx::Integer; check::Bool = true, scales = nothing, ratio = nothing
    ) where {T}
    n, N = t.ninputs, nnodes(t)
    M = magtype(T)

    vbar = adjoints(t, outidx)
    # Seeded at the output alone and carrying no rho weights, the absolute reverse sweep returns
    # the path sum that abs.(vbar) would equal if no adjoint path cancelled.
    seed = zeros(boundtype(T), N)
    seed[outidx] = 1
    acheck = abs_reverse(t, seed)

    utilde = tangent_channel(AbsoluteTangents(), t)
    # The gradient is read off while the tangents are still signed; `signed_magnitudes` then
    # applies the same reduction `tangent_channel(SignedTangents(), t)` does, so the corners below
    # agree with `cover_hessian` under that channel to the last bit.
    w = tangents(t)
    grad = copy(w[outidx])
    w = signed_magnitudes(w)

    B = tomag(T, hessian_bound(t, vbar, utilde))
    B_signed = tomag(T, hessian_bound(t, vbar, w))

    rhos_signed, theta = local_covers(t, vbar; check, scales, ratio)
    rhos_abs, _ = local_covers(t, acheck; check, scales, ratio)

    h_sum, h_quad = tomag.(T, merge_atoms(t, rhos_signed, utilde))
    h_sum_absadj, h_quad_absadj = tomag.(T, merge_atoms(t, rhos_abs, utilde))
    h_sum_signed, h_quad_signed = tomag.(T, merge_atoms(t, rhos_signed, w))
    h_sum_forward, h_quad_forward = tomag.(T, merge_atoms(t, rhos_abs, w))

    if check
        rtol = eps(M)^(3 // 4)
        # The forward and reverse gradients accumulate the same path sums in opposite orders, so
        # their roundoff scales with the size of the terms rather than of the result: a coordinate
        # that cancels to zero still absorbs the error of the largest path reaching it. That
        # scale is `utilde[outidx]`, floored at one to keep the comparison absolute near zero.
        gscale = tomag(T, utilde[outidx])
        grad_rev = view(vbar, 1:n)
        s, q = worst_severity(@. separation(grad, grad_rev) / (rtol * max(gscale, one(M))))
        s > 1 &&
            error("forward-sweep gradient mismatch at input $q: forward=$(grad[q]) reverse=$(vbar[q])")
        # These two share most operations, so their roundoff errors are correlated.
        h_sum_rev = tomag(T, abs_reverse(t, theta)[1:n])
        s, q = worst_severity(@. abs(h_sum - h_sum_rev) / (rtol * max(h_sum, one(M))))
        s > 1 &&
            error("reverse-sweep h_sum mismatch at input $q: forward=$(h_sum[q]) reverse=$(h_sum_rev[q])")
    end

    return CoverLattice(
        grad, B, B_signed, h_sum, h_quad, h_sum_absadj, h_quad_absadj,
        h_sum_signed, h_quad_signed, h_sum_forward, h_quad_forward
    )
end

function cover_lattice(f, x0::AbstractVector; check::Bool = true, scales = nothing, ratio = nothing)
    t, outidx = tape_function(f, x0)
    return cover_lattice(t, outidx; check, scales, ratio)
end
