# Rank-1 covers of a program's Hessian, computed from its tape without forming the Hessian.
#
# A symmetric cover of a symmetric matrix H is a nonnegative vector h with |H[i,j]| <= h[i]h[j].
# The construction rests on the exact identity
#
#     H = sum_k vbar_k * sum_{i,j in P_k} (d²φ_k/dv_i dv_j) * ∇v_i (∇v_j)',
#
# where node k = φ_k(v_{P_k}) has reverse-mode adjoint vbar_k and ∇v_i is the total gradient of
# intermediate v_i with respect to the inputs. Bounding every factor in absolute value gives the
# elementwise bound B >= |H|; a per-node rank-1 cover s_k with s_k s_k' dominating node k's
# contribution to B then yields covers of H under either merge (see [`cover_hessian`](@ref)).
#
# An inner-product node (`dotnode` in tape.jl) has a block-diagonal second-partial matrix
# (`PairedPartials`) that splits into q independent 2x2 blocks, one per parent pair; it is
# covered by q rank-1 atoms rather than one, and the quadrature merge combines atoms rather than
# whole nodes (see `atom_slots`). A linear n-ary node (`linearnode`) carries an empty
# second-partial matrix and contributes no atom at all.

"""
    CoverResult

Covers and diagnostics for one taped program, as returned by [`cover_hessian`](@ref).

- `grad`: the gradient, i.e. the reverse-mode adjoints at the input nodes.
- `B`: the elementwise bound `B >= abs.(H)`. A diagnostic, quadratic in `n`; the production
  path never forms it.
- `h_sum`: the sum merge, `h = sum_k s_k`.
- `h_quad`: the quadrature merge, `h[i] = sqrt(sum_atoms s_atom[i]^2)`, combining per [`atom_slots`](@ref)
  atom rather than per node (see [`cover_hessian`](@ref)).
- `h_sum_rev`: `h_sum` recomputed by a single reverse sweep in absolute-value arithmetic — the
  form the production path uses, and equal to `h_sum` up to roundoff.
- `L`: the tape length (node count), inputs included.
"""
struct CoverResult
    grad::Vector{Float64}
    B::Matrix{Float64}
    h_sum::Vector{Float64}
    h_quad::Vector{Float64}
    h_sum_rev::Vector{Float64}
    L::Int
end

"""
    adjoints(t::Tape, outidx::Int) -> vbar

Reverse-mode adjoints `vbar[k] = ∂v_outidx/∂v_k` for every node of `t`. The gradient of the
program with respect to its inputs is `vbar[1:t.ninputs]`.
"""
function adjoints(t::Tape, outidx::Int)
    N = nnodes(t)
    vbar = zeros(N)
    vbar[outidx] = 1.0
    # Tape order is topological, so N:-1:1 fully accumulates vbar[k] before k feeds its parents.
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
    for k in 1:m
        Vbar[k, outidxs[k]] += 1.0
    end
    # Tape order is topological, so N:-1:1 fully accumulates each Vbar[:, j] before j feeds parents.
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

Rank-1 cover of an off-diagonal-active 2×2 block `[c11 c12; c12 c22]` (`c12^2 > c11*c22`),
returning `rho1 >= sqrt(c11)`, `rho2 >= sqrt(c22)`, `rho1*rho2 >= c12`.

The constraints leave a one-parameter family; the parent value magnitudes `z1`, `z2` break the
tie as scale references, making `rho_i` carry the units of `sqrt(f)/[v_{p_i}]`. Because
intermediate values do not change when the inputs are rescaled, this keeps the cover
scale-covariant. When a parent value vanishes, the fallback preserves validity but not the
scale-covariance argument.
"""
function balanced_split(c11, c12, c22, z1, z2)
    if z1 > 0 && z2 > 0
        rho1 = max(sqrt(c12 * z2 / z1), sqrt(c11))
        rho2 = max(sqrt(c22), c12 / rho1)
    else
        rho1 = max(sqrt(c12), sqrt(c11))
        rho2 = max(c12 / rho1, sqrt(c22))
    end
    return rho1, rho2
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

`check` verifies the covering invariant at every recorded node.
"""
function local_covers(t::Tape, vbar::AbstractVector; check::Bool=true)
    N = nnodes(t)
    rhos = [Float64[] for _ in 1:N]
    theta = zeros(N)
    for k in (t.ninputs + 1):N
        av = abs(vbar[k])
        ps, C = t.parents[k], t.Cs[k]
        if C isa PairedPartials
            c = av
            c == 0 && continue
            q = C.q
            rho = zeros(2q)
            for j in 1:q
                z1 = abs(t.vals[ps[j]])
                z2 = abs(t.vals[ps[q + j]])
                a, b = balanced_split(0.0, c, 0.0, z1, z2)
                if check
                    rtol = 1e-9
                    a * b >= c * (1 - rtol) ||
                        error("local cover invariant violated at node $k, pair $j: rho=($a,$b), c=$c")
                end
                rho[j] = a
                rho[q + j] = b
                theta[ps[j]] += a
                theta[ps[q + j]] += b
            end
            rhos[k] = rho
        elseif C isa CoveredPartials
            av == 0 && continue
            rho = sqrt(av) .* C.s
            rhos[k] = rho
            for (i, p) in enumerate(ps)
                theta[p] += rho[i]
            end
        elseif isempty(C)
            continue
        elseif length(ps) == 1
            c11 = av * abs(C[1, 1])
            c11 == 0 && continue
            rho = sqrt(c11)
            rhos[k] = [rho]
            theta[ps[1]] += rho
        elseif length(ps) == 2
            c11 = av * abs(C[1, 1])
            c12 = av * abs(C[1, 2])
            c22 = av * abs(C[2, 2])
            (c11 == 0 && c12 == 0 && c22 == 0) && continue
            if c12^2 <= c11 * c22
                rho1, rho2 = sqrt(c11), sqrt(c22)
            else
                z1 = abs(t.vals[ps[1]])
                z2 = abs(t.vals[ps[2]])
                rho1, rho2 = balanced_split(c11, c12, c22, z1, z2)
            end
            if check
                rtol = 1e-9
                (rho1 >= sqrt(c11) * (1 - rtol) && rho2 >= sqrt(c22) * (1 - rtol) &&
                 rho1 * rho2 >= c12 * (1 - rtol)) ||
                    error("local cover invariant violated at node $k: rho=($rho1,$rho2), c=($c11,$c12,$c22)")
            end
            rhos[k] = [rho1, rho2]
            theta[ps[1]] += rho1
            theta[ps[2]] += rho2
        else
            error("node $k has $(length(ps)) parents; only unary and binary elementals are supported")
        end
    end
    return rhos, theta
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
    cover_hessian(t::Tape, outidx::Int; check::Bool=true) -> CoverResult
    cover_hessian(f, x0::AbstractVector; check::Bool=true) -> CoverResult

Rank-1 covers of the Hessian of the taped program with output node `outidx`, or of the scalar
program `f` at `x0`. Both merges of the per-node covers `s_k = sum_{i in P_k} rho_k[i] * utilde_i`
are valid covers of the Hessian:

- **sum merge**, `h_sum = sum_k s_k`: the cross terms of `(sum_k s_k)(sum_k s_k)'` are
  nonnegative slack;
- **quadrature merge**, `h_quad[i] = sqrt(sum_atoms s_atom[i]^2)`: valid by Cauchy–Schwarz, and
  the minimal cover of the no-cancellation surrogate `M = sum_atoms s_atom s_atom'`. An atom (see
  [`atom_slots`](@ref)) is a whole node for unary and binary elementals and one parent pair for
  an inner-product node, so an inner-product node contributes `q` separate terms to the
  quadrature merge rather than one.

Together they give the chain `abs.(H) <= B <= sum_atoms s_atom s_atom' <= h*h'` for either `h`.

`check` verifies the per-node cover invariants and that the absolute-value reverse sweep
reproduces `h_sum`.

The returned `B`, `h_quad`, and the forward absolute sensitivities behind them cost `O(n*L)`;
they are diagnostics. `h_sum_rev` is computed on the production path — one forward pass, one
adjoint sweep, and one absolute-value reverse sweep, about twice the cost of a gradient.
"""
function cover_hessian(t::Tape, outidx::Int; check::Bool=true)
    n, N = t.ninputs, nnodes(t)
    vbar = adjoints(t, outidx)
    utilde = sensitivities(t)
    B = hessian_bound(t, vbar, utilde)
    rhos, theta = local_covers(t, vbar; check)

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
        for q in 1:n
            denom = max(abs(h_sum[q]), 1.0)
            abs(h_sum[q] - h_sum_rev[q]) <= 1e-12 * denom ||
                error("reverse-sweep h_sum mismatch at input $q: forward=$(h_sum[q]) reverse=$(h_sum_rev[q])")
        end
    end

    return CoverResult(vbar[1:n], B, h_sum, h_quad, h_sum_rev, N)
end

function cover_hessian(f, x0::AbstractVector; check::Bool=true)
    t, outidx = tape_function(f, x0)
    return cover_hessian(t, outidx; check)
end
