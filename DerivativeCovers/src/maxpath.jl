# A row statistic of the Hessian's correlations, from two (max, ×) sweeps.
#
# The surrogate `M = sum_atoms s_atom s_atom'` of the quadrature merge has off-diagonal entries
# whose largest normalized value in each row, `max_{l != i} M_il / (d_i d_l)`, is the
# highest-correlation start of a factor-analysis fit. Forming it needs every atom's sensitivity
# vector, `O(n L)`. Moving the maximum over atoms outside the product,
#
#     m_i = max_atoms s_atom[i] * max_{l != i} s_atom[l] / d_l / d_i,
#
# and taking each `s_atom` along its largest path rather than summed over paths, makes the
# statistic linear in the (max, ×) semiring: a forward sweep carries each node's largest weighted
# path to any input, and a reverse sweep carries each atom's maximum back to the inputs. Both
# sweeps keep the two largest entries with distinct input tags, which is what lets the final step
# exclude `l == i` without knowing `i` during the sweep.
#
# The statistic bounds nothing. It is below the row maximum of the surrogate wherever several
# atoms or paths contribute to the maximizing entry, and is not a cover of any matrix.

# Insert the candidate `(v, i)` into the top-two `(v1, i1, v2, i2)`, whose entries carry distinct
# tags. A zero candidate never enters, so an empty slot keeps tag 0.
@inline function offer(v1, i1, v2, i2, v, i)
    v > v2 || return (v1, i1, v2, i2)
    i == i1 && return (max(v1, v), i1, v2, i2)
    v > v1 && return (v, i, v1, i1)
    return (v1, i1, v, i)
end

# Per-node top-two with distinct input tags.
struct TopTwo{M}
    v1::Vector{M}
    i1::Vector{Int}
    v2::Vector{M}
    i2::Vector{Int}
end

TopTwo{M}(N::Integer) where {M} = TopTwo{M}(zeros(M, N), zeros(Int, N), zeros(M, N), zeros(Int, N))

@inline function offer!(T::TopTwo, k, v, i)
    T.v1[k], T.i1[k], T.v2[k], T.i2[k] = offer(T.v1[k], T.i1[k], T.v2[k], T.i2[k], v, i)
    return T
end

# The largest entry of node `k` whose tag is not `i`.
best_excluding(T::TopTwo, k, i) = T.i1[k] == i ? T.v2[k] : T.v1[k]

"""
    maxpath_forward(t::Tape, d) -> TopTwo

Forward (max, ×) sweep with per-input seeds `1 / d[l]`: for every node `k`, the two largest values
of `U[k, l] / d[l]` over inputs `l`, with their inputs, where `U[k, l]` is the largest product of
absolute first partials along any path from input `l` to node `k`. An input with `d[l] == 0` seeds
nothing.

`O(L)` time and memory.
"""
function maxpath_forward(t::Tape{T}, d::AbstractVector) where {T}
    n, N = t.ninputs, nnodes(t)
    check_rowstat_scales(d, n)
    F = TopTwo{magtype(T)}(N)
    for (l, dl) in zip(1:n, d)
        dl == 0 && continue
        F.v1[l], F.i1[l] = inv(dl), l
    end
    for k in (n + 1):N
        for (j, p) in enumerate(t.parents[k])
            a = magnitude(t.ds[k][j])
            a == 0 && continue
            offer!(F, k, a * F.v1[p], F.i1[p])
            offer!(F, k, a * F.v2[p], F.i2[p])
        end
    end
    return F
end

function check_rowstat_scales(d, n)
    Base.require_one_based_indexing(d)
    length(d) == n ||
        throw(DimensionMismatch("d has length $(length(d)), expected $n (one per input)"))
    all(x -> x >= 0 && isfinite(x), d) ||
        throw(ArgumentError("the scales d must be finite and nonnegative"))
    return d
end

"""
    maxpath_rowcorrelation(t::Tape, outidx::Integer, d; check=true, scales=nothing, ratio=nothing) -> m

The (max, ×) row statistic of the Hessian at `outidx` relative to the scales `d`:

    m[i] = max over atoms of smax[i] * max_{l != i} smax[l] / d[l] / d[i],

where an atom's `smax[l] = max_{slots} rho[slot] * U[parent, l]` takes each slot along its largest
path (`U` as in [`maxpath_forward`](@ref)) and the largest of its slots, and `rho` are the local
covers of [`local_cover_slots`](@ref) under the same `scales` and `ratio` the cover sweeps use.

This is the max-merge counterpart of the largest normalized off-diagonal entry in row `i` of the
quadrature merge's surrogate `sum_atoms s_atom s_atom'`, and is at most that entry. It is exact
for a sum of squared linear forms written with each input once per form. It is a statistic for
starting a metric fit, not a bound, and not a cover. With `d` covariant (for instance a cover),
`m` is invariant under rescaling of the inputs and of the output. A coordinate with `d[i] == 0`
neither seeds the forward sweep nor receives a value: `m[i] = 0`.

Two sweeps: the forward (max, ×) sweep, and one reverse sweep that carries the adjoint and the
(max, ×) channel together, `O(L)` time and memory.
"""
function maxpath_rowcorrelation(
        t::Tape{T}, outidx::Integer, d::AbstractVector; check::Bool = true,
        scales = nothing, ratio = nothing
    ) where {T}
    n, N = t.ninputs, nnodes(t)
    M = magtype(T)
    F = maxpath_forward(t, d)
    r = split_ratio(ratio, T)
    sc = split_scales(scales, t)
    vbar = zeros(T, N)
    vbar[outidx] = 1
    R = TopTwo{M}(N)
    for k in N:-1:1
        vk = vbar[k]
        avk = magnitude(vk)
        ps = t.parents[k]
        rho = local_cover_slots(r, t, k, avk, sc; check)
        if rho !== nothing
            for atom in atom_slots(t.Cs[k], length(ps))
                g = (zero(M), 0, zero(M), 0)
                for i in atom
                    p = ps[i]
                    g = offer(g..., rho[i] * F.v1[p], F.i1[p])
                    g = offer(g..., rho[i] * F.v2[p], F.i2[p])
                end
                g1, j1, g2, j2 = g
                for i in atom
                    p = ps[i]
                    offer!(R, p, rho[i] * g1, j1)
                    offer!(R, p, rho[i] * g2, j2)
                end
            end
        end
        (iszero(avk) && R.v1[k] == 0) && continue
        ds = t.ds[k]
        for (i, p) in enumerate(ps)
            vbar[p] += ds[i] * vk
            a = magnitude(ds[i])
            a == 0 && continue
            offer!(R, p, a * R.v1[k], R.i1[k])
            offer!(R, p, a * R.v2[k], R.i2[k])
        end
    end
    return M[di == 0 ? zero(M) : best_excluding(R, i, i) / di for (i, di) in zip(1:n, d)]
end
