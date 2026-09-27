# A one-sweep estimate of the quadrature merge.
#
# The exact quadrature merge needs O(nL) forward sensitivities. Instead, propagate squared seeds
# through squared partials so path contributions combine in quadrature. This drops positive
# covariance when paths reconverge and is exact on a linearized tree.
#
# The result is not a cover. Comparing it with the certified sum channel estimates how much of
# that cover comes from overlapping paths.

"""
    VarianceCover{T, M}

Result of [`cover_variance`](@ref). `T` is the element type of the [`Tape`](@ref) it was computed
from, and `M =` [`magtype`](@ref)`(T)` the type of the two channels. `VarianceCover{T}` stands for
`VarianceCover{T, magtype(T)}`.

- `grad::Vector{T}`: gradient of the scalar function.
- `h_sum::Vector{M}`: certified sum-merge Hessian cover.
- `h_var::Vector{M}`: variance estimate of the quadrature merge, under a random-sign model with
  one sign per parent slot (see [`cover_variance`](@ref)). This is **not a certified cover**.
- `severity::Vector{M}`: coordinatewise ratio `h_sum ./ h_var`. Larger values indicate more
  overlap from reconvergent paths.

The estimates satisfy `h_var <= h_quad <= h_sum` coordinatewise. Because `h_var` drops
covariances between reconvergent contributions, it need not bound the Hessian.
"""
struct VarianceCover{T, M}
    grad::Vector{T}
    h_sum::Vector{M}
    h_var::Vector{M}
    severity::Vector{M}
end
VarianceCover{T}(grad, h_sum, h_var, severity) where {T} =
    VarianceCover{T, magtype(T)}(grad, h_sum, h_var, severity)

"""
    variance_seeds(t::Tape, rhos) -> theta2

Squared seed weights: `theta2[j]` sums `rhos[k][i]^2` over every node `k` and parent slot `i`
with `t.parents[k][i] == j`. This is the variance-channel counterpart of the linear seeds `theta`
that [`local_covers`](@ref) returns, and it seeds the squared channel of
[`dependency_sweep`](@ref).
"""
function variance_seeds(t::Tape{T}, rhos::AbstractVector) where {T}
    N = nnodes(t)
    theta2 = zeros(magtype(T), N)
    for k in (t.ninputs + 1):N
        rho = rhos[k]
        isempty(rho) && continue
        for (i, p) in enumerate(t.parents[k])
            theta2[p] += rho[i]^2
        end
    end
    return theta2
end

"""
    dependency_sweep(t::Tape, theta, theta2) -> (abar, vbar)

The staged form's propagation of the hard and variance channels, one reverse sweep each:

    abar[j] = theta[j]  + sum_{k : j in P_k} magnitude(∂φ_k/∂v_j)   * abar[k]
    vbar[j] = theta2[j] + sum_{k : j in P_k} magnitude(∂φ_k/∂v_j)^2 * vbar[k]

The hard channel is [`abs_reverse`](@ref): seeded with `theta` from [`local_covers`](@ref), its
value at the inputs is the sum merge. The variance channel squares both the seeds and the
coefficients, so it collects the same path contributions in quadrature; seeded with the matching
`theta2` from [`variance_seeds`](@ref), `sqrt.(vbar)` at the inputs estimates the quadrature merge
from below. Both channels hold magnitudes: `vbar` here is the variance channel, not an adjoint.

[`cover_hessian`](@ref) forms both channels within its single fused sweep instead; this form
seeds them independently.
"""
function dependency_sweep(t::Tape, theta::AbstractVector, theta2::AbstractVector)
    N = nnodes(t)
    length(theta) == N ||
        throw(DimensionMismatch("theta has length $(length(theta)), expected $N (one per node)"))
    length(theta2) == N ||
        throw(DimensionMismatch("theta2 has length $(length(theta2)), expected $N (one per node)"))
    return _reverse_sweep(_absslot, t, theta), _reverse_sweep(_sqslot, t, theta2)
end

"""
    dependency_severity(h_sum, h_var) -> severity

Per-coordinate ratio `h_sum[i] / h_var[i]` of the two channels of [`cover_variance`](@ref).

A coordinate reached by a single contribution has the two channels agree and a severity of 1; a
coordinate whose cover is built by summing many overlapping contributions has the hard channel
grow like their sum and the variance channel like their root-sum-square, so the severity grows
with the effective overlap multiplicity.

The two channels vanish together — a coordinate carries seed weight through some path in both or
in neither — so coordinates with no Hessian dependence get a severity of 1 rather than `NaN`. A
vanishing `h_var` under a nonzero `h_sum` is reachable only by underflow, and gives `Inf`.
"""
function dependency_severity(h_sum::AbstractVector, h_var::AbstractVector)
    axes(h_sum) == axes(h_var) ||
        throw(DimensionMismatch("h_sum and h_var must share indices: $(axes(h_sum)) vs $(axes(h_var))"))
    return map(h_sum, h_var) do s, v
        r = float(s) / float(v)
        return isnan(r) ? one(r) : r
    end
end

"""
    cover_variance(t::Tape, outidx::Integer; check::Bool=true, scales=nothing, ratio=nothing) -> VarianceCover
    cover_variance(f, x0::AbstractVector; check::Bool=true, scales=nothing, ratio=nothing) -> VarianceCover

Compute a certified sum-merge cover and an inexpensive estimate of the quadrature merge.

Pass either a tape and scalar output node, or a function and evaluation point. The returned
[`VarianceCover`](@ref) contains the gradient, both channels, and their coordinatewise ratio.

Only `h_sum` is a certified cover; `h_var` may underestimate the Hessian. Both come from the
single reverse sweep of [`cover_hessian`](@ref), which returns them as well, so this costs `O(L)`
time and memory, independent of the number of inputs; `cover_variance` adds the severity.
`scales` and `ratio` govern the split, as in [`cover_hessian`](@ref), and `check=true` verifies
the local covering conditions.

The same channels follow from the staged form: [`local_covers`](@ref) and
[`variance_seeds`](@ref) supply the seeds of both channels, and [`dependency_sweep`](@ref)
propagates them.

`h_sum[i]` adds one contribution per pair of a seeded parent slot and a path from that slot to
input `i`: the slot's seed times the product of the absolute first partials along the path.
`h_var[i]^2` sums the squares of the same contributions. That is the expected square of their
sum when each parent slot carries an independent, equiprobable sign, on its first partial and on
its seed — the model [`gramvariance`](@ref) uses. A variable that enters one elemental through
two slots is therefore two paths whose contributions combine in quadrature, although in the
program they share a sign: `x[1] * x[1]` seeds each of its two slots with `1`, so `h_sum = 2` adds
them and `h_var = √2`, while `x[1]^2`, a unary elemental with a single slot, gives `√2` for
both. Where such paths reconverge further along, `h_var` falls below the exact scale: for
`x[1] * x[1] * x[1]` at `x = 3`, `h_var = √15` against `sqrt(abs(f''(3))) = √18`.

# Examples

```jldoctest
julia> f(x) = exp(x[1]) + x[1] * x[2]^2;

julia> vc = cover_variance(f, [0.5, 2.0]);

julia> round.(vc.h_sum; digits=4)
2-element Vector{Float64}:
 4.1125
 2.4142

julia> round.(vc.h_var; digits=4)
2-element Vector{Float64}:
 3.1062
 1.7321

julia> round.(vc.severity; digits=4)
2-element Vector{Float64}:
 1.3239
 1.3938

julia> cover_variance(x -> x[1] * x[1], [3.0]).h_sum   # two slots, two paths
1-element Vector{Float64}:
 2.0

julia> cover_variance(x -> x[1] * x[1], [3.0]).h_var ≈ [sqrt(2)]
true

julia> cover_variance(x -> x[1] * x[1] * x[1], [3.0]).h_var ≈ [sqrt(15)]   # |f''| = 18
true
```
"""
function cover_variance(t::Tape, outidx::Integer; check::Bool = true, scales = nothing, ratio = nothing)
    c = cover_hessian(CoverOnly(), t, outidx; check, scales, ratio)
    return VarianceCover(c.grad, c.h_sum, c.h_var, dependency_severity(c.h_sum, c.h_var))
end

function cover_variance(f, x0::AbstractVector; check::Bool = true, scales = nothing, ratio = nothing)
    t, outidx = tape_function(f, x0)
    return cover_variance(t, outidx; check, scales, ratio)
end
