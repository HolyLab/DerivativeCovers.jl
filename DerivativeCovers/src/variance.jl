# The variance channel: a deterministic estimate of the quadrature merge, at the cost of one sweep.
#
# The quadrature merge is sqrt(diag(M)) with M = sum_k s_k s_k', and evaluating it exactly needs
# the forward sensitivities, hence O(n*L) time and memory. Propagate the *squares* of the seeds
# through a reverse sweep whose coefficients are the squared partials, and every path contribution
# accumulates in quadrature rather than summing: that is precisely what treating the reconvergent
# adjoints as uncorrelated amounts to. The dropped covariances are nonnegative, because every seed
# and every coefficient in the sweep is, so the channel underestimates diag(M) wherever paths
# reconverge and is exact when the linearized graph is a tree.
#
# The estimate is therefore *not* a cover. Its value is comparative: run it alongside the hard
# sum-merge channel in the same traversal and the per-coordinate ratio of the two measures how
# much of each coordinate's cover is overlap between reconvergent contributions rather than
# genuine Hessian magnitude.

"""
    VarianceCover

The hard and variance channels of one taped program, as returned by [`cover_variance`](@ref).

- `grad`: the gradient.
- `h_sum`: the sum merge — a guaranteed cover of the Hessian.
- `h_var`: the variance-channel estimate of the quadrature merge. It satisfies
  `h_var <= h_quad <= h_sum` coordinatewise and is **not itself a cover**: it drops the
  covariances between reconvergent contributions, so `abs(H[i,j])` may exceed `h_var[i]*h_var[j]`.
- `severity`: the per-coordinate ratio `h_sum ./ h_var`, at least 1, and large exactly on the
  coordinates whose cover is dominated by reconvergence rather than by Hessian magnitude.
"""
struct VarianceCover
    grad::Vector{Float64}
    h_sum::Vector{Float64}
    h_var::Vector{Float64}
    severity::Vector{Float64}
end

"""
    variance_seeds(t::Tape, rhos) -> theta2

Squared seed weights: `theta2[j]` sums `rhos[k][i]^2` over every node `k` and parent slot `i`
with `t.parents[k][i] == j`. This is the variance-channel counterpart of the linear seeds `theta`
that [`local_covers`](@ref) returns, and it seeds the squared channel of
[`dependency_sweep`](@ref).
"""
function variance_seeds(t::Tape, rhos::AbstractVector)
    N = nnodes(t)
    theta2 = zeros(N)
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

One reverse traversal carrying the hard and variance channels together:

    abar[j] = theta[j]  + sum_{k : j in P_k} abs(∂φ_k/∂v_j)   * abar[k]
    vbar[j] = theta2[j] + sum_{k : j in P_k}    (∂φ_k/∂v_j)^2 * vbar[k]

The hard channel is [`abs_reverse`](@ref): seeded with `theta` from [`local_covers`](@ref), its
value at the inputs is the sum merge. The variance channel squares both the seeds and the
coefficients, so it collects the same path contributions in quadrature; seeded with the matching
`theta2` from [`variance_seeds`](@ref), `sqrt.(vbar)` at the inputs estimates the quadrature merge
from below.
"""
function dependency_sweep(t::Tape, theta::AbstractVector, theta2::AbstractVector)
    N = nnodes(t)
    length(theta) == N ||
        throw(DimensionMismatch("theta has length $(length(theta)), expected $N (one per node)"))
    length(theta2) == N ||
        throw(DimensionMismatch("theta2 has length $(length(theta2)), expected $N (one per node)"))
    abar = collect(float.(theta))
    vbar = collect(float.(theta2))
    for k in N:-1:1
        ak, vk = abar[k], vbar[k]
        (ak == 0 && vk == 0) && continue
        for (i, p) in enumerate(t.parents[k])
            d = t.ds[k][i]
            d == 0 && continue
            abar[p] += abs(d) * ak
            vbar[p] += d^2 * vk
        end
    end
    return abar, vbar
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
    cover_variance(t::Tape, outidx::Integer; check::Bool=true, rule=ValueSplit()) -> VarianceCover
    cover_variance(f, x0::AbstractVector; check::Bool=true, rule=ValueSplit()) -> VarianceCover

The hard sum-merge cover and the variance-channel estimate of the quadrature merge, for the taped
program with output node `outidx` or for the scalar program `f` at `x0`, together with their
per-coordinate ratio (see [`dependency_severity`](@ref)).

Only `h_sum` is a guaranteed cover; `h_var` may fall below the Hessian it estimates. The pair
costs one adjoint sweep and one two-channel reverse sweep — `O(L)` time and memory, independent
of the input dimension — so the estimate is available wherever the sum merge is, unlike the exact
quadrature merge.

`rule` selects the [`SplitRule`](@ref) behind the per-node covers; both channels are seeded from
the same splits. `check` verifies the per-node cover invariants.

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
```
"""
function cover_variance(t::Tape, outidx::Integer; check::Bool=true, rule::SplitRule=ValueSplit())
    n = t.ninputs
    vbar = adjoints(t, outidx)
    rhos, theta = local_covers(t, vbar; check, rule)
    abar, w = dependency_sweep(t, theta, variance_seeds(t, rhos))
    h_sum = abar[1:n]
    h_var = sqrt.(w[1:n])
    return VarianceCover(vbar[1:n], h_sum, h_var, dependency_severity(h_sum, h_var))
end

function cover_variance(f, x0::AbstractVector; check::Bool=true, rule::SplitRule=ValueSplit())
    t, outidx = tape_function(f, x0)
    return cover_variance(t, outidx; check, rule)
end
