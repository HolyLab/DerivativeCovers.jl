# The squared channel: a one-sweep estimate of the quadrature merge.
#
# The exact quadrature merge needs O(nL) forward sensitivities. Instead, propagate squared seeds
# through squared partials so path contributions combine in quadrature. The root of the result is
# the sign-averaged cover, the root mean square of the quadrature merge when every local first
# partial and seed carries an independent, equiprobable sign. It drops positive covariance when
# paths reconverge and is exact on a linearized tree.
#
# The sign-averaged cover does not cover the Hessian. Comparing it with the certified sum channel
# estimates how much of that cover comes from overlapping paths.

"""
    squared_seeds(t::Tape, rhos) -> theta2

Squared seed weights: `theta2[j]` sums `rhos[k][i]^2` over every node `k` and parent slot `i`
with `t.parents[k][i] == j`. This is the squared-channel counterpart of the linear seeds `theta`
that [`local_covers`](@ref) returns, and it seeds the squared channel of
[`dependency_sweep`](@ref).
"""
function squared_seeds(t::Tape{T}, rhos::AbstractVector) where {T}
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

The staged form's propagation of the hard and squared channels, one reverse sweep each:

    abar[j] = theta[j]  + sum_{k : j in P_k} magnitude(∂φ_k/∂v_j)   * abar[k]
    vbar[j] = theta2[j] + sum_{k : j in P_k} magnitude(∂φ_k/∂v_j)^2 * vbar[k]

The hard channel is [`abs_reverse`](@ref): seeded with `theta` from [`local_covers`](@ref), its
value at the inputs is the sum merge. The squared channel squares both the seeds and the
coefficients, so it collects the same path contributions in quadrature; seeded with the matching
`theta2` from [`squared_seeds`](@ref), `sqrt.(vbar)` at the inputs is the sign-averaged cover, an
estimate of the quadrature merge from below. Both channels hold magnitudes: `vbar` here is the
squared channel, not an adjoint.

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
