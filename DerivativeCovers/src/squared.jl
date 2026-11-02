# The squared channel: a one-sweep estimate of the quadrature merge.
#
# The exact quadrature merge needs O(nL) forward sensitivities. Instead, combine the path
# contributions in quadrature as the sweep goes, so that the channel holds the root of the sum of
# their squares. That root is the sign-averaged cover, the root mean square of the quadrature merge
# when every local first partial and seed carries an independent, equiprobable sign. It drops
# positive covariance when paths reconverge and is exact on a linearized tree.
#
# The channel accumulates with `hypot` and never forms a square, so it leaves the floating-point
# range only where the linear channel's own products of seeds and partials do, never because a
# square of one of them does.
#
# The sign-averaged cover does not cover the Hessian. Comparing it with the certified sum channel
# estimates how much of that cover comes from overlapping paths.

"""
    quadrature_seeds(t::Tape, etas) -> rho

Seed weights of the squared channel, in root form: `rho[j]^2` sums `etas[k][i]^2` over every node
`k` and parent slot `i` with `t.parents[k][i] == j`, accumulated with `hypot`. This is the
squared-channel counterpart of the linear seeds `theta` that [`local_covers`](@ref) returns, and
it seeds the squared channel of [`dependency_sweep`](@ref).
"""
function quadrature_seeds(t::Tape{T}, etas::AbstractVector) where {T}
    N = nnodes(t)
    rho = zeros(magtype(T), N)
    for k in (t.ninputs + 1):N
        eta = etas[k]
        isempty(eta) && continue
        for (i, p) in enumerate(t.parents[k])
            rho[p] = hypot(rho[p], eta[i])
        end
    end
    return rho
end

"""
    dependency_sweep(t::Tape, theta, rho) -> (abar, rbar)

The staged form's propagation of the hard and squared channels, one reverse sweep each:

    abar[j]   = theta[j]  + sum_{k : j in P_k} magnitude(∂φ_k/∂v_j)   * abar[k]
    rbar[j]^2 = rho[j]^2  + sum_{k : j in P_k} magnitude(∂φ_k/∂v_j)^2 * rbar[k]^2

The hard channel is [`abs_reverse`](@ref): seeded with `theta` from [`local_covers`](@ref), its
value at the inputs is the sum merge. The squared channel collects the same path contributions in
quadrature, holding the root of their sum of squares; seeded with the matching `rho` from
[`quadrature_seeds`](@ref), `rbar` at the inputs is the sign-averaged cover, an estimate of the
quadrature merge from below. It accumulates with `hypot`, so no square is formed. Both channels
hold magnitudes: `rbar` is not an adjoint.

[`cover_hessian`](@ref) forms both channels within its single fused sweep instead; this form
seeds them independently.
"""
function dependency_sweep(t::Tape, theta::AbstractVector, rho::AbstractVector)
    N = nnodes(t)
    length(theta) == N ||
        throw(DimensionMismatch("theta has length $(length(theta)), expected $N (one per node)"))
    length(rho) == N ||
        throw(DimensionMismatch("rho has length $(length(rho)), expected $N (one per node)"))
    return _reverse_sweep(_absslot, t, theta), _reverse_sweep(hypot, _absslot, t, rho)
end
