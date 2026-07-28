# Jacobian-vector and vector-Jacobian products, and the two-sided equilibration they enable.
#
# A Jacobian cover (a, b) defines the equilibrated Jacobian Jhat = diag(a)^-1 J diag(b)^-1, whose
# entries are all bounded by 1 exactly when (a, b) covers J. Since every entry of a matrix is
# bounded by its spectral norm, sigma = opnorm(Jhat, 2) turns any valid cover into the valid cover
# (sqrt(sigma)*a, sqrt(sigma)*b), which is tighter whenever sigma < 1 — that is, whenever the cover
# is loose in the spectral sense rather than merely entrywise. This is the first-order twin of the
# spectral shrink in `spectral.jl`, with the same converges-from-below discipline.

"""
    jvp(t::Tape, outidxs::AbstractVector{<:Integer}, v::AbstractVector) -> Jv
    jvp(r, x0::AbstractVector, v::AbstractVector) -> Jv

Jacobian-vector product `J*v` of the taped vector-valued program, formed without the Jacobian by a
single forward tangent sweep `dv_k = sum_i (∂φ_k/∂v_i) dv_i` seeded with `v` at the input nodes and
read off at `outidxs`. Costs one sweep and `O(L)` time and memory, independent of the output count
`m` — unlike [`adjoints`](@ref), which is `O(m*L)`.

A repeated entry in `outidxs` yields an identical entry of `Jv`.
"""
function jvp(t::Tape, outidxs::AbstractVector{<:Integer}, v::AbstractVector)
    n, N = t.ninputs, nnodes(t)
    length(v) == n ||
        throw(DimensionMismatch("v has length $(length(v)), expected $n (one per input)"))
    Base.require_one_based_indexing(v)

    dv = zeros(N)
    copyto!(view(dv, 1:n), v)
    # Tape order is topological, so every parent's tangent is final before its child reads it.
    for k in (n + 1):N
        acc = 0.0
        for (i, p) in enumerate(t.parents[k])
            acc += t.ds[k][i] * dv[p]
        end
        dv[k] = acc
    end
    return dv[outidxs]
end

function jvp(r, x0::AbstractVector, v::AbstractVector)
    t, outidxs = tape_function(r, x0)
    return jvp(t, outidxs, v)
end

"""
    vjp(t::Tape, outidxs::AbstractVector{<:Integer}, u::AbstractVector) -> Jtu
    vjp(r, x0::AbstractVector, u::AbstractVector) -> Jtu

Vector-Jacobian product `J'*u` of the taped vector-valued program: one reverse adjoint sweep seeded
with `u[k]` at node `outidxs[k]`, read off at the input nodes. Costs one sweep and `O(L)` time and
memory, independent of the output count `m`.

Outputs that name the same node accumulate their seeds, so a repeated entry in `outidxs`
contributes the sum of the corresponding entries of `u` — the transpose of [`jvp`](@ref)'s
duplicated rows.
"""
function vjp(t::Tape, outidxs::AbstractVector{<:Integer}, u::AbstractVector)
    n, N = t.ninputs, nnodes(t)
    length(u) == length(outidxs) ||
        throw(DimensionMismatch("u has length $(length(u)), expected $(length(outidxs)) (one per output)"))
    Base.require_one_based_indexing(u)

    vbar = zeros(N)
    for k in eachindex(outidxs)
        vbar[outidxs[k]] += u[k]
    end
    # Tape order is topological, so N:-1:1 fully accumulates vbar[k] before k feeds its parents.
    for k in N:-1:1
        vk = vbar[k]
        vk == 0 && continue
        for (i, p) in enumerate(t.parents[k])
            vbar[p] += t.ds[k][i] * vk
        end
    end
    return vbar[1:n]
end

function vjp(r, x0::AbstractVector, u::AbstractVector)
    t, outidxs = tape_function(r, x0)
    return vjp(t, outidxs, u)
end

"""
    equilibrate(J::AbstractMatrix, a::AbstractVector, b::AbstractVector) -> Jhat

The two-sided equilibrated Jacobian `Jhat = diag(a)^-1 J diag(b)^-1`, in which the cover pair
`(a, b)` becomes the all-ones covering matrix: `(a, b)` covers `J` if and only if every entry of
`Jhat` is at most 1 in magnitude. The equilibrated matrix is the scale-invariant object a
downstream linear method should see, since it is unchanged by any rescaling of the outputs or
inputs that the cover tracks.

A zero weight is admissible only if the row or column it scales vanishes, in which case that row or
column of `Jhat` is zero; otherwise `(a, b)` does not cover `J` and an error is thrown.
"""
function equilibrate(J::AbstractMatrix, a::AbstractVector, b::AbstractVector)
    axes(J, 1) == axes(a, 1) && axes(J, 2) == axes(b, 1) ||
        throw(DimensionMismatch("J, a, b must share indices: $(axes(J)) vs $(axes(a)), $(axes(b))"))
    for k in eachindex(a)
        a[k] > 0 && continue
        all(j -> iszero(J[k, j]), axes(J, 2)) ||
            throw(ArgumentError("a[$k] is zero but row $k of J is not: (a, b) does not cover J"))
    end
    for j in eachindex(b)
        b[j] > 0 && continue
        all(k -> iszero(J[k, j]), axes(J, 1)) ||
            throw(ArgumentError("b[$j] is zero but column $j of J is not: (a, b) does not cover J"))
    end
    Jhat = similar(J, float(eltype(J)))
    for j in axes(J, 2), k in axes(J, 1)
        s = a[k] * b[j]
        Jhat[k, j] = iszero(s) ? zero(eltype(Jhat)) : J[k, j] / s
    end
    return Jhat
end

"""
    scaled_singular_value(J::AbstractMatrix, a::AbstractVector, b::AbstractVector) -> sigma
    scaled_singular_value(t::Tape, outidxs, a::AbstractVector, b::AbstractVector; maxiter=200, tol=1e-6, rng=MersenneTwister(0)) -> sigma

Largest singular value of the equilibrated Jacobian `Jhat = diag(a)^-1 J diag(b)^-1` (see
[`equilibrate`](@ref)), which measures how loose the cover `(a, b)` is in the spectral sense:
`(a, b)` is valid iff every entry of `Jhat` is at most 1 in magnitude, and
`sigma >= max_kj abs(Jhat[k,j])`.

The dense method is exact, and drops zero weights on the same terms as [`equilibrate`](@ref).

The tape method is matrix-free: power iteration on `Jhat'Jhat` driven by [`jvp`](@ref) and
[`vjp`](@ref) — about two gradients per iteration, `O(L)` memory — stopped when the estimate
changes by less than `tol` relative. It requires `a` and `b` to be strictly positive. Each iterate
`norm(Jhat*v)` with `norm(v) == 1` is a *lower* bound on `sigma` and the sequence converges to it
from below, so the estimate must carry a safety-factor inflation before it may shrink a cover
(see [`jacobian_shrink`](@ref)). Failure to converge within `maxiter` is an error, not a silently
returned partial result.
"""
function scaled_singular_value(J::AbstractMatrix, a::AbstractVector, b::AbstractVector)
    Jhat = equilibrate(J, a, b)
    rows = [k for k in eachindex(a) if a[k] > 0]
    cols = [j for j in eachindex(b) if b[j] > 0]
    (isempty(rows) || isempty(cols)) && return 0.0
    return opnorm(Jhat[rows, cols], 2)
end

function scaled_singular_value(t::Tape, outidxs::AbstractVector{<:Integer},
                               a::AbstractVector, b::AbstractVector;
                               maxiter::Integer=200, tol::Real=1e-6,
                               rng::AbstractRNG=MersenneTwister(0))
    n, m = t.ninputs, length(outidxs)
    length(b) == n ||
        throw(DimensionMismatch("b has length $(length(b)), expected $n (one per input)"))
    length(a) == m ||
        throw(DimensionMismatch("a has length $(length(a)), expected $m (one per output)"))
    all(>(0), a) && all(>(0), b) ||
        throw(ArgumentError("the matrix-free singular value requires strictly positive a and b"))

    v = randn(rng, n)
    v ./= norm(v)
    sigma = 0.0
    for _ in 1:maxiter
        u = jvp(t, outidxs, v ./ b) ./ a          # u = Jhat * v
        signew = norm(u)                          # <= opnorm(Jhat) for any unit v
        signew == 0 && return 0.0                 # the Jacobian vanishes on the whole support
        w = vjp(t, outidxs, u ./ a) ./ b          # w = Jhat' * u; nonzero since dot(w, v) = signew^2
        converged = abs(signew - sigma) <= tol * signew
        sigma = signew
        v = w ./ norm(w)
        converged && return sigma
    end
    error("power iteration did not converge in $maxiter iterations (last estimate $sigma)")
end

"""
    jacobian_shrink(a::AbstractVector, b::AbstractVector, sigma::Real; safety::Real=1) -> (as, bs)

Shrink the cover pair `(a, b)` by the spectral slack `sigma = scaled_singular_value(J, a, b)`,
returning `as = sqrt(s)*a` and `bs = sqrt(s)*b` with `s = min(1, safety*sigma)`.

Validity is inherited: `abs(J[k,j]) <= sigma * a[k] * b[j]` for every `k`, `j`, because every entry
of the equilibrated Jacobian is bounded by its spectral norm. The slack is split evenly between the
two sides so that the shrink commutes with rescaling of the outputs and of the inputs; any other
split of `s` between `as` and `bs` is equally valid. The shrink is a strict improvement exactly
when `safety*sigma < 1`; the clamp keeps a spectrally saturated cover (`sigma >= 1`) from being
inflated, which would be valid but looser.

`safety` inflates `sigma` before it is used. It must be large enough that `safety*sigma` is an
upper bound on the true spectral norm: with an exact `sigma` the default `1` is correct, but the
power-iteration estimate converges from below and needs `safety > 1`.
"""
function jacobian_shrink(a::AbstractVector, b::AbstractVector, sigma::Real; safety::Real=1)
    sigma >= 0 || throw(ArgumentError("sigma must be nonnegative, got $sigma"))
    safety > 0 || throw(ArgumentError("safety must be positive, got $safety"))
    scale = sqrt(min(one(float(sigma)), safety * sigma))
    return a .* scale, b .* scale
end
