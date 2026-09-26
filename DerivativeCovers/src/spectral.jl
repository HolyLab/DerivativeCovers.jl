# Hessian-vector products and spectral shrinking.
#
# For D = diag(h), a valid cover makes every entry of Ĥ = D⁻¹HD⁻¹ at most one in magnitude.
# Therefore √norm(Ĥ, 2) h is also valid and is tighter when the scaled spectral norm is below one.

"""
    hvp(t::Tape, outidx::Integer, v::AbstractVector) -> Hv
    hvp(f, x0::AbstractVector, v::AbstractVector) -> Hv
    hvp!(Hv, t::Tape, outidx::Integer, v::AbstractVector) -> Hv

Hessian-vector product `H*v` of the taped program, formed without the Hessian by differentiating
the adjoint recursion along `v`: one forward tangent sweep `dv_k = sum_i (∂φ_k/∂v_i) dv_i` seeded
with `v`, then one reverse sweep carrying both the adjoints `vbar` and their tangents,

    vbardot_i += (∂φ_k/∂v_i) * vbardot_k + vbar_k * sum_j (∂²φ_k/∂v_i∂v_j) * dv_j.

The curvature term of each node comes from [`curvature_apply!`](@ref), which every
[`Curvature`](@ref) supplies in the form its structure allows — a [`PairedCurvature`](@ref) node
reads its `2q` nonzeros off the pairs rather than scanning a `2q × 2q` matrix, without which a
program built from `q`-long inner products would cost `O(q)` gradients instead of two. A node
whose second partials are only *declared* ([`has_exact_partials`](@ref) is `false`) makes an
exact product impossible: `hvp` throws on any tape containing one, unconditionally, rather than
letting success depend on whether the node's adjoint happens to vanish. The tape method of
[`scaled_spectral_radius`](@ref) inherits this restriction; its dense method does not.

Costs about two gradients and `O(L)` memory.

`hvp!` writes into `Hv`, which shares the axes of `v`. It saves the result allocation only: the
two sweeps still allocate their own `O(L)` working vectors, so it is not allocation-free.

# Examples

```jldoctest
julia> f(x) = exp(x[1]) + x[1] * x[2]^2;

julia> x0, v = [0.5, 2.0], [1.0, 0.0];

julia> round.(hvp(f, x0, v); digits=4)
2-element Vector{Float64}:
 1.6487
 4.0

julia> h = cover_hessian(f, x0).h_sum;   # the cover bounds the product

julia> all(abs.(hvp(f, x0, v)) .<= h .* (h' * abs.(v)))
true
```
"""
hvp(t::Tape, outidx::Integer, v::AbstractVector) =
    hvp!(similar(v, promote_type(eltype(t), float(eltype(v)))), t, outidx, v)

function hvp!(Hv::AbstractVector, t::Tape, outidx::Integer, v::AbstractVector)
    n, N = t.ninputs, nnodes(t)
    length(v) == n ||
        throw(DimensionMismatch("v has length $(length(v)), expected $n (one per input)"))
    axes(Hv) == axes(v) ||
        throw(DimensionMismatch("Hv and v must share indices: $(axes(Hv)) vs $(axes(v))"))
    Base.require_one_based_indexing(v)
    kdecl = findfirst(!has_exact_partials, t.Cs)
    kdecl === nothing ||
        throw(ArgumentError("hvp is unavailable: node $kdecl is a declared sub-map output (see submapnode) whose exact second partials are not recorded"))

    # The product runs in the tape's element type promoted with the tangent's.
    T = promote_type(eltype(t), eltype(Hv))
    dv = zeros(T, N)
    copyto!(view(dv, 1:n), v)
    for k in (n + 1):N
        acc = zero(T)
        for (i, p) in enumerate(t.parents[k])
            acc += t.ds[k][i] * dv[p]
        end
        dv[k] = acc
    end

    vbar = adjoints(t, outidx)
    vbardot = zeros(T, N)       # tangent of vbar; vbar[outidx] = 1 is constant, so this seeds at 0
    for k in N:-1:1
        ps = t.parents[k]
        isempty(ps) && continue
        bk, ak = vbar[k], vbardot[k]
        live = !iszero(magnitude(bk))
        (live || !iszero(magnitude(ak))) || continue
        ds = t.ds[k]
        for (i, p) in enumerate(ps)
            vbardot[p] += ds[i] * ak
        end
        if live
            # The ladder resolves the forms the tape recorded, whatever type the tangents carry.
            dispatchcurvature(t.Cs[k], eltype(t)) do C
                curvature_apply!(vbardot, C, ps, bk, dv)
            end
        end
    end
    Hv .= view(vbardot, 1:n)
    return Hv
end

@doc (@doc hvp) hvp!

function hvp(f, x0::AbstractVector, v::AbstractVector)
    t, outidx = tape_function(f, x0)
    return hvp(t, outidx, v)
end

"""
    scaled_spectral_radius(H::AbstractMatrix, h::AbstractVector) -> lambda
    scaled_spectral_radius(t::Tape, outidx::Integer, h::AbstractVector; maxiter=200, rtol=cbrt(eps(eltype(h))), rng=Xoshiro(0)) -> lambda

Spectral norm of the scaled Hessian `Hhat = D^-1 H D^-1`, `D = diag(h)`, which measures how loose
the cover `h` is in the spectral sense: `h` is valid iff every entry of `Hhat` is at most 1 in
magnitude, and `lambda >= max_ij abs(Hhat[i,j])`.

The dense method is exact. Coordinates with `h[i] == 0` are admissible only if row `i` of `H`
vanishes, in which case they are dropped; otherwise `h` does not cover `H` and an error is thrown.

The tape method is matrix-free: power iteration on `Hhat` driven by [`hvp`](@ref), stopped when
the estimate changes by less than `rtol` relative. It requires `h` to be strictly positive, and a
tape whose element type is its own [`magtype`](@ref): power iteration evaluates at a point, so a
set-valued tape (such as one recorded on intervals) is an error. Each
iterate `norm(Hhat*v)` with `norm(v) == 1` is a *lower* bound on `lambda` and the sequence
converges to it from below, so the estimate must be inflated by a safety factor before it is used
to shrink a cover (see [`spectral_shrink`](@ref)). Failure to converge within `maxiter` is an
error, not a silently returned partial result.
"""
function scaled_spectral_radius(H::AbstractMatrix, h::AbstractVector)
    axes(H, 1) == axes(H, 2) == axes(h, 1) ||
        throw(DimensionMismatch("H and h must share indices: $(axes(H)) vs $(axes(h))"))
    supp = [i for i in eachindex(h) if h[i] > 0]
    for i in eachindex(h)
        h[i] > 0 && continue
        all(j -> iszero(magnitude(H[i, j])), axes(H, 2)) ||
            throw(ArgumentError("h[$i] is zero but row $i of H is not: h does not cover H"))
    end
    isempty(supp) && return zero(real(one(eltype(H)) / (one(eltype(h)) * one(eltype(h)))))
    Hhat = [H[i, j] / (h[i] * h[j]) for i in supp, j in supp]
    return opnorm(Hhat, 2)
end

function scaled_spectral_radius(
        t::Tape, outidx::Integer, h::AbstractVector;
        maxiter::Integer = 200,
        rtol::Real = cbrt(eps(float(real(eltype(h))))),
        rng::AbstractRNG = Xoshiro(0)
    )
    n = t.ninputs
    magtype(eltype(t)) === eltype(t) || throw(
        ArgumentError(
            "power iteration is pointwise; it cannot bound the scaled spectral radius over a " *
                "set-valued tape of element type $(eltype(t))"
        )
    )
    Base.require_one_based_indexing(h)
    length(h) == n ||
        throw(DimensionMismatch("h has length $(length(h)), expected $n (one per input)"))
    all(>(0), h) ||
        throw(ArgumentError("the matrix-free spectral radius requires a strictly positive h"))

    T = promote_type(eltype(t), float(real(eltype(h))))
    v = randn(rng, T, n)
    v ./= norm(v)
    lambda = zero(T)
    for _ in 1:maxiter
        u = hvp(t, outidx, v ./ h) ./ h          # u = Hhat * v
        lambdanew = norm(u)                      # <= opnorm(Hhat) for any unit v
        lambdanew == 0 && return lambdanew       # the Hessian vanishes on the whole support
        converged = abs(lambdanew - lambda) <= rtol * lambdanew
        lambda = lambdanew
        v = u ./ lambdanew
        converged && return lambda
    end
    error("power iteration did not converge in $maxiter iterations (last estimate $lambda)")
end

"""
    spectral_shrink(h::AbstractVector, lambda::Real; safety::Real=1) -> hs

Shrink the cover `h` by the spectral slack `lambda = scaled_spectral_radius(H, h)`, returning
`hs = sqrt(min(1, safety*lambda)) * h`.

Validity is inherited: `abs(H[i,j]) <= lambda * h[i] * h[j]` for every `i`, `j`, because every
entry of the symmetric scaled Hessian is bounded by its spectral norm. The shrink is a strict
improvement exactly when `safety*lambda < 1`; the clamp keeps a spectrally saturated cover
(`lambda >= 1`) from being inflated, which would be valid but looser.

`safety` inflates `lambda` before it is used. It must be large enough that `safety*lambda` is an
upper bound on the true spectral norm: with an exact `lambda` the default `1` is correct, but the
power-iteration estimate converges from below and needs `safety > 1`.

`lambda` must bound the scaled norm wherever the cover is meant to hold: for a cover computed over
a set of points, such as a box of intervals, that is the supremum over the set. It must be a plain
number (its type is its own [`magtype`](@ref)); a set-valued `lambda`, such as an interval, is an
error.
"""
function spectral_shrink(h::AbstractVector, lambda::Real; safety::Real = 1)
    _check_pointwise_slack(lambda, "lambda")
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative, got $lambda"))
    safety > 0 || throw(ArgumentError("safety must be positive, got $safety"))
    return h .* sqrt(min(one(float(lambda)), safety * lambda))
end
