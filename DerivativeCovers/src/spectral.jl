# Hessian-vector products, and the spectral shrink they enable.
#
# A cover h defines the scaled Hessian Hhat = D^-1 H D^-1 with D = diag(h), whose entries are all
# bounded by 1 exactly when h covers H. Since every entry of a symmetric matrix is bounded by its
# spectral norm, lambda = norm(Hhat, 2) turns any valid cover into the valid cover sqrt(lambda)*h,
# which is tighter whenever lambda < 1 — that is, whenever the cover is loose in the spectral
# sense rather than merely entrywise.

"""
    hvp(t::Tape, outidx::Integer, v::AbstractVector) -> Hv
    hvp(f, x0::AbstractVector, v::AbstractVector) -> Hv
    hvp!(Hv, t::Tape, outidx::Integer, v::AbstractVector) -> Hv

Hessian-vector product `H*v` of the taped program, formed without the Hessian by differentiating
the adjoint recursion along `v`: one forward tangent sweep `dv_k = sum_i (∂φ_k/∂v_i) dv_i` seeded
with `v`, then one reverse sweep carrying both the adjoints `vbar` and their tangents,

    vbardot_i += (∂φ_k/∂v_i) * vbardot_k + vbar_k * sum_j (∂²φ_k/∂v_i∂v_j) * dv_j.

A node with an empty second-partial matrix (see [`Tape`](@ref)) has no curvature term; the sweep
skips it rather than indexing an empty `C`. A [`PairedPartials`](@ref) node has only `2q` nonzero
entries out of `(2q)^2`, so its curvature term is read off those pairs rather than by scanning the
dense `C`; without that, a program built from `q`-long inner products would cost `O(q)` gradients
instead of two. A declared sub-map output ([`submapnode`](@ref))
carries only a *bound* on its second partials, so an exact product is impossible: `hvp` throws
on any tape containing one, unconditionally, rather than letting success depend on whether the
node's adjoint happens to vanish. The tape method of [`scaled_spectral_radius`](@ref) inherits
this restriction; its dense method does not.

Costs about two gradients and `O(L)` memory.

`hvp!` writes into `Hv`, which shares the axes of `v`. It saves the result allocation only: the
two sweeps still allocate their own `O(L)` working vectors, so it is not allocation-free.
"""
hvp(t::Tape, outidx::Integer, v::AbstractVector) =
    hvp!(similar(v, float(eltype(v))), t, outidx, v)

function hvp!(Hv::AbstractVector, t::Tape, outidx::Integer, v::AbstractVector)
    n, N = t.ninputs, nnodes(t)
    length(v) == n ||
        throw(DimensionMismatch("v has length $(length(v)), expected $n (one per input)"))
    axes(Hv) == axes(v) ||
        throw(DimensionMismatch("Hv and v must share indices: $(axes(Hv)) vs $(axes(v))"))
    Base.require_one_based_indexing(v)
    kdecl = findfirst(C -> C isa CoveredPartials, t.Cs)
    kdecl === nothing ||
        throw(ArgumentError("hvp is unavailable: node $kdecl is a declared sub-map output (see submapnode) whose exact second partials are not recorded"))

    dv = zeros(N)
    copyto!(view(dv, 1:n), v)
    for k in (n + 1):N
        acc = 0.0
        for (i, p) in enumerate(t.parents[k])
            acc += t.ds[k][i] * dv[p]
        end
        dv[k] = acc
    end

    vbar = adjoints(t, outidx)
    vbardot = zeros(N)          # tangent of vbar; vbar[outidx] = 1 is constant, so this seeds at 0
    for k in N:-1:1
        ps = t.parents[k]
        isempty(ps) && continue
        bk, ak = vbar[k], vbardot[k]
        (bk == 0 && ak == 0) && continue
        ds, C = t.ds[k], t.Cs[k]
        if isempty(C)
            for (i, p) in enumerate(ps)
                vbardot[p] += ds[i] * ak
            end
        elseif C isa PairedPartials
            # Slot j pairs with slot q+j and with nothing else, so the curvature term is one
            # lookup per slot rather than a scan of the (2q)^2 entries.
            qh = C.q
            for j in 1:qh
                pw, px = ps[j], ps[qh + j]
                vbardot[pw] += ds[j] * ak + bk * dv[px]
                vbardot[px] += ds[qh + j] * ak + bk * dv[pw]
            end
        else
            for (i, p) in enumerate(ps)
                curv = 0.0
                for (j, q) in enumerate(ps)
                    curv += C[i, j] * dv[q]
                end
                vbardot[p] += ds[i] * ak + bk * curv
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
the estimate changes by less than `rtol` relative. It requires `h` to be strictly positive. Each
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
        all(j -> iszero(H[i, j]), axes(H, 2)) ||
            throw(ArgumentError("h[$i] is zero but row $i of H is not: h does not cover H"))
    end
    isempty(supp) && return zero(real(one(eltype(H)) / (one(eltype(h)) * one(eltype(h)))))
    Hhat = [H[i, j] / (h[i] * h[j]) for i in supp, j in supp]
    return opnorm(Hhat, 2)
end

function scaled_spectral_radius(t::Tape, outidx::Integer, h::AbstractVector;
                                maxiter::Integer=200,
                                rtol::Real=cbrt(eps(float(real(eltype(h))))),
                                rng::AbstractRNG=Xoshiro(0))
    n = t.ninputs
    length(h) == n ||
        throw(DimensionMismatch("h has length $(length(h)), expected $n (one per input)"))
    all(>(0), h) ||
        throw(ArgumentError("the matrix-free spectral radius requires a strictly positive h"))

    v = randn(rng, n)
    v ./= norm(v)
    lambda = 0.0
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
"""
function spectral_shrink(h::AbstractVector, lambda::Real; safety::Real=1)
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative, got $lambda"))
    safety > 0 || throw(ArgumentError("safety must be positive, got $safety"))
    return h .* sqrt(min(one(float(lambda)), safety * lambda))
end
