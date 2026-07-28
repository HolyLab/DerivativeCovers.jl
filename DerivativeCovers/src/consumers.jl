# Task-level harnesses: what a consumer of a cover actually experiences when the cover is loose.
#
# Each consumer here uses a cover as a metric on the variables — as a stopping test, a
# regularizer, a positive-definite shift, or a preconditioner — and each is scale-invariant:
# rescaling the variables (x -> D*x, g -> D*g, A -> D*A*D, h -> D*h) leaves every reported
# quantity unchanged, as does rescaling the objective. The degradation a loose cover causes is
# therefore a property of the looseness alone, not of the units the problem is written in.
#
# The pattern is uniform: a function of one cover implements the task, and a paired function of
# two covers (a reference `h` and a distortion `hd` of it, see inflation.jl) reports what the
# distortion cost.

"""
    cover_norm(g, h) -> r

Norm of the gradient `g` measured in the metric of the cover `h`, `norm(g ./ h)`. This is the
scale-invariant convergence test: it is dimensionless (each `g[i]/h[i]` divides a gradient by a
curvature scale) and unchanged when the variables are rescaled.

A cover coordinate may be exactly zero, which says the Hessian ignores that input. The norm is
then `Inf` if `g[i]` is nonzero — no step along `i` is justified by curvature, so the test cannot
be satisfied — and the coordinate is simply skipped if `g[i]` vanishes too.
"""
function cover_norm(g::AbstractVector, h::AbstractVector)
    axes(g) == axes(h) ||
        throw(DimensionMismatch("g and h must share indices: $(axes(g)) vs $(axes(h))"))
    T = typeof(float(one(eltype(g)) / one(eltype(h))))
    acc = zero(T)
    for i in eachindex(g, h)
        gi = abs(g[i])
        iszero(gi) && continue
        h[i] > 0 || return T(Inf)
        acc += (gi / h[i])^2
    end
    return sqrt(acc)
end

"""
    stopping_index(grads, covers, tol) -> k

Index of the first iterate along a trajectory at which the convergence test
[`cover_norm`](@ref)`(grads[k], covers[k]) <= tol` fires, or `nothing` if it never does.

`grads` and `covers` are the gradients and covers at successive iterates of an optimization run
(a single cover may be repeated when it is held fixed along the path). `tol` is likewise data
describing that run — the convergence threshold it was run to test against — not a knob on the
accuracy of the returned index.
"""
function stopping_index(grads, covers, tol::Real)
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))
    for k in eachindex(grads, covers)
        cover_norm(grads[k], covers[k]) <= tol && return k
    end
    return nothing
end

"""
    stopping_shift(grads, covers, dcovers, tol) -> (; k_ref, k_dist, shift, overshoot)

What an inflated cover costs the convergence test. `covers` are the reference covers along the
trajectory `grads` and `dcovers` their distortions; `k_ref` and `k_dist` are the iterates at
which [`cover_norm`](@ref) first falls to `tol` under each.

An inflated cover shrinks the measured norm, so the distorted test always fires no later than the
reference one: `shift = k_ref - k_dist >= 0` counts the iterations it stops early, and
`overshoot >= 1` is the *true* cover-norm at that premature stop divided by `tol` — how far the
run really was from the requested tolerance when it declared success.

`tol` is data describing the analyzed run, alongside `grads` and `covers` — the convergence
threshold that run was tested against — not a knob on the accuracy of the returned shift.

Throws if the trajectory is too short for either test to fire; extend it or loosen `tol`.
"""
function stopping_shift(grads, covers, dcovers, tol::Real)
    k_dist = stopping_index(grads, dcovers, tol)
    k_dist === nothing &&
        throw(ArgumentError("the distorted convergence test never fires along this trajectory"))
    k_ref = stopping_index(grads, covers, tol)
    k_ref === nothing &&
        throw(ArgumentError("the reference convergence test never fires along this trajectory"))
    overshoot = cover_norm(grads[k_dist], covers[k_dist]) / tol
    return (; k_ref, k_dist, shift = k_ref - k_dist, overshoot)
end

"""
    regularized_step(A, g, h, tau) -> x

Step of the regularized system `(A + tau*Diagonal(h.^2)) * x = -g`. The shift
`tau*Diagonal(h.^2)` carries the units of the curvature the cover certifies, so `tau` is
dimensionless and a single value of it means the same thing on every problem.
"""
function regularized_step(A::AbstractMatrix, g::AbstractVector, h::AbstractVector, tau::Real)
    axes(A, 1) == axes(A, 2) == axes(g, 1) == axes(h, 1) ||
        throw(DimensionMismatch("A, g and h must share indices: $(axes(A)), $(axes(g)), $(axes(h))"))
    tau >= 0 || throw(ArgumentError("tau must be nonnegative, got $tau"))
    return (A + tau * Diagonal(h .^ 2)) \ (-g)
end

"""
    regularization_bias(A, g, h, hd, tau) -> b

How far an inflated cover moves the regularized step. `h` is the reference cover of `A` and `hd`
a distortion of it; the bias is the distance between the two steps of [`regularized_step`](@ref),
relative to the reference step, both measured in the reference cover's metric:

    b = norm(h .* (x_dist - x_ref)) / norm(h .* x_ref)

Comparing the two *regularized* steps — rather than either against the unregularized `A \\ (-g)` —
isolates the effect of cover looseness from the effect of regularizing at all, since `A` and `tau`
are held fixed and only the cover changes.
"""
function regularization_bias(A::AbstractMatrix, g::AbstractVector, h::AbstractVector,
                             hd::AbstractVector, tau::Real)
    axes(h) == axes(hd) ||
        throw(DimensionMismatch("h and hd must share indices: $(axes(h)) vs $(axes(hd))"))
    xref = regularized_step(A, g, h, tau)
    xdist = regularized_step(A, g, hd, tau)
    den = norm(h .* xref)
    den > 0 ||
        throw(ArgumentError("the reference step vanishes in the cover metric; bias is undefined"))
    return norm(h .* (xdist .- xref)) / den
end

_require_positive_cover(h::AbstractVector) =
    all(>(0), h) ||
        throw(ArgumentError("this consumer needs a strictly positive cover; got minimum $(minimum(h))"))

"""
    psd_shift(A, h) -> tau

Smallest `tau >= 0` for which `A + tau*Diagonal(h.^2)` is positive semidefinite: the cheapest PSD
boost expressed in the units the cover certifies. It is the negative part of the smallest
eigenvalue of `A` scaled into cover coordinates, `A[i,j]/(h[i]*h[j])`, and so is invariant under
rescaling of the variables.

Requires a strictly positive cover, since the scaling divides by it. `A` must be symmetric.
"""
function psd_shift(A::AbstractMatrix, h::AbstractVector)
    axes(A, 1) == axes(A, 2) == axes(h, 1) ||
        throw(DimensionMismatch("A and h must share indices: $(axes(A)) vs $(axes(h))"))
    issymmetric(A) || throw(ArgumentError("A must be symmetric"))
    _require_positive_cover(h)
    lambda = eigmin(Symmetric(A ./ (h .* h')))
    return max(zero(lambda), -lambda)
end

"""
    psd_excess(A, h, hd) -> (; tau_ref, tau_dist, excess)

How much extra shift an inflated cover forces the PSD boost to apply. `tau_ref` and `tau_dist` are
the minimal shifts of [`psd_shift`](@ref) under the reference cover `h` and its distortion `hd`,
and `excess[i]` is the ratio of the shift actually applied to coordinate `i` to the shift the
reference boost applies there,

    excess[i] = (tau_dist * hd[i]^2) / (tau_ref * h[i]^2)

which is at least 1, and equals 1 for every coordinate exactly when the distortion is uniform:
scaling a cover by a constant leaves the shift matrix `tau*Diagonal(h.^2)` unchanged, so this
consumer is blind to isotropic looseness and feels only anisotropy. Summarize the profile with
`maximum` for the worst-hit coordinate. When `A` is already positive semidefinite both shifts are
zero and the excess is 1 throughout.
"""
function psd_excess(A::AbstractMatrix, h::AbstractVector, hd::AbstractVector)
    axes(h) == axes(hd) ||
        throw(DimensionMismatch("h and hd must share indices: $(axes(h)) vs $(axes(hd))"))
    tau_ref = psd_shift(A, h)
    tau_dist = psd_shift(A, hd)
    if iszero(tau_ref)
        # Scaling by a positive diagonal cannot change the sign of the spectrum, so a PSD matrix
        # needs no shift under any cover.
        iszero(tau_dist) ||
            error("A needs no shift under h but a shift $tau_dist under hd; A is not symmetric?")
        return (; tau_ref, tau_dist, excess = fill!(similar(h, float(eltype(h))), 1))
    end
    excess = @. (tau_dist * hd^2) / (tau_ref * h^2)
    return (; tau_ref, tau_dist, excess)
end

"""
    cover_cg(A, b, h; x0=zero(b), maxiter=length(b), reltol=sqrt(eps()), reset=16)
        -> (; x, iters, xs)

Conjugate gradients on `A*x = b`, preconditioned by the cover: the residual is measured in the
metric `Diagonal(h.^2)` that the cover certifies, making the search directions invariant under
rescaling of the variables. Returns the final iterate `x`, the number of iterations `iters`, and
the history `xs` of every iterate from `x0` on, which is what lets a caller judge two different
covers by one common yardstick.

A cover that matches `A`'s curvature scales makes the scaled system well conditioned, so the
iteration count is a direct read on cover quality: an exact cover of a diagonal `A` converges in
one step, and rescaling that cover by a constant changes nothing, since only the *shape* of the
metric steers the iteration. Iteration stops when the cover-metric residual norm falls to `reltol`
times its initial value, or after `maxiter` iterations — which is the only backstop, as that norm
is not monotone in the iteration and a badly scaled metric makes it visibly non-monotone. Every
`reset` iterations the residual is recomputed from `x` rather than updated, to keep rounding error
from accumulating.

Note that the convergence test is itself stated in the cover's metric, so two runs under different
covers stop at genuinely different accuracies; comparing them requires a metric fixed outside the
iteration, as [`step_degradation`](@ref) does.

Requires a strictly positive cover, and a symmetric positive definite `A`. Definiteness is
enforced as the iteration goes: a search direction of nonpositive curvature throws rather than
producing the NaN iterates that dividing by it would.
"""
function cover_cg(A::AbstractMatrix, b::AbstractVector, h::AbstractVector;
                  x0::AbstractVector = zero(b), maxiter::Integer = length(b),
                  reltol::Real = sqrt(eps(float(real(eltype(b))))), reset::Integer = 16)
    axes(A, 1) == axes(A, 2) == axes(b, 1) == axes(h, 1) == axes(x0, 1) ||
        throw(DimensionMismatch("A, b, h and x0 must share indices"))
    reset > 0 || throw(ArgumentError("reset must be positive, got $reset"))
    reltol > 0 || throw(ArgumentError("reltol must be positive, got $reltol"))
    _require_positive_cover(h)
    hsq = h .^ 2
    x = float.(x0)
    r = b - A * x
    p = r ./ hsq
    errold = dot(p, r)                    # squared cover-metric norm of the residual
    tol = reltol^2 * errold
    xs = [copy(x)]
    dp = similar(p)
    iters = 0
    while errold > tol && iters < maxiter
        Ap = A * p
        pAp = dot(p, Ap)
        pAp > 0 ||
            throw(ArgumentError("cover_cg requires a positive definite A; direction $(iters + 1) has nonpositive curvature dot(p, A*p) = $pAp"))
        alpha = errold / pAp
        @. x += alpha * p
        if (iters + 1) % reset == 0
            r .= b .- A * x
        else
            @. r -= alpha * Ap
        end
        @. dp = r / hsq
        errnew = dot(dp, r)
        iters += 1
        push!(xs, copy(x))
        @. p = dp + (errnew / errold) * p
        errold = errnew
    end
    return (; x, iters, xs)
end

"""
    step_degradation(A, g, h, hd; maxiter=length(g), target=1e-6, kwargs...)
        -> (; iters_ref, iters_dist, error_ref, error_dist)

What an inflated cover costs the preconditioned Newton step of [`cover_cg`](@ref) on `A*x = -g`.
Both runs are judged in the *reference* cover's metric, against the exact step `A \\ (-g)`:

    error(x) = norm(h .* (x - A \\ (-g))) / norm(h .* (A \\ (-g)))

`iters_ref` and `iters_dist` are the iterations each run needs to bring that error below `target`,
or `nothing` if `maxiter` iterations do not suffice — which is itself the degradation, a step
budget the distorted cover cannot deliver on. `error_ref` and `error_dist` are the errors of the
final iterates, and are the reading to take when the budget, not the accuracy, is what is fixed.

Holding the yardstick fixed matters: `cover_cg`'s own convergence test lives in whichever metric
it was handed, so a loose cover would otherwise flatter itself by declaring victory sooner.
Remaining keyword arguments are forwarded to [`cover_cg`](@ref).
"""
function step_degradation(A::AbstractMatrix, g::AbstractVector, h::AbstractVector,
                          hd::AbstractVector; maxiter::Integer = length(g),
                          target::Real = 1e-6, kwargs...)
    axes(h) == axes(hd) ||
        throw(DimensionMismatch("h and hd must share indices: $(axes(h)) vs $(axes(hd))"))
    b = -g
    xexact = A \ b
    den = norm(h .* xexact)
    den > 0 ||
        throw(ArgumentError("the exact step vanishes in the cover metric; degradation is undefined"))
    err(x) = norm(h .* (x .- xexact)) / den
    # `iters` counts steps taken, so iterate k of the history is reached after k-1 of them.
    hit(xs) = (k = findfirst(x -> err(x) <= target, xs); k === nothing ? nothing : k - 1)

    ref = cover_cg(A, b, h; maxiter, kwargs...)
    dist = cover_cg(A, b, hd; maxiter, kwargs...)
    return (; iters_ref = hit(ref.xs), iters_dist = hit(dist.xs),
            error_ref = err(ref.x), error_dist = err(dist.x))
end
