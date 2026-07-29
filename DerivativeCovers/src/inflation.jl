# Controlled distortions of a cover, for measuring what a consumer of covers tolerates.
#
# Every distortion here is an inflation: it multiplies a cover coordinatewise by factors that are
# never below 1. Inflation can only weaken the bound h[i]*h[j] >= abs(H[i,j]), so a distorted
# cover of H is still a cover of H, and a consumer's degradation under one is attributable to
# looseness alone.
#
# A pattern supplies a *shape* w with 0 <= w <= 1 and max(w) == 1; the factors are peak.^w. So
# `peak` is the inflation of the worst coordinate under every pattern, and the patterns differ
# only in how the remaining coordinates are spread beneath it. That separates the magnitude of a
# distortion from its anisotropy, which is what a tolerance study varies independently.

"""
    InflationPattern

How an inflation is distributed across coordinates. Subtypes supply a shape via
[`inflation_shape`](@ref), which [`inflation_factors`](@ref) raises to a peak factor:
[`UniformInflation`](@ref), [`LognormalInflation`](@ref), [`SpikeInflation`](@ref),
[`RampInflation`](@ref).
"""
abstract type InflationPattern end

"""
    UniformInflation()

Every coordinate inflated by the same factor. The isotropic reference: [`anisotropy`](@ref) of
its factors is exactly zero, so it is the control against which the anisotropic patterns are read.
"""
struct UniformInflation <: InflationPattern end

"""
    LognormalInflation()

Log-factors proportional to `abs(z)` for i.i.d. standard normal `z`, rescaled so the largest is
the requested peak. The factors are therefore lognormal in shape but exact in magnitude: the draw
decides which coordinates are loose and how sharply, never how loose the worst one is.

For large `n` the maximum of the draw runs well ahead of the bulk, so the profile is a
concentrated head over coordinates that stay near 1 — the anisotropy regime seen on the deep and
DAG families. Contrast [`UniformInflation`](@ref), the plateau regime.
"""
struct LognormalInflation <: InflationPattern end

"""
    SpikeInflation(index)

One coordinate inflated to the peak, every other left exactly at 1. The extreme point of
anisotropy at a given peak, and the pattern that isolates a single coordinate's contribution to a
consumer's degradation. `index` must be a valid index into the cover.
"""
struct SpikeInflation{I} <: InflationPattern
    index::I
end

"""
    RampInflation()

Log-factors graded linearly with position, from 1 at the first coordinate to the peak at the
last. Coordinate *order* is the ramp's axis, so permute the cover (or the problem) to choose
which coordinates take the looseness.
"""
struct RampInflation <: InflationPattern end

"""
    inflation_shape(pattern, h::AbstractVector; rng=Xoshiro(0)) -> w

The pattern's shape over the coordinates of `h`: `0 <= w <= 1` with `maximum(w) == 1`, sharing
the axes of `h`. `rng` is used only by the random patterns.
"""
function inflation_shape end

_shape(h::AbstractVector) = fill!(similar(h, float(eltype(h))), 0)

inflation_shape(::UniformInflation, h::AbstractVector; rng::AbstractRNG = Xoshiro(0)) =
    fill!(_shape(h), 1)

function inflation_shape(
        p::SpikeInflation, h::AbstractVector;
        rng::AbstractRNG = Xoshiro(0)
    )
    p.index in eachindex(h) ||
        throw(BoundsError(h, p.index))
    w = _shape(h)
    w[p.index] = 1
    return w
end

function inflation_shape(
        ::RampInflation, h::AbstractVector;
        rng::AbstractRNG = Xoshiro(0)
    )
    w = _shape(h)
    n = length(h)
    for (j, i) in enumerate(eachindex(h))
        w[i] = n == 1 ? 1 : (j - 1) / (n - 1)
    end
    return w
end

function inflation_shape(
        ::LognormalInflation, h::AbstractVector;
        rng::AbstractRNG = Xoshiro(0)
    )
    w = _shape(h)
    for i in eachindex(w)
        w[i] = abs(randn(rng))
    end
    peak = maximum(w)
    # A draw of all-exact-zeros has probability zero, but its shape would be 0/0.
    peak > 0 || throw(ArgumentError("degenerate lognormal draw: every log-factor is zero"))
    w ./= peak
    return w
end

"""
    inflation_factors(pattern, h::AbstractVector, peak; rng=Xoshiro(0)) -> f
    inflation_factors(pattern, n::Integer, peak; rng=Xoshiro(0)) -> f

Per-coordinate inflation factors `f = peak .^ inflation_shape(pattern, h)`, laid out on the axes
of `h` (or as a length-`n` vector). Every factor lies in `[1, peak]`, and the largest equals
`peak` exactly, whatever the pattern; `peak == 1` gives the identity distortion.

`f` *is* the realized per-coordinate profile of the distortion, so a caller that wants to report
what it applied should keep `f` and summarize it (with [`anisotropy`](@ref), say) rather than
recompute it from the inflated cover — a cover coordinate that is exactly zero stays zero under
[`inflate`](@ref) and reveals nothing about the factor it was multiplied by.
"""
function inflation_factors(
        pattern::InflationPattern, h::AbstractVector, peak::Real;
        rng::AbstractRNG = Xoshiro(0)
    )
    peak >= 1 || throw(ArgumentError("peak must be at least 1, got $peak"))
    return peak .^ inflation_shape(pattern, h; rng)
end

inflation_factors(
    pattern::InflationPattern, n::Integer, peak::Real;
    rng::AbstractRNG = Xoshiro(0)
) =
    inflation_factors(pattern, zeros(n), peak; rng)

"""
    inflate(h::AbstractVector, f::AbstractVector) -> hd
    inflate(pattern, h::AbstractVector, peak; rng=Xoshiro(0)) -> hd

Loosen the cover `h` coordinatewise by the factors `f` (or by the factors that `pattern` and
`peak` generate, per [`inflation_factors`](@ref)). Since every factor is at least 1, `hd` covers
every matrix `h` covers.

A coordinate with `h[i] == 0` — which a valid cover has wherever the Hessian ignores that input —
stays at zero: a multiplicative distortion cannot loosen a bound that is already exact. Factors
below 1 are rejected, as they could break the cover.
"""
function inflate(h::AbstractVector, f::AbstractVector)
    axes(h) == axes(f) ||
        throw(DimensionMismatch("h and f must share indices: $(axes(h)) vs $(axes(f))"))
    all(>=(1), f) ||
        throw(ArgumentError("every inflation factor must be at least 1; got minimum $(minimum(f))"))
    return h .* f
end

inflate(
    pattern::InflationPattern, h::AbstractVector, peak::Real;
    rng::AbstractRNG = Xoshiro(0)
) =
    inflate(h, inflation_factors(pattern, h, peak; rng))

"""
    anisotropy(f::AbstractVector) -> a

Spread of the inflation factors `f`, as the log ratio of their largest to their geometric mean.
It is zero exactly when every coordinate is inflated alike, and grows as the same peak inflation
is concentrated on fewer coordinates: at peak `p` over `n` coordinates, `UniformInflation` gives
0, `RampInflation` about `log(p)/2`, and `SpikeInflation` `log(p)*(1 - 1/n)`.

Unlike the peak, this is a property of the realized profile rather than of the request, so it is
the strength axis to plot a consumer's degradation against when the pattern is random.
"""
function anisotropy(f::AbstractVector)
    all(>(0), f) || throw(ArgumentError("inflation factors must be positive"))
    return log(maximum(f)) - sum(log, f) / length(f)
end
