# Quality metrics for a cover: is it valid, and how loose is it?

# The type of a ratio of magnitudes of entries of `A` and `B`.
_ratio_type(A, B) = typeof(float(one(magtype(eltype(A))) / one(magtype(eltype(B)))))

"""
    cover_ratio(H, h) -> r

The worst-case ratio `max_ij abs(H[i,j]) / (h[i]*h[j])` over the nonzero entries of `H`. The vector
`h` covers `H` exactly when `r <= 1`. Zero entries of `H` impose no constraints and are skipped; a
nonzero entry with a zero denominator gives `Inf`. `abs` is [`magnitude`](@ref), so an `H` whose
entries enclose a set of Hessians (such as an interval matrix) gives the worst ratio over the set.
"""
function cover_ratio(H::AbstractMatrix, h::AbstractVector)
    axes(H, 1) == axes(H, 2) == axes(h, 1) ||
        throw(DimensionMismatch("H and h must share indices: $(axes(H)) vs $(axes(h))"))
    worst = zero(_ratio_type(H, h))
    for j in axes(H, 2), i in axes(H, 1)
        Hij = magnitude(H[i, j])
        iszero(Hij) && continue
        d = h[i] * h[j]
        worst = max(worst, d > 0 ? Hij / d : oftype(worst, Inf))
    end
    return worst
end

"""
    bound_ratio(H, B) -> r

The worst-case ratio `max_ij abs(H[i,j]) / B[i,j]` over the nonzero entries of `H`. The elementwise
bound `B >= abs.(H)` holds exactly when `r <= 1`.
"""
function bound_ratio(H::AbstractMatrix, B::AbstractMatrix)
    axes(H) == axes(B) ||
        throw(DimensionMismatch("H and B must share indices: $(axes(H)) vs $(axes(B))"))
    worst = zero(_ratio_type(H, B))
    for j in axes(H, 2), i in axes(H, 1)
        Hij = magnitude(H[i, j])
        iszero(Hij) && continue
        worst = max(worst, B[i, j] > 0 ? Hij / B[i, j] : oftype(worst, Inf))
    end
    return worst
end

"""
    tightness(h, href) -> (max, geomean, nextra)

Compare a cover `h` against a reference cover `href`, coordinate by coordinate. `max` and
`geomean` summarize the profile `h[i]/href[i]` over the coordinates where `href` is positive;
`nextra` counts coordinates where the reference is zero but `h` is not. These coordinates
represent pure looseness and have no finite ratio. Both summaries are `NaN` when the reference is
zero everywhere.

The full per-coordinate profile `h ./ href` is the anisotropy view: a single scalar hides
whether looseness is spread evenly or concentrated on a few coordinates.
"""
function tightness(h::AbstractVector, href::AbstractVector)
    axes(h) == axes(href) ||
        throw(DimensionMismatch("h and href must share indices: $(axes(h)) vs $(axes(href))"))
    T = _ratio_type(h, href)
    rs = T[]
    nextra = 0
    for i in eachindex(h, href)
        if href[i] > 0
            push!(rs, h[i] / href[i])
        elseif h[i] > 0
            nextra += 1
        end
    end
    isempty(rs) && return (T(NaN), T(NaN), nextra)
    return (maximum(rs), exp(sum(log, rs) / length(rs)), nextra)
end
