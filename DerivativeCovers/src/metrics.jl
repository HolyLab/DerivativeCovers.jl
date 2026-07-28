# Quality metrics for a cover: is it valid, and how loose is it?

_ratio_type(A, B) = typeof(float(one(eltype(A)) / one(eltype(B))))

"""
    cover_ratio(H, h) -> r

Worst-case ratio `max_ij abs(H[i,j]) / (h[i]*h[j])` over the nonzero entries of `H`. `h` is a
valid cover of `H` exactly when `r <= 1`. Zero entries of `H` constrain nothing and are skipped;
a nonzero entry over a vanishing `h[i]*h[j]` gives `Inf`.
"""
function cover_ratio(H::AbstractMatrix, h::AbstractVector)
    axes(H, 1) == axes(H, 2) == axes(h, 1) ||
        throw(DimensionMismatch("H and h must share indices: $(axes(H)) vs $(axes(h))"))
    worst = zero(_ratio_type(H, h))
    for j in axes(H, 2), i in axes(H, 1)
        Hij = abs(H[i, j])
        iszero(Hij) && continue
        d = h[i] * h[j]
        worst = max(worst, d > 0 ? Hij / d : oftype(worst, Inf))
    end
    return worst
end

"""
    bound_ratio(H, B) -> r

Worst-case ratio `max_ij abs(H[i,j]) / B[i,j]` over the nonzero entries of `H`. The elementwise
bound holds, `B >= abs.(H)`, exactly when `r <= 1`.
"""
function bound_ratio(H::AbstractMatrix, B::AbstractMatrix)
    axes(H) == axes(B) ||
        throw(DimensionMismatch("H and B must share indices: $(axes(H)) vs $(axes(B))"))
    worst = zero(_ratio_type(H, B))
    for j in axes(H, 2), i in axes(H, 1)
        Hij = abs(H[i, j])
        iszero(Hij) && continue
        worst = max(worst, B[i, j] > 0 ? Hij / B[i, j] : oftype(worst, Inf))
    end
    return worst
end

"""
    rank1_ratio(B, h) -> r

Worst-case ratio `max_ij B[i,j] / (h[i]*h[j])` over the nonzero entries of `B`. The rank-1 cover
dominates the elementwise bound, `h*h' >= B`, exactly when `r <= 1`.
"""
function rank1_ratio(B::AbstractMatrix, h::AbstractVector)
    axes(B, 1) == axes(B, 2) == axes(h, 1) ||
        throw(DimensionMismatch("B and h must share indices: $(axes(B)) vs $(axes(h))"))
    worst = zero(_ratio_type(B, h))
    for j in axes(B, 2), i in axes(B, 1)
        iszero(B[i, j]) && continue
        d = h[i] * h[j]
        worst = max(worst, d > 0 ? B[i, j] / d : oftype(worst, Inf))
    end
    return worst
end

"""
    tightness(h, href) -> (max, geomean, nextra)

Compare a cover `h` against a reference cover `href`, coordinate by coordinate. `max` and
`geomean` summarize the profile `h[i]/href[i]` over the coordinates where `href` is positive;
`nextra` counts the coordinates the reference leaves at zero but `h` does not, which are pure
looseness and have no finite ratio. Both summaries are `NaN` when the reference is everywhere
zero.

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
