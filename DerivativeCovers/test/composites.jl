# Generic array fixtures used with both TVar recording and ForwardDiff.

"""
    softmax(x::AbstractVector)

`exp.(x) ./ sum(exp.(x))`, computed as `e = exp.(x .- m); e ./ sum(e)` with `m = maximum(value, x)`
detached from the tape (a plain number, not a `TVar`). Subtracting a constant from every entry
does not change softmax, so the detachment is exact; it also keeps the recorded graph — and hence
the cover — identical to the unshifted form while giving the usual protection against overflow in
`exp`. Recording the subtraction on the tape instead would inflate the cover: the max entry's
derivative contributions cancel mathematically, but an absolute-value sweep cannot see the
cancellation and bounds each contribution separately.
"""
function softmax(x::AbstractVector)
    isempty(x) && throw(ArgumentError("softmax of an empty vector is undefined"))
    m = maximum(value, x)
    e = exp.(x .- m)
    return e ./ sum(e)
end

"""
    layernorm(x::AbstractVector; epsilon::Real=0)

`(x .- mean(x)) ./ sqrt(var(x) + epsilon)`, using the biased (divide-by-`n`) variance. Unlike
`softmax`'s max, the mean and variance carry real derivatives and must be recorded on the
tape rather than detached. `epsilon` defaults to `0`, so a constant input yields a non-finite
result rather than a silently ill-scaled one.
"""
function layernorm(x::AbstractVector; epsilon::Real = 0)
    n = length(x)
    n >= 2 || throw(ArgumentError("layernorm needs at least 2 entries"))
    mu = sum(x) / n
    d = x .- mu
    s2 = sum(abs2, d) / n
    return d ./ sqrt(s2 + epsilon)
end
