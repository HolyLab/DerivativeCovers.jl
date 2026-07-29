# Elementary operations recorded on the tape. Each records its value, its first partials, and
# its second partials; a program built from anything outside this set hits a MethodError.

import Base: +, -, *, /, ^, exp, log, sin, cos, tan, atan, tanh, sqrt, inv, abs2, adjoint,
    transpose
import LinearAlgebra: dot, Adjoint, Transpose, AbstractTriangular

# A TVar is a broadcast scalar: `xs ./ s` with `s::TVar` broadcasts `s` against every entry of
# `xs` instead of falling back to Base's default `collect(s)`, which errors on a scalar TVar.
Base.broadcastable(x::TVar) = Ref(x)

# A TVar is a real scalar, so it is its own adjoint and its own transpose. Both are needed: they
# make `Base.promote_op(adjoint, TVar)` and `Base.promote_op(transpose, TVar)` be `TVar`, so `v'`
# and `transpose(v)` on a `Vector{TVar}` produce wrappers with a `TVar` element type that the
# row-vector product methods below can dispatch on. Without them the element type is `Union{}` and
# any wrapper entry that is itself transposed on access throws a `MethodError`.
adjoint(x::TVar) = x
transpose(x::TVar) = x

# --- constant folding --------------------------------------------------------------------------
# A constant TVar (`isconstant`) has a value and no tape, so it enters a node as a coefficient
# rather than as a parent. Every method below that takes a TVar therefore begins by disposing of
# the constant cases: `unfold` strips a constant to its value, which sends the operation to the
# `Real` method that records against the operand still carrying a tape, and `fold` restores the
# TVar type when both operands were constants and no node was recorded at all. Constants are thus
# closed under the elementals, which is what lets a `Vector{TVar}` hold them.
unfold(x::TVar) = isconstant(x) ? value(x) : x
fold(x::TVar) = x
fold(v::Real) = TVar(v)
eitherconstant(x::TVar, y::TVar) = isconstant(x) | isconstant(y)

# --- binary, both TVar ---
+(x::TVar, y::TVar) = eitherconstant(x, y) ? fold(unfold(x) + unfold(y)) :
    binarynode(x, y, value(x) + value(y), 1.0, 1.0, 0, 0, 0)
-(x::TVar, y::TVar) = eitherconstant(x, y) ? fold(unfold(x) - unfold(y)) :
    binarynode(x, y, value(x) - value(y), 1.0, -1.0, 0, 0, 0)
*(x::TVar, y::TVar) = eitherconstant(x, y) ? fold(unfold(x) * unfold(y)) :
    binarynode(x, y, value(x) * value(y), value(y), value(x), 0, 1, 0)
function /(x::TVar, y::TVar)
    eitherconstant(x, y) && return fold(unfold(x) / unfold(y))
    u, w = value(x), value(y)
    return binarynode(x, y, u / w, 1 / w, -u / w^2, 0, -1 / w^2, 2u / w^3)
end

# --- unary sign ---
+(x::TVar) = x
-(x::TVar) = isconstant(x) ? TVar(-value(x)) : unarynode(x, -value(x), -1.0, 0)

# --- TVar with a numeric constant, folded into a unary node ---
+(x::TVar, c::Real) = isconstant(x) ? TVar(value(x) + c) : unarynode(x, value(x) + c, 1.0, 0)
+(c::Real, x::TVar) = x + c
-(x::TVar, c::Real) = isconstant(x) ? TVar(value(x) - c) : unarynode(x, value(x) - c, 1.0, 0)
-(c::Real, x::TVar) = isconstant(x) ? TVar(c - value(x)) : unarynode(x, c - value(x), -1.0, 0)
*(x::TVar, c::Real) = isconstant(x) ? TVar(value(x) * c) : unarynode(x, value(x) * c, float(c), 0)
*(c::Real, x::TVar) = x * c
/(x::TVar, c::Real) = isconstant(x) ? TVar(value(x) / c) : unarynode(x, value(x) / c, 1 / c, 0)
# A TVar scales an array of constants entrywise, as any scalar does. `*` on a scalar and an array
# is only defined for `Number`s, and a TVar is a tape node rather than a Number.
*(x::TVar, c::AbstractArray{<:Real}) = x .* c
*(c::AbstractArray{<:Real}, x::TVar) = c .* x
function /(c::Real, x::TVar)
    isconstant(x) && return TVar(c / value(x))
    w = value(x)
    return unarynode(x, c / w, -c / w^2, 2c / w^3)
end

# --- values of the element type obtained from numbers ---
# `zero(eltype(x))`, `one(eltype(x))` and `eltype(x)(c)` are how generic numeric code asks for a
# typed literal, and `convert` is how one lands in a container of that type. Each returns a
# constant TVar: a value with no tape, which the arithmetic above folds into a node on first
# contact with a recording TVar. The type-level forms have no tape they could refer to, so
# constants are what makes these definable at all.
Base.zero(::Type{TVar}) = TVar(0.0)
Base.one(::Type{TVar}) = TVar(1.0)
Base.oneunit(::Type{TVar}) = TVar(1.0)
Base.zero(x::TVar) = TVar(0.0)
Base.one(x::TVar) = TVar(1.0)
Base.oneunit(x::TVar) = TVar(1.0)
Base.convert(::Type{TVar}, c::Real) = TVar(c)

# --- power with a constant exponent ---
# A leading factor of zero makes the whole term vanish identically: `x^0` is constant and `x^1`
# is linear, so the derivatives those factors multiply are zero at every `v`. Applying the factor
# before the power keeps that true at `v == 0`, where the power itself is singular and the product
# would otherwise be `0 * Inf`. A nonzero factor leaves a genuine singularity singular, as
# `sqrt` and `inv` are at zero.
function ^(x::TVar, p::Real)
    isconstant(x) && return TVar(value(x)^p)
    v = value(x)
    d = iszero(p) ? zero(v) : p * v^(p - 1)
    C = iszero(p) || isone(p) ? zero(v) : p * (p - 1) * v^(p - 2)
    return unarynode(x, v^p, d, C)
end
# Disambiguate from Base's `x^n` integer-literal path.
^(x::TVar, p::Integer) = invoke(^, Tuple{TVar, Real}, x, p)

# --- power with a variable exponent ---
# `u^w = exp(w*log(u))` is real-differentiable only for `u > 0`; a nonpositive base is a domain
# error rather than a NaN propagated into the cover.
function ^(x::TVar, y::TVar)
    eitherconstant(x, y) && return fold(unfold(x)^unfold(y))
    u, w = value(x), value(y)
    u > 0 || throw(DomainError(u, "x^y with a variable exponent requires a positive base"))
    v, lu = u^w, log(u)
    return binarynode(
        x, y, v, w * u^(w - 1), v * lu,
        w * (w - 1) * u^(w - 2), u^(w - 1) * (1 + w * lu), v * lu^2
    )
end
function ^(c::Real, x::TVar)
    isconstant(x) && return TVar(float(c)^value(x))
    c > 0 || throw(DomainError(c, "c^x with a variable exponent requires a positive base"))
    v, lc = float(c)^value(x), log(c)
    return unarynode(x, v, v * lc, v * lc^2)
end

# --- unary transcendental / algebraic ---
# Each folds a constant argument to a constant result, as the arithmetic above does.
function exp(x::TVar)
    isconstant(x) && return TVar(exp(value(x)))
    e = exp(value(x))
    return unarynode(x, e, e, e)
end
function log(x::TVar)
    isconstant(x) && return TVar(log(value(x)))
    v = value(x)
    return unarynode(x, log(v), 1 / v, -1 / v^2)
end
function sin(x::TVar)
    isconstant(x) && return TVar(sin(value(x)))
    v = value(x)
    return unarynode(x, sin(v), cos(v), -sin(v))
end
function cos(x::TVar)
    isconstant(x) && return TVar(cos(value(x)))
    v = value(x)
    return unarynode(x, cos(v), -sin(v), -cos(v))
end
function tan(x::TVar)
    isconstant(x) && return TVar(tan(value(x)))
    t = tan(value(x))
    s2 = 1 + t^2                 # sec²
    return unarynode(x, t, s2, 2t * s2)
end
function atan(x::TVar)
    isconstant(x) && return TVar(atan(value(x)))
    v = value(x)
    d = 1 / (1 + v^2)
    return unarynode(x, atan(v), d, -2v * d^2)
end
function tanh(x::TVar)
    isconstant(x) && return TVar(tanh(value(x)))
    t = tanh(value(x))
    s2 = 1 - t^2                 # sech²
    return unarynode(x, t, s2, -2t * s2)
end
function inv(x::TVar)
    isconstant(x) && return TVar(inv(value(x)))
    v = value(x)
    return unarynode(x, inv(v), -1 / v^2, 2 / v^3)
end
function sqrt(x::TVar)
    isconstant(x) && return TVar(sqrt(value(x)))
    v = value(x)
    return unarynode(x, sqrt(v), 0.5 * v^(-0.5), -0.25 * v^(-1.5))
end
abs2(x::TVar) = isconstant(x) ? TVar(value(x)^2) : unarynode(x, value(x)^2, 2 * value(x), 2)

# --- inner products and matrix-vector products ---
# A variable-times-variable product records one inner-product node per output entry; a
# constant-coefficient product records one linear node per output entry, carrying no
# second-order term (see `dotnode`/`linearnode` in tape.jl).
# A `Vector{TVar}` may hold constants (`eltype(x)[...]`), and a side that is entirely constant is
# a coefficient vector: it records as a linear node, exactly as a `Vector{Float64}` would. The node
# forms are per-vector, not per-entry, so an operand that mixes the two states is split by pair
# instead (`_partitioned_dot`).
function dot(ws::AbstractVector{TVar}, xs::AbstractVector{TVar})
    wconst, xconst = count(isconstant, ws), count(isconstant, xs)
    wconst == 0 && xconst == 0 && return dotnode(ws, xs)
    xconst == 0 && wconst == length(ws) && return linearnode(map(value, ws), xs)
    wconst == 0 && xconst == length(xs) && return linearnode(map(value, xs), ws)
    return _partitioned_dot(ws, xs)
end

# An inner product whose operands mix constant and recording TVars, split by the state of each
# pair: pairs recording on both sides form one inner-product node, pairs with one constant side
# form a linear node against those constants as coefficients, and pairs constant on both sides an
# additive offset. A pair with a constant side has no second-order term, so it contributes no atom
# under any reading, and the additions that recombine the parts are transparent to every sweep:
# the atom set is the one the scalarized product gives, for at most three nodes instead of 2q.
function _partitioned_dot(ws::AbstractVector{TVar}, xs::AbstractVector{TVar})
    idxs = eachindex(ws, xs)
    isempty(idxs) && throw(ArgumentError("cannot record an empty inner product"))
    wpair, xpair = TVar[], TVar[]     # both sides recording
    wcoef, wrest = Float64[], TVar[]  # ws[j] constant, xs[j] recording
    xcoef, xrest = Float64[], TVar[]  # xs[j] constant, ws[j] recording
    offset = 0.0
    for j in idxs
        w, x = ws[j], xs[j]
        if isconstant(w) && isconstant(x)
            offset += value(w) * value(x)
        elseif isconstant(w)
            push!(wcoef, value(w)); push!(wrest, x)
        elseif isconstant(x)
            push!(xcoef, value(x)); push!(xrest, w)
        else
            push!(wpair, w); push!(xpair, x)
        end
    end
    parts = TVar[]
    isempty(wpair) || push!(parts, dotnode(wpair, xpair))
    isempty(wrest) || push!(parts, linearnode(wcoef, wrest))
    isempty(xrest) || push!(parts, linearnode(xcoef, xrest))
    isempty(parts) && return TVar(offset)   # every pair constant on both sides
    total = length(parts) == 1 ? only(parts) : linearnode(ones(length(parts)), parts)
    return iszero(offset) ? total : total + offset
end
dot(w::AbstractVector{<:Real}, xs::AbstractVector{TVar}) = _linear_dot(w, xs)
dot(ws::AbstractVector{TVar}, x::AbstractVector{<:Real}) = _linear_dot(x, ws)

# An inner product against constant coefficients. `linearnode` takes recording operands only, but a
# `TVar` container can hold constants — the structural zeros and unit diagonal a triangular wrapper
# supplies on `getindex`, or a literal placed in a container by generic code — so those fold into
# an additive offset and only the recording entries become parents. A constant side carries no
# second-order term, so this records the atom set the scalarized product gives.
function _linear_dot(coeffs::AbstractVector{<:Real}, xs::AbstractVector{TVar})
    any(isconstant, xs) || return linearnode(coeffs, xs)
    idxs = eachindex(coeffs, xs)
    rcoef, rest = Float64[], TVar[]
    offset = 0.0
    for j in idxs
        x = xs[j]
        if isconstant(x)
            offset += coeffs[j] * value(x)
        else
            push!(rcoef, float(coeffs[j])); push!(rest, x)
        end
    end
    isempty(rest) && return TVar(offset)   # every entry constant
    total = linearnode(rcoef, rest)
    return iszero(offset) ? total : total + offset
end

# Row-by-row matrix-vector product. Each row dispatches to the `dot` methods above, so a
# constant row or a constant `x` records a `linearnode` and a mixed pair a `dotnode`.
function matvecnodes(W::AbstractMatrix, x::AbstractVector)
    axes(W, 2) == axes(x, 1) ||
        throw(DimensionMismatch("W has axes $(axes(W)), x has axes $(axes(x))"))
    return TVar[dot(view(W, m, :), x) for m in axes(W, 1)]
end

*(W::AbstractMatrix{TVar}, x::AbstractVector{TVar}) = matvecnodes(W, x)
*(W::AbstractMatrix{<:Real}, x::AbstractVector{TVar}) = matvecnodes(W, x)
*(W::AbstractMatrix{TVar}, x::AbstractVector{<:Real}) = matvecnodes(W, x)

# Disambiguate against LinearAlgebra's generic `Adjoint`/`Transpose` row-vector and `Diagonal`
# matrix-vector products, which are `AbstractMatrix` and so tie with the three methods above:
# an adjoint/transpose row vector times a vector is the same dot product `dot` already records,
# and a diagonal matrix times a vector is the elementwise product the scalar `*` already records.
*(
    W::Union{Adjoint{TVar, <:AbstractVector}, Transpose{TVar, <:AbstractVector}},
    x::AbstractVector{TVar}
) = dot(parent(W), x)
*(
    W::Union{Adjoint{TVar, <:AbstractVector}, Transpose{TVar, <:AbstractVector}},
    x::AbstractVector{<:Real}
) = dot(parent(W), x)
*(
    W::Union{Adjoint{<:Real, <:AbstractVector}, Transpose{<:Real, <:AbstractVector}},
    x::AbstractVector{TVar}
) = dot(parent(W), x)

# Elementwise product, not a row-by-row one: routing a `Diagonal` through `matvecnodes` would
# record n dense linear nodes at O(n²) cost. Broadcasting expands a length-1 or otherwise
# mismatched operand rather than rejecting it, so the axes must be checked here.
function diagvecnodes(W::Diagonal, x::AbstractVector)
    axes(W, 2) == axes(x, 1) ||
        throw(DimensionMismatch("W has axes $(axes(W)), x has axes $(axes(x))"))
    return diag(W) .* x
end

*(W::Diagonal{TVar}, x::AbstractVector{TVar}) = diagvecnodes(W, x)
*(W::Diagonal{TVar}, x::AbstractVector{<:Real}) = diagvecnodes(W, x)
*(W::Diagonal{<:Real}, x::AbstractVector{TVar}) = diagvecnodes(W, x)

# LinearAlgebra's triangular matrix-vector product likewise ties with the three `AbstractMatrix`
# methods. Triangular structure needs no special handling: `matvecnodes` reads each row through
# `getindex`, which supplies the structural zeros and the unit diagonal.
*(W::AbstractTriangular{TVar}, x::AbstractVector{TVar}) = matvecnodes(W, x)
*(W::AbstractTriangular{TVar}, x::AbstractVector{<:Real}) = matvecnodes(W, x)
*(W::AbstractTriangular{<:Real}, x::AbstractVector{TVar}) = matvecnodes(W, x)

# A sum of TVars records as one linear n-ary node (see `linearnode`) rather than a chain of
# n-1 binary additions: additions relay adjoints and absolute sweeps with unit weight and
# contribute no atoms, so every cover quantity agrees with the chained form to roundoff. The
# gain is tape length: one node instead of n-1.
function Base.sum(xs::AbstractArray{TVar})
    isempty(xs) && throw(ArgumentError("cannot record an empty sum"))
    # `linearnode` pairs coefficients with terms by index, so the unit coefficients carry `v`'s
    # axes.
    v = vec(xs)
    any(isconstant, xs) || return linearnode(fill!(similar(v, Float64), 1), v)
    # Constants in the sum are an additive offset, which the unary constant node carries; a sum of
    # constants alone has nothing to record and stays a constant.
    recording = [x for x in xs if !isconstant(x)]
    offset = sum(value(x) for x in xs if isconstant(x))
    isempty(recording) && return TVar(offset)
    total = length(recording) == 1 ? only(recording) : linearnode(ones(length(recording)), recording)
    return iszero(offset) ? total : total + offset
end
Base.sum(f, xs::AbstractArray{TVar}) = sum(map(f, xs))
