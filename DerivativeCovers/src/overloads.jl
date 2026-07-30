# Operations supported by TVar recording.

import Base: +, -, *, /, ^, exp, log, sin, cos, tan, atan, tanh, sqrt, inv, abs2, adjoint,
    transpose
import LinearAlgebra: dot, Adjoint, Transpose, AbstractTriangular
# The in-place rules below extend LinearAlgebra's generic product entry points, which are reached
# by qualification rather than by name.
import LinearAlgebra

# Treat TVar as a scalar during broadcasting.
Base.broadcastable(x::TVar) = Ref(x)

# Defining scalar adjoint and transpose keeps row-vector wrappers typed as TVar.
adjoint(x::TVar) = x
transpose(x::TVar) = x

# --- constant folding --------------------------------------------------------------------------
# Constants have no tape node. `unfold` exposes their values before dispatch; `fold` wraps a
# constant-only result back into TVar so heterogeneous numeric code remains type-stable.
unfold(x::TVar) = isconstant(x) ? value(x) : x
fold(x::TVar) = x
fold(v::Real) = TVar(v)
eitherconstant(x::TVar, y::TVar) = isconstant(x) | isconstant(y)

# --- binary, both TVar ---
+(x::TVar, y::TVar) = eitherconstant(x, y) ? fold(unfold(x) + unfold(y)) :
    binarynode(x, y, value(x) + value(y), 1, 1, 0, 0, 0)
-(x::TVar, y::TVar) = eitherconstant(x, y) ? fold(unfold(x) - unfold(y)) :
    binarynode(x, y, value(x) - value(y), 1, -1, 0, 0, 0)
*(x::TVar, y::TVar) = eitherconstant(x, y) ? fold(unfold(x) * unfold(y)) :
    binarynode(x, y, value(x) * value(y), value(y), value(x), 0, 1, 0)
function /(x::TVar, y::TVar)
    eitherconstant(x, y) && return fold(unfold(x) / unfold(y))
    u, w = value(x), value(y)
    return binarynode(x, y, u / w, 1 / w, -u / w^2, 0, -1 / w^2, 2u / w^3)
end

# --- unary sign ---
+(x::TVar) = x
-(x::TVar) = isconstant(x) ? TVar(-value(x)) : unarynode(x, -value(x), -1, 0)

# --- TVar with a numeric constant, folded into a unary node ---
# An operation that leaves the operand alone returns it rather than recording a node. Generic
# numerical code seeds accumulators with `zero(T)` and scales by unit factors, so these identities
# are common enough to lengthen a tape appreciably, and the node they would record has unit first
# partial and no curvature: dropping it leaves every derivative unchanged.
#
# An additive shift qualifies whenever it leaves the value alone, which also declines the case
# where it would turn a negative zero positive. A scaling qualifies only for a unit factor: the
# coefficient is the node's first partial, so a product that happens to leave the value alone --
# any factor at all, when the value is zero -- still has a derivative to record.
function +(x::TVar, c::Real)
    isconstant(x) && return TVar(value(x) + c)
    v = value(x)
    return v + c === v ? x : unarynode(x, v + c, 1, 0)
end
+(c::Real, x::TVar) = x + c
function -(x::TVar, c::Real)
    isconstant(x) && return TVar(value(x) - c)
    v = value(x)
    return v - c === v ? x : unarynode(x, v - c, 1, 0)
end
-(c::Real, x::TVar) = isconstant(x) ? TVar(c - value(x)) : unarynode(x, c - value(x), -1, 0)
*(x::TVar, c::Real) = isconstant(x) ? TVar(value(x) * c) :
    isone(c) ? x : unarynode(x, value(x) * c, c, 0)
*(c::Real, x::TVar) = x * c
/(x::TVar, c::Real) = isconstant(x) ? TVar(value(x) / c) :
    isone(c) ? x : unarynode(x, value(x) / c, 1 / c, 0)
# TVar is not a Number, so define scalar-array multiplication explicitly.
*(x::TVar, c::AbstractArray{<:Real}) = x .* c
*(c::AbstractArray{<:Real}, x::TVar) = c .* x
function /(c::Real, x::TVar)
    isconstant(x) && return TVar(c / value(x))
    w = value(x)
    return unarynode(x, c / w, -c / w^2, 2c / w^3)
end

# --- values of the element type obtained from numbers ---
# Generic code obtains typed literals through zero, one, conversion, and constructors. These
# return constant TVars because no tape is available at the type level. Written against
# `TVar{T}` they keep the program's precision, since generic code reaches them through
# `eltype(x)`; the unparameterized forms have no `T` to preserve and default to Float64.
Base.zero(::Type{TVar{T}}) where {T} = TVar{T}(0)
Base.one(::Type{TVar{T}}) where {T} = TVar{T}(1)
Base.oneunit(::Type{TVar{T}}) where {T} = TVar{T}(1)
Base.zero(::Type{TVar}) = TVar(0.0)
Base.one(::Type{TVar}) = TVar(1.0)
Base.oneunit(::Type{TVar}) = TVar(1.0)
Base.zero(x::TVar) = zero(typeof(x))
Base.one(x::TVar) = one(typeof(x))
Base.oneunit(x::TVar) = oneunit(typeof(x))
Base.convert(::Type{TVar}, c::Real) = TVar(c)
Base.convert(::Type{TVar{T}}, c::Real) where {T} = TVar{T}(c)

# --- power with a constant exponent ---
# Apply zero derivative factors before singular powers so x^0 and x^1 remain constant and linear
# at x = 0. Genuine singularities remain singular.
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
# Variable-variable products use inner-product nodes; constant coefficients use linear nodes.
# Mixed TVar vectors are partitioned by pair.
function dot(ws::AbstractVector{<:TVar}, xs::AbstractVector{<:TVar})
    wconst, xconst = count(isconstant, ws), count(isconstant, xs)
    wconst == 0 && xconst == 0 && return dotnode(ws, xs)
    xconst == 0 && wconst == length(ws) && return linearnode(map(value, ws), xs)
    wconst == 0 && xconst == length(xs) && return linearnode(map(value, xs), ws)
    return _partitioned_dot(ws, xs)
end

# Partition a mixed dot product into a bilinear node, a linear node, and a constant offset. This
# preserves the scalarized atom set with at most three nodes. The staging vectors take their
# element types from the operands, so the coefficients reach the recorders in the operands' own
# precision; `offset` accumulates as a constant TVar, which folds rather than recording a node.
function _partitioned_dot(ws::AbstractVector{<:TVar}, xs::AbstractVector{<:TVar})
    idxs = eachindex(ws, xs)
    isempty(idxs) && throw(ArgumentError("cannot record an empty inner product"))
    wpair, xpair = similar(ws, 0), similar(xs, 0)     # both sides recording
    wconst, wrest = similar(ws, 0), similar(xs, 0)    # ws[j] constant, xs[j] recording
    xconst, xrest = similar(xs, 0), similar(ws, 0)    # xs[j] constant, ws[j] recording
    offset = zero(eltype(ws))
    for j in idxs
        w, x = ws[j], xs[j]
        if isconstant(w) && isconstant(x)
            offset += w * x
        elseif isconstant(w)
            push!(wconst, w); push!(wrest, x)
        elseif isconstant(x)
            push!(xconst, x); push!(xrest, w)
        else
            push!(wpair, w); push!(xpair, x)
        end
    end
    parts = similar(ws, 0)
    isempty(wpair) || push!(parts, dotnode(wpair, xpair))
    isempty(wrest) || push!(parts, linearnode(map(value, wconst), wrest))
    isempty(xrest) || push!(parts, linearnode(map(value, xconst), xrest))
    isempty(parts) && return offset   # every pair constant on both sides
    total = length(parts) == 1 ? only(parts) : linearnode(ones(Bool, length(parts)), parts)
    # A zero offset leaves the value alone, which `+` declines to record.
    return total + offset
end
dot(w::AbstractVector{<:Real}, xs::AbstractVector{<:TVar}) = _linear_dot(w, xs)
dot(ws::AbstractVector{<:TVar}, x::AbstractVector{<:Real}) = _linear_dot(x, ws)

# Fold constant entries into an offset; only recording entries become parents of the linear node.
function _linear_dot(coeffs::AbstractVector{<:Real}, xs::AbstractVector{<:TVar})
    any(isconstant, xs) || return linearnode(coeffs, xs)
    idxs = eachindex(coeffs, xs)
    rcoef, rest = similar(coeffs, 0), similar(xs, 0)
    offset = zero(eltype(xs))
    for j in idxs
        x = xs[j]
        if isconstant(x)
            offset += coeffs[j] * value(x)
        else
            push!(rcoef, coeffs[j]); push!(rest, x)
        end
    end
    isempty(rest) && return offset   # every entry constant
    return linearnode(rcoef, rest) + offset
end

# Row-by-row matrix-vector product. Each row dispatches to the `dot` methods above, so a
# constant row or a constant `x` records a `linearnode` and a mixed pair a `dotnode`. The
# comprehension takes its element type from those nodes, which carry the tape's precision.
function matvecnodes(W::AbstractMatrix, x::AbstractVector)
    axes(W, 2) == axes(x, 1) ||
        throw(DimensionMismatch("W has axes $(axes(W)), x has axes $(axes(x))"))
    return [dot(view(W, m, :), x) for m in axes(W, 1)]
end

*(W::AbstractMatrix{<:TVar}, x::AbstractVector{<:TVar}) = matvecnodes(W, x)
*(W::AbstractMatrix{<:Real}, x::AbstractVector{<:TVar}) = matvecnodes(W, x)
*(W::AbstractMatrix{<:TVar}, x::AbstractVector{<:Real}) = matvecnodes(W, x)

# Resolve dispatch ties with LinearAlgebra's AbstractMatrix methods.
*(
    W::Union{Adjoint{<:TVar, <:AbstractVector}, Transpose{<:TVar, <:AbstractVector}},
    x::AbstractVector{<:TVar}
) = dot(parent(W), x)
*(
    W::Union{Adjoint{<:TVar, <:AbstractVector}, Transpose{<:TVar, <:AbstractVector}},
    x::AbstractVector{<:Real}
) = dot(parent(W), x)
*(
    W::Union{Adjoint{<:Real, <:AbstractVector}, Transpose{<:Real, <:AbstractVector}},
    x::AbstractVector{<:TVar}
) = dot(parent(W), x)

# A diagonal product is elementwise; the explicit axes check prevents broadcast expansion.
function diagvecnodes(W::Diagonal, x::AbstractVector)
    axes(W, 2) == axes(x, 1) ||
        throw(DimensionMismatch("W has axes $(axes(W)), x has axes $(axes(x))"))
    return diag(W) .* x
end

*(W::Diagonal{<:TVar}, x::AbstractVector{<:TVar}) = diagvecnodes(W, x)
*(W::Diagonal{<:TVar}, x::AbstractVector{<:Real}) = diagvecnodes(W, x)
*(W::Diagonal{<:Real}, x::AbstractVector{<:TVar}) = diagvecnodes(W, x)

# Triangular wrappers need only dispatch disambiguation; getindex supplies their structural values.
*(W::AbstractTriangular{<:TVar}, x::AbstractVector{<:TVar}) = matvecnodes(W, x)
*(W::AbstractTriangular{<:TVar}, x::AbstractVector{<:Real}) = matvecnodes(W, x)
*(W::AbstractTriangular{<:Real}, x::AbstractVector{<:TVar}) = matvecnodes(W, x)

# Entry-by-entry matrix-matrix product: one node per output entry, as `matvecnodes` records one
# per output row. `mul!` is the only route to it. A `*` method of its own would tie with
# LinearAlgebra's algebra of lazily wrapped vectors, which are `AbstractMatrix`es too, and
# LinearAlgebra's fallback `*` already allocates a destination and calls `mul!`.
function matmatnodes(W::AbstractMatrix, X::AbstractMatrix)
    axes(W, 2) == axes(X, 1) ||
        throw(DimensionMismatch("W has axes $(axes(W)), X has axes $(axes(X))"))
    return [dot(view(W, m, :), view(X, :, n)) for m in axes(W, 1), n in axes(X, 2)]
end

# --- in-place products -------------------------------------------------------------------------
# `mul!` records what the corresponding `*` records. Left to LinearAlgebra's generic implementation
# it accumulates each output from scalar `*` and `+`, spending a node per multiply-add and three
# more per output on the typed zero it seeds the accumulator with, so an `m × n` matrix-vector
# product costs `O(m*n)` nodes where the row-by-row rule costs `O(m)`.
#
# The rules attach to `generic_matvecmul!` and `generic_matmatmul!`, the entry points LinearAlgebra
# routes a product to once it has decided no specialized implementation applies, rather than to
# `mul!` itself. Structured wrappers own `mul!` methods, and which ones they own varies by Julia
# version, so a rule on `mul!` would need an entry per wrapper type just to break the ties. These
# entry points sit below all of that: one method per product shape, and nothing to tie with. The
# wrapper LinearAlgebra strips from an operand arrives as the character `tA`, and `wrap` restores
# it, so the recorders see the matrix the caller passed.
#
# The wrappers that keep a `mul!` of their own -- triangular, `Bidiagonal`, `Tridiagonal` -- record
# their scalarized form instead. That covers the same Hessian from a longer tape; their `*` methods
# above still record one node per output row.

# Write a recorded product into `y`. The destination's axes must be the product's: LinearAlgebra's
# out-of-place `*` sizes its matrix destination one-based, so an offset operand reaching `mul!`
# that way is a mismatch rather than a result silently relabelled.
function storeproduct!(y::AbstractArray{<:TVar}, z::AbstractArray, alpha::Number, beta::Number)
    axes(y) == axes(z) ||
        throw(DimensionMismatch("mul! destination has axes $(axes(y)), product has axes $(axes(z))"))
    iszero(beta) || return y .= alpha .* z .+ beta .* y
    # A unit `alpha` is the 3-argument call; multiplying through would record a node per output.
    return isone(alpha) ? (y .= z) : (y .= alpha .* z)
end

recordmatvec!(y, tA, W, x, alpha, beta) =
    storeproduct!(y, matvecnodes(LinearAlgebra.wrap(W, tA), x), alpha, beta)
recordmatmat!(Y, tA, tB, W, X, alpha, beta) =
    storeproduct!(Y, matmatnodes(LinearAlgebra.wrap(W, tA), LinearAlgebra.wrap(X, tB)), alpha, beta)

# Julia 1.11 replaced the trailing `MulAddMul` of both entry points with a scalar pair.
@static if VERSION >= v"1.11"
    LinearAlgebra.generic_matvecmul!(
        y::AbstractVector{<:TVar}, tA, W::AbstractVecOrMat, x::AbstractVector,
        alpha::Number, beta::Number
    ) = recordmatvec!(y, tA, W, x, alpha, beta)

    LinearAlgebra.generic_matmatmul!(
        Y::AbstractMatrix{<:TVar}, tA, tB, W::AbstractVecOrMat, X::AbstractVecOrMat,
        alpha::Number, beta::Number
    ) = recordmatmat!(Y, tA, tB, W, X, alpha, beta)
else
    LinearAlgebra.generic_matvecmul!(
        y::AbstractVector{<:TVar}, tA, W::AbstractVecOrMat, x::AbstractVector,
        add::LinearAlgebra.MulAddMul
    ) = recordmatvec!(y, tA, W, x, add.alpha, add.beta)

    LinearAlgebra.generic_matmatmul!(
        Y::AbstractMatrix{<:TVar}, tA, tB, W::AbstractVecOrMat, X::AbstractVecOrMat,
        add::LinearAlgebra.MulAddMul
    ) = recordmatmat!(Y, tA, tB, W, X, add.alpha, add.beta)
end

# Record a sum as one linear node instead of n-1 addition nodes.
function Base.sum(xs::AbstractArray{<:TVar})
    isempty(xs) && throw(ArgumentError("cannot record an empty sum"))
    # `linearnode` pairs coefficients with terms by index, so the unit coefficients carry `v`'s
    # axes. It converts them into the tape's element type, and a unit is exact in every type, so
    # the cheapest one that spells it will do.
    v = vec(xs)
    any(isconstant, xs) || return linearnode(fill!(similar(v, Bool), true), v)
    # Constants in the sum are an additive offset, which the unary constant node carries; a sum of
    # constants alone has nothing to record and stays a constant.
    recording = [x for x in xs if !isconstant(x)]
    offset = sum(x for x in xs if isconstant(x))
    isempty(recording) && return offset
    total = length(recording) == 1 ? only(recording) :
        linearnode(ones(Bool, length(recording)), recording)
    return total + offset
end
Base.sum(f, xs::AbstractArray{<:TVar}) = sum(map(f, xs))
