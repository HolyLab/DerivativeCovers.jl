# Elementary operations recorded on the tape. Each records its value, its first partials, and
# its second partials; a program built from anything outside this set hits a MethodError.

import Base: +, -, *, /, ^, exp, log, sin, cos, tanh, sqrt, abs2
import LinearAlgebra: dot, Adjoint, Transpose

# --- binary, both TVar ---
+(x::TVar, y::TVar) = binarynode(x, y, value(x) + value(y), 1.0, 1.0, 0, 0, 0)
-(x::TVar, y::TVar) = binarynode(x, y, value(x) - value(y), 1.0, -1.0, 0, 0, 0)
*(x::TVar, y::TVar) = binarynode(x, y, value(x) * value(y), value(y), value(x), 0, 1, 0)
function /(x::TVar, y::TVar)
    u, w = value(x), value(y)
    return binarynode(x, y, u / w, 1 / w, -u / w^2, 0, -1 / w^2, 2u / w^3)
end

# --- unary minus ---
-(x::TVar) = unarynode(x, -value(x), -1.0, 0)

# --- TVar with a numeric constant, folded into a unary node ---
+(x::TVar, c::Real) = unarynode(x, value(x) + c, 1.0, 0)
+(c::Real, x::TVar) = x + c
-(x::TVar, c::Real) = unarynode(x, value(x) - c, 1.0, 0)
-(c::Real, x::TVar) = unarynode(x, c - value(x), -1.0, 0)
*(x::TVar, c::Real) = unarynode(x, value(x) * c, float(c), 0)
*(c::Real, x::TVar) = x * c
/(x::TVar, c::Real) = unarynode(x, value(x) / c, 1 / c, 0)
function /(c::Real, x::TVar)
    w = value(x)
    return unarynode(x, c / w, -c / w^2, 2c / w^3)
end

# --- power with a constant exponent ---
function ^(x::TVar, p::Real)
    v = value(x)
    return unarynode(x, v^p, p * v^(p - 1), p * (p - 1) * v^(p - 2))
end
# Disambiguate from Base's `x^n` integer-literal path.
^(x::TVar, p::Integer) = invoke(^, Tuple{TVar,Real}, x, p)

# --- unary transcendental / algebraic ---
function exp(x::TVar)
    e = exp(value(x))
    return unarynode(x, e, e, e)
end
function log(x::TVar)
    v = value(x)
    return unarynode(x, log(v), 1 / v, -1 / v^2)
end
function sin(x::TVar)
    v = value(x)
    return unarynode(x, sin(v), cos(v), -sin(v))
end
function cos(x::TVar)
    v = value(x)
    return unarynode(x, cos(v), -sin(v), -cos(v))
end
function tanh(x::TVar)
    t = tanh(value(x))
    s2 = 1 - t^2                 # sech²
    return unarynode(x, t, s2, -2t * s2)
end
function sqrt(x::TVar)
    v = value(x)
    return unarynode(x, sqrt(v), 0.5 * v^(-0.5), -0.25 * v^(-1.5))
end
abs2(x::TVar) = unarynode(x, value(x)^2, 2 * value(x), 2)

# --- inner products and matrix-vector products ---
# A variable-times-variable product records one inner-product node per output entry; a
# constant-coefficient product records one linear node per output entry, carrying no
# second-order term (see `dotnode`/`linearnode` in tape.jl).
dot(ws::AbstractVector{TVar}, xs::AbstractVector{TVar}) = dotnode(ws, xs)
dot(w::AbstractVector{<:Real}, xs::AbstractVector{TVar}) = linearnode(w, xs)
dot(ws::AbstractVector{TVar}, x::AbstractVector{<:Real}) = linearnode(x, ws)

function *(W::AbstractMatrix{TVar}, x::AbstractVector{TVar})
    axes(W, 2) == axes(x, 1) ||
        throw(DimensionMismatch("W has axes $(axes(W)), x has axes $(axes(x))"))
    return TVar[dot(view(W, m, :), x) for m in axes(W, 1)]
end
function *(W::AbstractMatrix{<:Real}, x::AbstractVector{TVar})
    axes(W, 2) == axes(x, 1) ||
        throw(DimensionMismatch("W has axes $(axes(W)), x has axes $(axes(x))"))
    return TVar[linearnode(view(W, m, :), x) for m in axes(W, 1)]
end
function *(W::AbstractMatrix{TVar}, x::AbstractVector{<:Real})
    axes(W, 2) == axes(x, 1) ||
        throw(DimensionMismatch("W has axes $(axes(W)), x has axes $(axes(x))"))
    return TVar[linearnode(x, view(W, m, :)) for m in axes(W, 1)]
end

# Disambiguate against LinearAlgebra's generic `Adjoint`/`Transpose` row-vector and `Diagonal`
# matrix-vector products, which are `AbstractMatrix` and so tie with the three methods above:
# an adjoint/transpose row vector times a vector is the same dot product `dot` already records,
# and a diagonal matrix times a vector is the elementwise product the scalar `*` already records.
*(W::Union{Adjoint{TVar,<:AbstractVector},Transpose{TVar,<:AbstractVector}},
  x::AbstractVector{TVar}) = dot(parent(W), x)
*(W::Union{Adjoint{TVar,<:AbstractVector},Transpose{TVar,<:AbstractVector}},
  x::AbstractVector{<:Real}) = dot(parent(W), x)
*(W::Union{Adjoint{<:Real,<:AbstractVector},Transpose{<:Real,<:AbstractVector}},
  x::AbstractVector{TVar}) = dot(parent(W), x)

*(W::Diagonal{TVar}, x::AbstractVector{TVar}) = W.diag .* x
*(W::Diagonal{TVar}, x::AbstractVector{<:Real}) = W.diag .* x
*(W::Diagonal{<:Real}, x::AbstractVector{TVar}) = W.diag .* x
