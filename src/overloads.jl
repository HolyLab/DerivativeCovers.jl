# Elementary operations recorded on the tape. Each records its value, its first partials, and
# its second partials; a program built from anything outside this set hits a MethodError.

import Base: +, -, *, /, ^, exp, log, sin, cos, tan, atan, tanh, sqrt, inv, abs2, adjoint
import LinearAlgebra: dot, Adjoint, Transpose

# A TVar is a broadcast scalar: `xs ./ s` with `s::TVar` broadcasts `s` against every entry of
# `xs` instead of falling back to Base's default `collect(s)`, which errors on a scalar TVar.
Base.broadcastable(x::TVar) = Ref(x)

# A TVar is a real scalar, so it is its own adjoint. Defining this also makes
# `Base.promote_op(adjoint, TVar)` be `TVar`, so `v'` on a `Vector{TVar}` produces an
# `Adjoint{TVar,...}` that the row-vector product methods below can dispatch on.
adjoint(x::TVar) = x

# --- binary, both TVar ---
+(x::TVar, y::TVar) = binarynode(x, y, value(x) + value(y), 1.0, 1.0, 0, 0, 0)
-(x::TVar, y::TVar) = binarynode(x, y, value(x) - value(y), 1.0, -1.0, 0, 0, 0)
*(x::TVar, y::TVar) = binarynode(x, y, value(x) * value(y), value(y), value(x), 0, 1, 0)
function /(x::TVar, y::TVar)
    u, w = value(x), value(y)
    return binarynode(x, y, u / w, 1 / w, -u / w^2, 0, -1 / w^2, 2u / w^3)
end

# --- unary sign ---
+(x::TVar) = x
-(x::TVar) = unarynode(x, -value(x), -1.0, 0)

# --- TVar with a numeric constant, folded into a unary node ---
+(x::TVar, c::Real) = unarynode(x, value(x) + c, 1.0, 0)
+(c::Real, x::TVar) = x + c
-(x::TVar, c::Real) = unarynode(x, value(x) - c, 1.0, 0)
-(c::Real, x::TVar) = unarynode(x, c - value(x), -1.0, 0)
*(x::TVar, c::Real) = unarynode(x, value(x) * c, float(c), 0)
*(c::Real, x::TVar) = x * c
/(x::TVar, c::Real) = unarynode(x, value(x) / c, 1 / c, 0)
# A TVar scales an array of constants entrywise, as any scalar does. `*` on a scalar and an array
# is only defined for `Number`s, and a TVar is a tape node rather than a Number.
*(x::TVar, c::AbstractArray{<:Real}) = x .* c
*(c::AbstractArray{<:Real}, x::TVar) = c .* x
function /(c::Real, x::TVar)
    w = value(x)
    return unarynode(x, c / w, -c / w^2, 2c / w^3)
end

# --- numeric literals obtained from the element type ---
# `one(eltype(x))` and friends are the generic-code idiom for a typed literal. A TVar is a
# reference to a node on a specific tape, and these forms have no tape in hand (the type-level
# ones cannot, the instance ones need not), so they return plain `Real` constants, which is what
# the tape calls a constant: it folds into a node on first contact with a TVar through the
# arithmetic above. This departs from the usual `one(::Type{T})::T` contract, unavoidably for a
# tape-attached tracer.
Base.zero(::Type{TVar}) = 0.0
Base.one(::Type{TVar}) = 1.0
Base.oneunit(::Type{TVar}) = 1.0
Base.zero(::TVar) = 0.0
Base.one(::TVar) = 1.0
Base.oneunit(::TVar) = 1.0

# --- power with a constant exponent ---
function ^(x::TVar, p::Real)
    v = value(x)
    return unarynode(x, v^p, p * v^(p - 1), p * (p - 1) * v^(p - 2))
end
# Disambiguate from Base's `x^n` integer-literal path.
^(x::TVar, p::Integer) = invoke(^, Tuple{TVar,Real}, x, p)

# --- power with a variable exponent ---
# `u^w = exp(w*log(u))` is real-differentiable only for `u > 0`; a nonpositive base is a domain
# error rather than a NaN propagated into the cover.
function ^(x::TVar, y::TVar)
    u, w = value(x), value(y)
    u > 0 || throw(DomainError(u, "x^y with a variable exponent requires a positive base"))
    v, lu = u^w, log(u)
    return binarynode(x, y, v, w * u^(w - 1), v * lu,
                      w * (w - 1) * u^(w - 2), u^(w - 1) * (1 + w * lu), v * lu^2)
end
function ^(c::Real, x::TVar)
    c > 0 || throw(DomainError(c, "c^x with a variable exponent requires a positive base"))
    v, lc = float(c)^value(x), log(c)
    return unarynode(x, v, v * lc, v * lc^2)
end

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
function tan(x::TVar)
    t = tan(value(x))
    s2 = 1 + t^2                 # sec²
    return unarynode(x, t, s2, 2t * s2)
end
function atan(x::TVar)
    v = value(x)
    d = 1 / (1 + v^2)
    return unarynode(x, atan(v), d, -2v * d^2)
end
function tanh(x::TVar)
    t = tanh(value(x))
    s2 = 1 - t^2                 # sech²
    return unarynode(x, t, s2, -2t * s2)
end
function inv(x::TVar)
    v = value(x)
    return unarynode(x, inv(v), -1 / v^2, 2 / v^3)
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

# A sum of TVars records as one linear n-ary node (see `linearnode`) rather than a chain of
# n-1 binary additions: additions relay adjoints and absolute sweeps with unit weight and
# contribute no atoms, so every cover quantity agrees with the chained form to roundoff. The
# gain is tape length: one node instead of n-1.
function Base.sum(xs::AbstractArray{TVar})
    isempty(xs) && throw(ArgumentError("cannot record an empty sum"))
    return linearnode(ones(length(xs)), vec(xs))
end
Base.sum(f, xs::AbstractArray{TVar}) = sum(map(f, xs))
