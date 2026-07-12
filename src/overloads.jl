# Elementary operations recorded on the tape. Each records its value, its first partials, and
# its second partials; a program built from anything outside this set hits a MethodError.

import Base: +, -, *, /, ^, exp, log, sin, cos, tanh, sqrt, abs2

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
