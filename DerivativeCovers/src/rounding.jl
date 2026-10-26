# Upward-rounded arithmetic on magnitudes, for covers that must hold over a set of points.

"""
    Upward{F}(x)

A nonnegative magnitude of floating-point type `F` whose arithmetic rounds upward: `+`, `*`, `/`,
`sqrt`, `hypot` and integer powers return a value no smaller than the exact result. Each elementary
operation rounds to nearest and then steps to `nextfloat`, except where the result is exact (an
operand that is zero, or a factor or divisor that is one); `x <= nextfloat(fl(x))` holds for any
rounding to nearest, so this needs no control over the hardware rounding mode. `hypot` and the
powers compose these operations, so they may step more than once.

A cover is a bound built from magnitudes by sums, products and square roots, each increasing in
its operands, so rounding every step upward makes the computed cover at least the exact one.
Subtraction is not increasing in its second operand and throws, and transcendental functions are
not defined; a computation that reaches one fails rather than silently losing the bound. Division
rounds upward too, and is an upper bound only where its divisor is fixed rather than computed,
such as a chosen weight.

Mixing an `Upward` with a plain real promotes to `Upward`, so an operation rounds upward whenever
either operand is one. Converting an `Upward{F}` to `F` is exact.

[`boundtype`](@ref) selects this type for the sweeps over a tape whose values stand for sets.
"""
struct Upward{F <: AbstractFloat} <: Real
    x::F

    Upward{F}(x::F) where {F <: AbstractFloat} = new{F}(x)
end

# The smallest `F` at least `x`.
function Upward{F}(x::Real) where {F <: AbstractFloat}
    y = F(x)
    return Upward{F}(y < x ? nextfloat(y) : y)
end
Upward{F}(x::Upward) where {F <: AbstractFloat} = Upward{F}(x.x)
Upward(x::AbstractFloat) = Upward{typeof(x)}(x)

Base.convert(::Type{Upward{F}}, x::Real) where {F} = Upward{F}(x)
Base.convert(::Type{Upward{F}}, x::Upward{F}) where {F} = x
Base.convert(::Type{F}, x::Upward{F}) where {F <: AbstractFloat} = x.x
Base.promote_rule(::Type{Upward{F}}, ::Type{S}) where {F, S <: Real} =
    Upward{float(promote_type(F, S))}
Base.promote_rule(::Type{Upward{F}}, ::Type{Upward{G}}) where {F, G} = Upward{promote_type(F, G)}

Base.float(x::Upward) = x
Base.zero(::Type{Upward{F}}) where {F} = Upward{F}(zero(F))
Base.one(::Type{Upward{F}}) where {F} = Upward{F}(one(F))
Base.abs(x::Upward) = Upward(abs(x.x))
Base.abs2(x::Upward) = x * x
for f in (:iszero, :isone, :isfinite, :isinf, :isnan, :signbit)
    @eval Base.$f(x::Upward) = $f(x.x)
end
Base.:(==)(a::Upward, b::Upward) = a.x == b.x
Base.:<(a::Upward, b::Upward) = a.x < b.x
Base.:<=(a::Upward, b::Upward) = a.x <= b.x
Base.isless(a::Upward, b::Upward) = isless(a.x, b.x)
Base.show(io::IO, x::Upward) = show(io, x.x)

Base.:+(a::Upward{F}, b::Upward{F}) where {F} =
    Upward{F}(iszero(a.x) || iszero(b.x) ? a.x + b.x : nextfloat(a.x + b.x))
Base.:*(a::Upward{F}, b::Upward{F}) where {F} =
    Upward{F}(_exactfactor(a.x) || _exactfactor(b.x) ? a.x * b.x : nextfloat(a.x * b.x))
Base.:/(a::Upward{F}, b::Upward{F}) where {F} =
    Upward{F}(iszero(a.x) || isone(b.x) ? a.x / b.x : nextfloat(a.x / b.x))
# Whether the result is exact is decided from the operand, not the result:
# `sqrt(nextfloat(one(F)))` rounds to one, so a result of one does not imply an exact result.
function Base.sqrt(a::Upward{F}) where {F}
    r = sqrt(a.x)
    return Upward{F}(_exactfactor(a.x) ? r : nextfloat(r))
end
# The 2-norm of two bounds, through the scaled form `m * sqrt(1 + (s/m)^2)` with `s <= m`, so that
# it neither overflows nor underflows where the result is representable. Each operation is
# increasing in its operands and rounds upward, so the result is an upper bound. Upward rounding
# keeps the square of a nonzero ratio above zero, so the result steps above `m` even when that
# square underflows.
function Base.hypot(a::Upward{F}, b::Upward{F}) where {F}
    s, m = minmax(a.x, b.x)
    iszero(s) && return Upward{F}(m)
    r = Upward{F}(s) / Upward{F}(m)
    return Upward{F}(m) * sqrt(one(Upward{F}) + r * r)
end

# Repeated multiplication, each product rounded upward. A negative power is decreasing in `a`.
function Base.:^(a::Upward, n::Integer)
    n >= 0 || throw(DomainError(n, "a negative power does not preserve an upper bound"))
    r = one(a)
    for _ in 1:n
        r *= a
    end
    return r
end

_exactfactor(x) = iszero(x) || isone(x)

Base.:-(::Upward, ::Upward) = throw(
    ArgumentError(
        "subtraction is decreasing in its second operand, so an upward-rounded difference of " *
            "upper bounds is not an upper bound; convert to a plain float to subtract"
    )
)

"""
    boundtype(T)

The type the cover sweeps over a [`Tape{T}`](@ref Tape) accumulate their bounds in. For a type
whose values are points this is [`magtype`](@ref)`(T)`, and the covers hold up to the rounding of
the sweeps. A type whose values stand for sets, such as an interval, opts in to rigorous covers by
returning an [`Upward`](@ref) type, so that every sum, product and square root the sweeps take
rounds upward and the covers hold for every point of the set. The covers are returned in
`magtype(T)` either way.
"""
boundtype(::Type{T}) where {T} = magtype(T)

# `x` in the type a sweep over a `Tape{T}` accumulates bounds in, where that rounds upward; `x`
# itself otherwise.
tobound(::Type{T}, x) where {T} = _tobound(boundtype(T), x)
_tobound(::Type{U}, x) where {U <: Upward} = convert(U, x)
_tobound(::Type, x) = x

# An array of bounds from a sweep over a `Tape{T}`, in `magtype(T)`. The conversion is exact, so
# its entries are the bounds the sweep computed.
tomag(::Type{T}, A::AbstractArray) where {T} = convert(AbstractArray{magtype(T)}, A)

# The floating-point value of a bound, for the choices that select among valid covers and so need
# no rounding direction.
plainvalue(x::Upward) = x.x
plainvalue(x) = x

# Whether the sweeps over a `Tape{T}` round upward.
roundsup(::Type{T}) where {T} = boundtype(T) <: Upward
