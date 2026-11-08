# A floating-point number with a separate `Int` exponent. Self-contained: nothing here depends on
# the rest of the package.

"""
    ExpScaled{F}(m, e=0)
    ExpScaled(x::AbstractFloat)

The real number `m * 2^e`, stored as a significand of type `F <: AbstractFloat` and an `Int`
exponent. It has the precision of `F` and an exponent range limited only by `Int`, so products and
sums of positive values never underflow to zero or overflow to `Inf`.

Wherever the exact result of an operation is a normal number of `F`, `+`, `-`, `*`, `/` and `sqrt`
round to the same value `F` would give: the significand is computed in `F`, and scaling by a power
of two does not change how a normal number rounds. Results below `F`'s normal range or above its
largest value keep full precision instead. Conversion back to `F` rounds once, saturating to
zero or `Inf` outside `F`'s range.

The significand is held in `frexp` form: `0.5 <= abs(m) < 1` for a finite nonzero value, and the
exponent is `0` for zero and for non-finite values, which carry their `F` value in `m`. An
exponent that would leave the range of `Int` is an `OverflowError`.

# Examples

```jldoctest
julia> x = DerivativeCovers.ExpScaled(1.0e-300)^2
DerivativeCovers.ExpScaled{Float64}(0.8969771056830114, -1993)

julia> Float64(x), Float64(sqrt(x))
(0.0, 1.0e-300)
```
"""
struct ExpScaled{F <: AbstractFloat} <: Real
    sig::F
    exp::Int

    function ExpScaled{F}(m, e) where {F <: AbstractFloat}
        m = convert(F, m)::F
        (isfinite(m) && !iszero(m)) || return new{F}(m, 0)
        mm, ee = frexp(m)
        return new{F}(mm, Base.Checked.checked_add(convert(Int, e)::Int, ee))
    end
end

ExpScaled{F}(x::Real) where {F <: AbstractFloat} = ExpScaled{F}(x, 0)
ExpScaled{F}(x::ExpScaled) where {F <: AbstractFloat} = ExpScaled{F}(x.sig, x.exp)
ExpScaled(m::AbstractFloat, e = 0) = ExpScaled{typeof(m)}(m, e)
ExpScaled(x::ExpScaled) = x

(::Type{G})(x::ExpScaled) where {G <: AbstractFloat} = convert(G, ldexp(x.sig, x.exp))

Base.promote_rule(::Type{ExpScaled{F}}, ::Type{ExpScaled{G}}) where {F, G} =
    ExpScaled{promote_type(F, G)}
Base.promote_rule(::Type{ExpScaled{F}}, ::Type{T}) where {F, T <: Real} =
    ExpScaled{promote_type(F, T)}
# Base claims every `Real` for `BigFloat`; without this the two rules disagree and `promote_type`
# recurses.
Base.promote_rule(::Type{BigFloat}, ::Type{ExpScaled{F}}) where {F} =
    ExpScaled{promote_type(F, BigFloat)}

Base.float(::Type{E}) where {E <: ExpScaled} = E
Base.float(x::ExpScaled) = x
Base.zero(::Type{ExpScaled{F}}) where {F} = ExpScaled{F}(zero(F))
Base.one(::Type{ExpScaled{F}}) where {F} = ExpScaled{F}(one(F))

Base.frexp(x::ExpScaled) = (x.sig, x.exp)
Base.ldexp(x::ExpScaled{F}, n::Integer) where {F} =
    ExpScaled{F}(x.sig, Base.Checked.checked_add(x.exp, convert(Int, n)::Int))

Base.iszero(x::ExpScaled) = iszero(x.sig)
Base.isfinite(x::ExpScaled) = isfinite(x.sig)
Base.isinf(x::ExpScaled) = isinf(x.sig)
Base.isnan(x::ExpScaled) = isnan(x.sig)
Base.signbit(x::ExpScaled) = signbit(x.sig)
Base.abs(x::ExpScaled{F}) where {F} = ExpScaled{F}(abs(x.sig), x.exp)
Base.:-(x::ExpScaled{F}) where {F} = ExpScaled{F}(-x.sig, x.exp)

# Normalization makes the stored pair unique for each value, except for the sign of zero.
Base.:(==)(a::ExpScaled, b::ExpScaled) = a.sig == b.sig && a.exp == b.exp

function Base.:<(a::ExpScaled, b::ExpScaled)
    # A non-finite significand compares correctly against any finite one, whose magnitude is < 1.
    (isfinite(a.sig) && isfinite(b.sig)) || return a.sig < b.sig
    sa, sb = sign(a.sig), sign(b.sig)
    sa == sb || return sa < sb
    a.exp == b.exp && return a.sig < b.sig
    return sa > 0 ? a.exp < b.exp : a.exp > b.exp
end

# `x == num * 2^pow / den`, which `hash(::Real)` reads, so equal values hash alike whatever their
# type.
function Base.decompose(x::ExpScaled)
    num, pow, den = Base.decompose(x.sig)
    (isfinite(x.sig) && !iszero(x.sig)) || return num, pow, den
    return num, pow + x.exp, den
end

Base.:*(a::ExpScaled{F}, b::ExpScaled{F}) where {F} =
    ExpScaled{F}(a.sig * b.sig, Base.Checked.checked_add(a.exp, b.exp))
Base.:/(a::ExpScaled{F}, b::ExpScaled{F}) where {F} =
    ExpScaled{F}(a.sig / b.sig, Base.Checked.checked_sub(a.exp, b.exp))

function Base.:+(a::ExpScaled{F}, b::ExpScaled{F}) where {F}
    # A non-finite operand decides the sum whatever the finite one's exponent.
    (isfinite(a.sig) && isfinite(b.sig)) || return ExpScaled{F}(a.sig + b.sig)
    iszero(b.sig) && return iszero(a.sig) ? ExpScaled{F}(a.sig + b.sig) : a
    iszero(a.sig) && return b
    a.exp < b.exp && ((a, b) = (b, a))
    # Exact unless the shift takes `b` below `F`'s normal range, and then `b` is far under half an
    # ulp of `a`, which rounds to the same sum.
    return ExpScaled{F}(a.sig + ldexp(b.sig, Base.Checked.checked_sub(b.exp, a.exp)), a.exp)
end
Base.:-(a::ExpScaled{F}, b::ExpScaled{F}) where {F} = a + (-b)

function Base.sqrt(x::ExpScaled{F}) where {F}
    m, e = x.sig, x.exp
    if isodd(e)
        m *= 2
        e -= 1
    end
    return ExpScaled{F}(sqrt(m), e >> 1)
end

function Base.show(io::IO, x::ExpScaled{F}) where {F}
    print(io, ExpScaled{F}, "(")
    show(io, x.sig)
    return print(io, ", ", x.exp, ")")
end
