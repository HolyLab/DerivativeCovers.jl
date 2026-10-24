"""
    DerivativeCoversIntervalArithmeticExt

Covers over a box. Recording a program on `Interval` inputs, `tape_function(f, interval.(lo, hi))`,
gives a `Tape{Interval{T}}` whose values and partials enclose those at every point of the box.
`magnitude` of an interval is its largest absolute value (`mag`), and `boundtype` is
`DerivativeCovers.Upward`, so the sweeps round every sum, product and square root of magnitudes
upward. The covers they compute from such a tape are plain floating-point vectors that hold at
every point of the box, exactly rather than up to rounding, and the gradient is an interval vector
enclosing the gradient over the box.

Control flow that the box does not decide raises IntervalArithmetic's
`InconclusiveBooleanOperation` while recording: a comparison the program makes on a variable's
value, such as `value(x) > 0` for an `x` whose interval contains zero, or one an elemental makes,
such as the positive-base check of `x^y`. (Comparisons of the variables themselves are not
defined.) `abs` of an interval containing zero is accepted only as the base of a constant power
`p >= 2` (see [`DerivativeCovers.decided_sign`](@ref)), and `atan(y, x)` refuses a box that touches
its branch cut, `y = 0` with `x <= 0`. An elemental evaluated outside its domain
on part of the box, such as `log` of an interval containing zero, does not raise in
IntervalArithmetic; it returns an interval decorated `trv`. `check_values` looks for that decoration on every recorded value, first partial and unary
or binary second partial, and throws a `DomainError` naming the first node that carries it.
`forward_cover`, which keeps no tape, makes the same check at each elemental as it reaches it.

The gradient entries usually report `isguaranteed(x) == false`. That flag records that a bare
floating-point number entered the computation, which happens wherever an elemental's constant
partial (the `1` of `x + y`) or a numeric literal of the program is converted to an interval. It
does not affect the covers, which are floating-point bounds computed from the interval enclosures.
"""
module DerivativeCoversIntervalArithmeticExt

using DerivativeCovers: DerivativeCovers, CoverSite, Tape, TapeSite, UnaryCurvature,
    BinaryCurvature, check_values, nnodes, site_label, site_values
using IntervalArithmetic: Interval, decoration, inf, mag, sup, trv

DerivativeCovers.magnitude(x::Interval) = mag(x)

# Covers over a box hold at every point of it, so the sweeps round their bounds upward.
DerivativeCovers.boundtype(::Type{Interval{F}}) where {F <: AbstractFloat} = DerivativeCovers.Upward{F}

DerivativeCovers.decided_sign(x::Interval) = inf(x) > 0 ? 1 : sup(x) < 0 ? -1 : 0

# The gap between two intervals, zero when they overlap.
DerivativeCovers.separation(x::Interval, y::Interval) =
    max(zero(inf(x)), inf(x) - sup(y), inf(y) - sup(x))

# `trv` and `ill` mark a result that is undefined on part of its inputs.
undefined(x::Interval) = decoration(x) <= trv
# A second partial recorded as a plain number is a constant of the elemental.
undefined(x) = false

function DerivativeCovers.check_values(site::CoverSite, val::Interval, ds, cs)
    undefined(val) && throw(
        DomainError(
            val,
            "the program leaves the domain of an elemental somewhere on the box: the value of " *
                "$(site_label(site)) is decorated $(decoration(val)) ($(site_values(site)))"
        )
    )
    for (i, d) in enumerate(ds)
        undefined(d) && throw(
            DomainError(
                d,
                "the program is not differentiable everywhere on the box: the first partial " *
                    "of $(site_label(site)) with respect to operand $i is decorated " *
                    "$(decoration(d)) ($(site_values(site)))"
            )
        )
    end
    for c in cs
        undefined(c) && throw(
            DomainError(
                c,
                "the program is not twice differentiable everywhere on the box: a second " *
                    "partial of $(site_label(site)) is decorated $(decoration(c)) " *
                    "($(site_values(site)))"
            )
        )
    end
    return nothing
end

function DerivativeCovers.check_values(t::Tape{<:Interval})
    for k in 1:nnodes(t)
        C = t.Cs[k]
        cs = C isa UnaryCurvature ? (C.c,) :
            C isa BinaryCurvature ? (C.c11, C.c12, C.c22) : ()
        check_values(TapeSite(t, k), t.vals[k], t.ds[k], cs)
    end
    return nothing
end

end
