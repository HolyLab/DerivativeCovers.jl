"""
    DerivativeCoversIntervalArithmeticExt

Covers over a box. Recording a program on `Interval` inputs, `tape_function(f, interval.(lo, hi))`,
gives a `Tape{Interval{T}}` whose values and partials enclose those at every point of the box.
`magnitude` of an interval is its largest absolute value (`mag`), so the covers the sweeps compute
from such a tape are plain floating-point vectors that hold at every point of the box, and the
gradient is an interval vector enclosing the gradient over the box.

Control flow in the program that the box does not decide, such as `x > 0` for an `x` whose
interval contains zero, raises IntervalArithmetic's `InconclusiveBooleanOperation` while
recording. An elemental evaluated outside its domain on part of the box, such as `log` of an
interval containing zero, does not raise in IntervalArithmetic; it returns an interval decorated
`trv`. `check_values` looks for that decoration on every recorded value, first partial and unary
or binary second partial, and throws a `DomainError` naming the first node that carries it.

The gradient entries usually report `isguaranteed(x) == false`. That flag records that a bare
floating-point number entered the computation, which happens wherever an elemental's constant
partial (the `1` of `x + y`) or a numeric literal of the program is converted to an interval. It
does not affect the covers, which are floating-point bounds computed from the interval enclosures.
"""
module DerivativeCoversIntervalArithmeticExt

using DerivativeCovers: DerivativeCovers, Tape, UnaryCurvature, BinaryCurvature, nnodes
using IntervalArithmetic: Interval, decoration, inf, mag, sup, trv

DerivativeCovers.magnitude(x::Interval) = mag(x)

# The gap between two intervals, zero when they overlap.
DerivativeCovers.separation(x::Interval, y::Interval) =
    max(zero(inf(x)), inf(x) - sup(y), inf(y) - sup(x))

# `trv` and `ill` mark a result that is undefined on part of its inputs.
undefined(x::Interval) = decoration(x) <= trv

function DerivativeCovers.check_values(t::Tape{<:Interval})
    for k in 1:nnodes(t)
        v = t.vals[k]
        undefined(v) && throw(
            DomainError(
                v,
                "the program leaves the domain of an elemental somewhere on the box: the " *
                    "value of node $k is decorated $(decoration(v))"
            )
        )
        for (i, d) in enumerate(t.ds[k])
            undefined(d) && throw(
                DomainError(
                    d,
                    "the program is not differentiable everywhere on the box: the first " *
                        "partial of node $k with respect to node $(t.parents[k][i]) is " *
                        "decorated $(decoration(d))"
                )
            )
        end
        C = t.Cs[k]
        cs = C isa UnaryCurvature ? (C.c,) :
            C isa BinaryCurvature ? (C.c11, C.c12, C.c22) : ()
        for c in cs
            c isa Interval && undefined(c) && throw(
                DomainError(
                    c,
                    "the program is not twice differentiable everywhere on the box: a second " *
                        "partial of node $k is decorated $(decoration(c))"
                )
            )
        end
    end
    return nothing
end

end
