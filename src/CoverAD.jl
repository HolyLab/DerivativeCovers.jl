"""
    CoverAD

Rank-1 covers of a program's Hessian at about twice the cost of a gradient.

A scalar program is recorded on a [`Tape`](@ref) by running it on [`TVar`](@ref) inputs;
[`cover_hessian`](@ref) then returns nonnegative vectors `h` with `abs(H[i,j]) <= h[i]*h[j]`,
without ever forming the Hessian `H`. The covers are deterministic, guaranteed, and exactly
covariant under rescaling of the inputs (`h -> inv(D)*h` when `x -> D*x`) and of the objective
(`h -> sqrt(c)*h` when `f -> c*f`).
"""
module CoverAD

export Tape, TVar, tape_function, nnodes, value
export CoverResult, cover_hessian, adjoints, sensitivities, hessian_bound,
       local_covers, balanced_split, abs_reverse
export cover_ratio, bound_ratio, rank1_ratio, tightness

include("tape.jl")
include("overloads.jl")
include("covers.jl")
include("metrics.jl")

end
