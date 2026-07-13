"""
    CoverAD

Rank-1 covers of a program's Hessian at about twice the cost of a gradient.

A scalar program is recorded on a [`Tape`](@ref) by running it on [`TVar`](@ref) inputs;
[`cover_hessian`](@ref) then returns nonnegative vectors `h` with `abs(H[i,j]) <= h[i]*h[j]`,
without ever forming the Hessian `H`. The covers are deterministic, guaranteed, and exactly
covariant under rescaling of the inputs (`h -> inv(D)*h` when `x -> D*x`) and of the objective
(`h -> sqrt(c)*h` when `f -> c*f`).

The submodule [`CoverAD.Problems`](@ref) supplies parameterized families of programs, with
reference Hessians, for measuring how loose a cover is and what makes it so.
"""
module CoverAD

using LinearAlgebra: norm, opnorm
using Random: AbstractRNG, MersenneTwister, randn

export Tape, TVar, tape_function, nnodes, value
export CoverResult, cover_hessian, adjoints, sensitivities, hessian_bound,
       local_covers, balanced_split, abs_reverse
export cyclic_partition, node_levels, level_partition, group_seeds, cover_hybrid
export VarianceCover, cover_variance, variance_seeds, dependency_sweep, dependency_severity
export hvp, scaled_spectral_radius, spectral_shrink
export cover_ratio, bound_ratio, rank1_ratio, tightness

include("tape.jl")
include("overloads.jl")
include("covers.jl")
include("merges.jl")
include("variance.jl")
include("spectral.jl")
include("metrics.jl")
include("problems.jl")

end
