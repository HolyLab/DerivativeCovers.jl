"""
    DerivativeCovers

Rank-1 covers of a program's Hessian at about twice the cost of a gradient.

A scalar program is recorded on a [`Tape`](@ref) by running it on [`TVar`](@ref) inputs;
[`cover_hessian`](@ref) then returns nonnegative vectors `h` with `abs(H[i,j]) <= h[i]*h[j]`,
without ever forming the Hessian `H`. The covers are deterministic, guaranteed, and exactly
covariant under rescaling of the inputs (`h -> inv(D)*h` when `x -> D*x`) and of the objective
(`h -> sqrt(c)*h` when `f -> c*f`). Input covariance holds as stated for the default
[`ValueSplit`](@ref) rule; under [`SensitivitySplit`](@ref) it holds when the input weights are
rescaled alongside (`w -> D*w`).

The submodule [`DerivativeCovers.Problems`](@ref) supplies parameterized families of programs, with
reference Hessians, for measuring how loose a cover is and what makes it so.
"""
module DerivativeCovers

using LinearAlgebra: Diagonal, Symmetric, dot, eigmin, issymmetric, norm, opnorm
using Random: AbstractRNG, MersenneTwister
using MatrixCovers: SupportComponents, ncomponents, rowcomponent, colcomponent
import MatrixCovers: support_components, gramcover, gramcover!

export Tape, TVar, tape_function, nnodes, value, isconstant
export CoverResult, cover_hessian, adjoints, sensitivities, hessian_bound,
       local_covers, local_cover_slots, balanced_split, abs_reverse, paired_reverse
export SplitRule, ValueSplit, SensitivitySplit, split_scales
export cyclic_partition, node_levels, level_partition, group_seeds, cover_hybrid
export VarianceCover, cover_variance, variance_seeds, dependency_sweep, dependency_severity
export hvp, scaled_spectral_radius, spectral_shrink
export cover_ratio, bound_ratio, rank1_ratio, tightness
export InflationPattern, UniformInflation, LognormalInflation, SpikeInflation, RampInflation
export inflation_shape, inflation_factors, inflate, anisotropy
export cover_norm, stopping_index, stopping_shift, regularized_step, regularization_bias,
       psd_shift, psd_excess, cover_cg, step_degradation
export softmax, layernorm
export NLSCover, cover_nls, nls_loss_tape, nls_partition, gn_ab
export JacobianCover, cover_jacobian, cover_jacobian_hybrid, jacobian_surrogate,
       jacobian_ratio, jacobian_representability, jacobian_minimax
export abs_forward, marginal_weights, balance_cover
export support_components, gramcover, gramcover!
export SupportComponents, ncomponents, rowcomponent, colcomponent
export jvp, vjp, equilibrate, scaled_singular_value, jacobian_shrink
export submapnode, submap, declare_submap, compose_jacobian_cover, compose_hessian_cover

# The tape-construction and cover-invariant entry points: the supported route for adding a new
# elemental. Public but not exported — they belong to the extension surface, not the top namespace.
# `public` is 1.11+ syntax and would be a parse error on the 1.10 given in `[compat]`, so the
# declaration goes through `eval` where the parser never sees the keyword.
@static if VERSION >= v"1.11"
    eval(Expr(:public, :pushnode!, :unarynode, :binarynode, :dotnode, :linearnode, :noderef,
              :constantnode, :atom_slots, :certifies_dead, :check_finite_partials))
end

include("tape.jl")
include("overloads.jl")
include("composites.jl")
include("covers.jl")
include("merges.jl")
include("variance.jl")
include("spectral.jl")
include("metrics.jl")
include("inflation.jl")
include("consumers.jl")
include("nls.jl")
include("support.jl")
include("jacobian.jl")
include("gram.jl")
include("equilibrate.jl")
include("compose.jl")
include("problems.jl")

end
