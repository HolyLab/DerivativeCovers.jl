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

`using DerivativeCovers` brings in the entry points and the types they return, and nothing else.
The quantities defined on a cover (`cover_ratio`, `tightness`, `equilibrate`, …) and the surface
for adding a new elemental (`pushnode!`, `unarynode`, …) are public but unexported, so they are
reached by qualification. Tape methods for `MatrixCovers.support_components`,
`MatrixCovers.gramcover`, and `MatrixCovers.gramcover!` are defined here, but those functions
belong to MatrixCovers: calling them unqualified takes `using MatrixCovers` as well.

Two submodules serve the study of cover quality rather than its use:
[`DerivativeCovers.Problems`](@ref) supplies parameterized families of programs with reference
Hessians, and [`DerivativeCovers.Diagnostics`](@ref) supplies the instruments that measure what a
loose cover costs a consumer of it.
"""
module DerivativeCovers

using LinearAlgebra: Diagonal, dot, norm, opnorm
using Random: AbstractRNG, MersenneTwister
using MatrixCovers: SupportComponents
import MatrixCovers: support_components, gramcover, gramcover!

# The entry points a caller needs after `using DerivativeCovers`: the tape, the four cover
# constructors and the two that trade cost against tightness, the types they return, the split
# rules that parameterize them, and the matrix-free products.
export Tape, TVar, tape_function
export cover_hessian, cover_jacobian, cover_nls, cover_variance
export cover_hybrid, cover_jacobian_hybrid
export CoverResult, JacobianCover, NLSCover, VarianceCover
export SplitRule, ValueSplit, SensitivitySplit
export hvp, jvp, vjp

# Addressable by qualification, and covered by the same compatibility promise as the exports,
# but kept out of the top namespace. Two kinds of name qualify: the contracts outside code
# implements or records against, and the quantities that are defined independently of how any
# sweep computes them. Neither can be disturbed by a reimplementation of the sweeps themselves,
# which is why they carry the promise and the sweep machinery does not.
#
# `public` is 1.11+ syntax and would be a parse error on the 1.10 given in `[compat]`, so the
# declaration goes through `eval` where the parser never sees the keyword.
@static if VERSION >= v"1.11"
    # Tape construction and the invariants a new elemental must satisfy.
    eval(Expr(:public, :pushnode!, :unarynode, :binarynode, :dotnode, :linearnode, :noderef,
              :constantnode, :atom_slots, :certifies_dead, :check_finite_partials, :submapnode,
              :split_scales, :value, :isconstant, :nnodes))
    # Ratios, bounds, and rescalings defined on a cover from any source.
    eval(Expr(:public, :cover_ratio, :bound_ratio, :rank1_ratio, :tightness, :jacobian_ratio,
              :jacobian_representability, :jacobian_minimax, :dependency_severity,
              :scaled_spectral_radius, :scaled_singular_value, :spectral_shrink, :jacobian_shrink,
              :equilibrate))
    # Adapters between the result types and the functions that take bare vectors.
    eval(Expr(:public, :compose_hessian_cover, :compose_jacobian_cover, :gn_ab, :nls_loss_tape,
              :declare_submap, :submap))
end

include("tape.jl")
include("overloads.jl")
include("covers.jl")
include("merges.jl")
include("variance.jl")
include("spectral.jl")
include("metrics.jl")
include("nls.jl")
include("support.jl")
include("jacobian.jl")
include("gram.jl")
include("equilibrate.jl")
include("compose.jl")
include("diagnostics.jl")
include("problems.jl")

end
