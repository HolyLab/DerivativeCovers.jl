"""
    DerivativeCovers

Compute certified rank-1 bounds on Hessians and Jacobians without forming either matrix.

A Hessian cover is a nonnegative vector `h` satisfying
`abs(H[i,j]) <= h[i] * h[j]` for every entry of `H`. [`cover_hessian`](@ref) records a scalar
function on a [`Tape`](@ref) and computes such a cover, together with the gradient, in `O(L)`
work for a tape of length `L`. The result is deterministic and certified.

The default [`ValueSplit`](@ref) rule is covariant under rescaling: if `x -> D*x`, then
`h -> inv(D)*h`; if `f -> c*f`, then `h -> sqrt(c)*h`.
[`SensitivitySplit`](@ref) has the same input covariance when its weights are rescaled as
`w -> D*w`.

`using DerivativeCovers` imports the main entry points and result types. Public utilities such as
`cover_ratio`, `tightness`, and `equilibrate` are not exported; call them with the
`DerivativeCovers.` prefix. The [`DerivativeCovers.Diagnostics`](@ref) submodule provides tools
for studying how cover looseness affects downstream algorithms.
"""
module DerivativeCovers

using LinearAlgebra: Diagonal, diag, dot, norm, opnorm
using Random: AbstractRNG, Xoshiro
using MatrixCovers: SupportComponents
import MatrixCovers: support_components, gramcover, gramcover!

# The entry points a caller needs after `using DerivativeCovers`: the tape, the four cover
# constructors and the two that trade cost against tightness, the modes that select how much
# `cover_hessian` returns, the types they return, the split rules that parameterize them, and the
# matrix-free products.
export Tape, TVar, tape_function
export cover_hessian, cover_jacobian, cover_nls, cover_variance
export cover_hybrid, cover_jacobian_hybrid
export CoverOnly, WithDiagnostics
export HessianCover, HessianDiagnostics, JacobianCover, NLSCover, VarianceCover
export SplitRule, ValueSplit, SensitivitySplit
export hvp, hvp!, jvp, jvp!, vjp, vjp!

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
    eval(
        Expr(
            :public, :pushnode!, :unarynode, :binarynode, :dotnode, :linearnode, :noderef,
            :constantnode, :atom_slots, :certifies_dead, :check_finite_partials, :submapnode,
            :split_scales, :value, :isconstant, :nnodes
        )
    )
    # Ratios, bounds, and rescalings defined on a cover from any source.
    eval(
        Expr(
            :public, :cover_ratio, :bound_ratio, :rank1_ratio, :tightness, :jacobian_ratio,
            :jacobian_representability, :jacobian_minimax, :dependency_severity,
            :scaled_spectral_radius, :scaled_singular_value, :spectral_shrink, :jacobian_shrink,
            :equilibrate
        )
    )
    # Adapters between the result types and the functions that take bare vectors.
    eval(
        Expr(
            :public, :compose_hessian_cover, :compose_jacobian_cover, :gn_ab, :nls_loss_tape,
            :declare_submap, :submap
        )
    )
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
include("show.jl")
include("diagnostics.jl")

end
