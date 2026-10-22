"""
    DerivativeCovers

Compute certified rank-1 bounds on Hessians and Jacobians without forming either matrix.

A Hessian cover is a nonnegative vector `h` satisfying
`abs(H[i,j]) <= h[i] * h[j]` for every entry of `H`. [`cover_hessian`](@ref) records a scalar
function on a [`Tape`](@ref) and computes such a cover, together with the gradient, in `O(L)`
work for a tape of length `L`. The result is deterministic and certified.

A cover is covariant under rescaling: if `x -> D*x`, then `h -> inv(D)*h`; if `f -> c*f`, then
`h -> sqrt(c)*h`. That holds exactly on any program whose blocks all split on parent values
carrying a scale of their own. Where one does not, the split falls back on declared input scales
and stays covariant when those are rescaled alongside, `scales -> D*scales`;
[`guards_any_block`](@ref) says whether a given program is in that case.

`using DerivativeCovers` imports the main entry points and result types. Public utilities such as
`cover_ratio`, `tightness`, and `equilibrate` are not exported; call them with the
`DerivativeCovers.` prefix. The [`DerivativeCovers.Diagnostics`](@ref) submodule provides tools
for studying how cover looseness affects downstream algorithms.
"""
module DerivativeCovers

using LinearAlgebra: Diagonal, diag, dot, norm, opnorm
using Random: AbstractRNG, Xoshiro
using SparseArrays: SparseArrays, AbstractSparseVector
using StaticArrays: StaticMatrix, StaticVector, similar_type
using MatrixCovers: SupportComponents, initialize_cover, iscover, symcover
import MatrixCovers: support_components, gramcover, gramcover!

# Main entry points and result types.
export Tape, TVar, tape_function
export FVar, forward_cover, ForwardCover
export cover_hessian, cover_jacobian, cover_nls
export CoverOnly, WithDiagnostics
export HessianCover, HessianDiagnostics
export JacobianCover, JacobianDiagnostics
export NLSCover, NLSDiagnostics
export hvp, hvp!, jvp, jvp!, vjp, vjp!

# Public but unexported extension points and cover utilities. Julia 1.10 cannot parse `public`,
# so construct the declaration dynamically on Julia 1.11 and later.
@static if VERSION >= v"1.11"
    # Tape construction and the invariants a new elemental must satisfy.
    eval(
        Expr(
            :public, :pushnode!, :unarynode, :binarynode, :dotnode, :linearnode, :noderef,
            :constantnode, :atom_slots, :check_finite_partials, :submapnode,
            :split_scales, :split_block, :split_ratio, :guards_any_block, :sensitivity_scales,
            :SplitScale, :dead_slot, :value, :isconstant, :nnodes, :natoms, :CoverVar,
            :constantvar, :forward_scale, :magnitude, :magtype, :boundtype, :Upward, :separation,
            :decided_sign, :check_values, :adjoints
        )
    )
    # The second-partial forms a node may carry, and the operations the sweeps perform on them.
    eval(
        Expr(
            :public, :Curvature, :NoCurvature, :UnaryCurvature, :BinaryCurvature,
            :DenseCurvature, :PairedCurvature, :DeclaredCurvature, :cover_slots, :cover_slots!,
            :bound_contribution!, :curvature_apply!, :has_exact_partials,
            :CoverSite, :TapeSite, :site_label, :site_values
        )
    )
    # Ratios, bounds, and rescalings defined on a cover from any source.
    eval(
        Expr(
            :public, :cover_ratio, :bound_ratio, :tightness, :jacobian_ratio,
            :jacobian_representability, :jacobian_minimax,
            :scaled_spectral_radius, :scaled_singular_value, :equilibrate
        )
    )
    # Covers that merge in groups, interpolating between the sum and quadrature merges.
    eval(Expr(:public, :cover_hybrid, :cover_jacobian_hybrid))
    # Adapters between the result types and the functions that take bare vectors.
    eval(
        Expr(
            :public, :compose_hessian_cover, :compose_jacobian_cover, :gn_ab, :nls_loss_tape,
            :declare_submap, :submap
        )
    )
    # A scale, not a cover, of a tape's Gram matrix.
    eval(Expr(:public, :gramvariance, :gramvariance!))
end

include("rounding.jl")
include("curvature.jl")
include("tape.jl")
include("overloads.jl")
include("covers.jl")
include("forward.jl")
include("merges.jl")
include("variance.jl")
include("maxpath.jl")
include("spectral.jl")
include("metrics.jl")
include("nls.jl")
# `covariant.jl` defines `LiveInfo`, which appears in `jacobian.jl` method signatures.
include("support.jl")
include("covariant.jl")
include("jacobian.jl")
include("gram.jl")
include("equilibrate.jl")
include("compose.jl")
include("show.jl")
include("diagnostics.jl")

end
