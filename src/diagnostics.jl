"""
    DerivativeCovers.Diagnostics

Instruments for measuring what a loose cover costs.

Nothing here takes a [`Tape`](@ref DerivativeCovers.Tape) or any other package type: these
functions act on bare covers `h`, gradients `g`, and matrices `A`, so a cover from any source can
be fed to them.

The two halves compose. An [`InflationPattern`](@ref) distorts a reference cover by a controlled
amount and shape — [`inflate`](@ref) only ever weakens a bound, so the distortion is still a valid
cover of the same matrix. The consumers ([`cover_norm`](@ref), [`regularized_step`](@ref),
[`psd_shift`](@ref), [`cover_cg`](@ref)) then use a cover as a metric on the variables, and each
has a paired form ([`stopping_shift`](@ref), [`regularization_bias`](@ref), [`psd_excess`](@ref),
[`step_degradation`](@ref)) reporting what the distortion cost. Since every consumer is
scale-invariant, the degradation is attributable to looseness alone.
"""
module Diagnostics

using LinearAlgebra: Diagonal, Symmetric, dot, eigmin, issymmetric, norm
using Random: AbstractRNG, MersenneTwister

export InflationPattern, UniformInflation, LognormalInflation, SpikeInflation, RampInflation
export inflation_shape, inflation_factors, inflate, anisotropy
export cover_norm, stopping_index, stopping_shift, regularized_step, regularization_bias,
       psd_shift, psd_excess, cover_cg, step_degradation

include("inflation.jl")
include("consumers.jl")

end
