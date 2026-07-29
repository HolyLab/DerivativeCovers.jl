# Gram-matrix covers derived from an asymmetric Jacobian cover.
#
# MatrixCovers implements the construction. These methods derive support components from a tape
# in O(L), without forming Ũ, and delegate the remaining work.

"""
    gramcover(a, b, t::Tape, outidxs) -> s
    gramcover(a, b, t::Tape, outidxs, w::AbstractVector) -> s
    gramcover(a, b, t::Tape, outidxs, W::AbstractMatrix) -> s

Symmetric cover of the Gram matrix `G = J'*W*J` of the Jacobian `J` of the tape
outputs `outidxs`, read off an asymmetric cover `(a, b)` of `J` without forming
either `J` or `G`: a nonnegative vector `s`, one entry per input, with
`abs(G[j,k]) <= s[j]*s[k]` for every pair. `a` is indexed by output position
(`1:length(outidxs)`) and `b` by input (`1:t.ninputs`); `W` defaults to the
identity (`G = J'*J`, the Gauss–Newton matrix) and a vector `w` means
`W = Diagonal(w)`.

This is the tape counterpart of `MatrixCovers.gramcover(a, b, Ũ[, w or W])`: it
takes the support components from [`support_components`](@ref)`(t, outidxs)`, whose
`O(L)` tape sweep never materializes the surrogate `Ũ`, and hands them to
`MatrixCovers.gramcover`. See that function for the bound, its gauge invariance,
and the roundoff guarantee. `(a, b)` is assumed to cover `J`; verify with
[`jacobian_ratio`](@ref) if that is in doubt.
"""
gramcover(
    a::AbstractVector, b::AbstractVector, t::Tape,
    outidxs::AbstractVector{<:Integer}, args...
) =
    gramcover(a, b, support_components(t, outidxs), args...)

"""
    gramcover!(s, a, b, t::Tape, outidxs) -> s
    gramcover!(s, a, b, t::Tape, outidxs, w::AbstractVector) -> s
    gramcover!(s, a, b, t::Tape, outidxs, W::AbstractMatrix) -> s

Write [`gramcover`](@ref)'s Gram-matrix cover of the tape outputs `outidxs` into
`s` and return it. `s` must be indexed like the inputs (`eachindex(s) ==
1:t.ninputs`).
"""
gramcover!(
    s::AbstractVector, a::AbstractVector, b::AbstractVector, t::Tape,
    outidxs::AbstractVector{<:Integer}, args...
) =
    gramcover!(s, a, b, support_components(t, outidxs), args...)
