# Symmetric covers of weighted Gram matrices `J'*W*J` of a tape's Jacobian, from one reverse sweep.
#
# For `v >= 0` with `abs(W[k,l]) <= v[k]*v[l]`, the vector `c = Ũ'*v` covers `G = J'*W*J`:
#
#     abs(G[i,j]) <= ∑_{k,l} Ũ_ki abs(W[k,l]) Ũ_lj <= ∑_{k,l} Ũ_ki v_k v_l Ũ_lj = c_i c_j.
#
# `c` is linear in the seeds, so seeding `v[k]` at output node `outidxs[k]` gives it in one sweep.

"""
    gramcover(t::Tape, outidxs) -> s
    gramcover(t::Tape, outidxs, w::AbstractVector) -> s
    gramcover(t::Tape, outidxs, W::Diagonal) -> s
    gramcover(t::Tape, outidxs, W::AbstractMatrix; v=<symmetric cover of abs.(W)>) -> s

Symmetric cover of the Gram matrix `G = J'*W*J` of the Jacobian `J` of the tape outputs
`outidxs`: a nonnegative vector `s`, one entry per input (`1:t.ninputs`), with
`abs(G[i,j]) <= s[i]*s[j]` for every pair. Neither `J` nor `G` is formed. `W` defaults to the
identity (`G = J'*J`, the Gauss–Newton matrix), and a vector `w` means `W = Diagonal(w)`; the
weights may have either sign. `w`, `v`, and both axes of `W` are indexed like `outidxs`, one
entry per output.

No cover of `J` is needed: rescaling the outputs by a diagonal `C` together with
`W -> inv(C)*W*inv(C)` leaves `G` unchanged, so the weight alone fixes the output scale.

For any `v >= 0` with `abs(W[k,l]) <= v[k]*v[l]`, the result is `s = Ũ'*v`, where `Ũ >= abs.(J)`
is the absolute-sensitivity surrogate ([`jacobian_surrogate`](@ref)), and
`abs(G[i,j]) <= ∑_{k,l} Ũ[k,i]*abs(W[k,l])*Ũ[l,j] <= s[i]*s[j]`. For diagonal weights
`v = sqrt.(abs.(w))` (all ones when unweighted), so an output with zero weight contributes
nothing. For a general `W` the default `v` is `MatrixCovers.symcover` of
`max.(abs.(W), abs.(transpose(W)))`, which also admits an asymmetric `W`; a supplied `v` must be
finite, nonnegative, and satisfy `abs(W[k,l]) <= v[k]*v[l]`. A repeated entry of `outidxs` is a
repeated row of `J`, and its seeds add.

`s` comes from one absolute-value reverse sweep seeded `v[k]` at output node `outidxs[k]`, so it
costs `O(L)` for a tape of length `L` and never forms `Ũ`. It sums the weighted rows of `Ũ`, and
for `w >= 0` (or no weight) it brackets the per-output quadrature
`h_gn[j] = sqrt(∑_k w[k]*Ũ[k,j]^2)`:

    h_gn[j] <= s[j] <= sqrt(n_j) * h_gn[j],     n_j = count(>(0), Ũ[:, j]),

where `n_j` is the number of outputs that input `j` reaches. `h_gn` is the minimal symmetric
cover of `Ũ'*W*Ũ` and costs one sweep channel per output, `O(m*L)`. Unweighted, it is the `h_gn`
of [`cover_nls`](@ref); with `Ũ` in hand from a `WithDiagnostics` result, a diagonal weight gives
it as `MatrixCovers.gramcover(Utilde, w)`.

Like the package's other sweep-based covers, `s` carries the roundoff of the sweep without
inflation, so the bound holds to a relative error of a few `eps` per node on the longest path.

# Examples

```jldoctest
julia> r(x) = [x[1] * x[2], exp(x[1]) - x[3], x[2] + 2x[3]];

julia> t, outidxs = tape_function(r, [0.5, 2.0, -1.0]);

julia> round.(DerivativeCovers.gramcover(t, outidxs); digits=4)   # column sums of abs.(J)
3-element Vector{Float64}:
 3.6487
 1.5
 3.0

julia> round.(DerivativeCovers.gramcover(t, outidxs, [1.0, 0.0, 4.0]); digits=4)   # output 2 dropped
3-element Vector{Float64}:
 2.0
 2.5
 4.0
```
"""
gramcover(t::Tape, outidxs::AbstractVector{<:Integer}) =
    _output_sweep(t, outidxs, Iterators.repeated(1, length(outidxs)))

gramcover(t::Tape, outidxs::AbstractVector{<:Integer}, w::AbstractVector) =
    _output_sweep(t, outidxs, _diagonal_gram_seeds(outidxs, w))

gramcover(t::Tape, outidxs::AbstractVector{<:Integer}, W::Diagonal) =
    gramcover(t, outidxs, W.diag)

function gramcover(
        t::Tape, outidxs::AbstractVector{<:Integer}, W::AbstractMatrix;
        v::AbstractVector = _default_gram_seeds(outidxs, W)
    )
    return _output_sweep(t, outidxs, _check_gram_seeds(outidxs, W, v))
end

"""
    gramcover!(s, t::Tape, outidxs) -> s
    gramcover!(s, t::Tape, outidxs, w::AbstractVector) -> s
    gramcover!(s, t::Tape, outidxs, W::Diagonal) -> s
    gramcover!(s, t::Tape, outidxs, W::AbstractMatrix; v=<symmetric cover of abs.(W)>) -> s

Write [`gramcover`](@ref)'s cover of the Gram matrix of the tape outputs `outidxs` into `s` and
return it. `s` must be indexed like the inputs (`eachindex(s) == 1:t.ninputs`).
"""
function gramcover!(
        s::AbstractVector, t::Tape, outidxs::AbstractVector{<:Integer}, args...; kwargs...
    )
    eachindex(s) == Base.OneTo(t.ninputs) || throw(
        DimensionMismatch(
            "s has indices $(eachindex(s)), expected $(Base.OneTo(t.ninputs)) (one per input)"
        )
    )
    return copyto!(s, gramcover(t, outidxs, args...; kwargs...))
end

# `v[k] = sqrt(abs(w[k]))`, so `v[k]^2` covers the diagonal weight `w[k]`.
function _diagonal_gram_seeds(outidxs, w::AbstractVector)
    _check_diagonal_weight(outidxs, w)
    return (sqrt(abs(w[k])) for k in eachindex(w))
end

function _check_diagonal_weight(outidxs, w::AbstractVector)
    eachindex(w) == eachindex(outidxs) || throw(
        DimensionMismatch(
            "w has indices $(eachindex(w)), expected $(eachindex(outidxs)) " *
                "(one per output, indexed like outidxs)"
        )
    )
    for k in eachindex(w)
        isfinite(w[k]) || throw(ArgumentError("weights must be finite; w[$k] = $(w[k])"))
    end
    return nothing
end

# A symmetric cover of `abs.(W)` that also covers `abs.(transpose(W))`, so it bounds every entry
# of an asymmetric `W`.
function _default_gram_seeds(outidxs, W::AbstractMatrix)
    _check_gram_weight(outidxs, W)
    return symcover(max.(abs.(W), abs.(transpose(W))))
end

function _check_gram_weight(outidxs, W::AbstractMatrix)
    ax = axes(outidxs, 1)
    axes(W) == (ax, ax) || throw(
        DimensionMismatch(
            "W has axes $(axes(W)), expected $((ax, ax)) " *
                "(outputs × outputs, indexed like outidxs)"
        )
    )
    return nothing
end

function _check_gram_seeds(outidxs, W::AbstractMatrix, v::AbstractVector)
    _check_gram_weight(outidxs, W)
    eachindex(v) == eachindex(outidxs) || throw(
        DimensionMismatch(
            "v has indices $(eachindex(v)), expected $(eachindex(outidxs)) " *
                "(one per output, indexed like outidxs)"
        )
    )
    for k in eachindex(v)
        (isfinite(v[k]) && v[k] >= 0) ||
            throw(ArgumentError("v must be finite and nonnegative; v[$k] = $(v[k])"))
    end
    iscover(v, v, W) ||
        throw(ArgumentError("v must satisfy abs(W[k,l]) <= v[k]*v[l] for all k, l"))
    return v
end

"""
    gramvariance(t::Tape, outidxs) -> h
    gramvariance(t::Tape, outidxs, w::AbstractVector) -> h
    gramvariance(t::Tape, outidxs, W::Diagonal) -> h

Root-mean-square column scale of the Gram matrix `G = J'*W*J` of the Jacobian `J` of the tape
outputs `outidxs`: a nonnegative vector `h`, one entry per input (`1:t.ninputs`), with

    h[j] = sqrt(∑_k abs(w[k]) * Ũ⁽²⁾[k,j]),

where `Ũ⁽²⁾[k,j]` sums, over the paths from input `j` to output `k`, the squared product of the
local partial magnitudes along the path. `W` defaults to the identity, and a vector `w` means
`W = Diagonal(w)`, indexed like `outidxs`; only `abs.(w)` enters, so an output with zero weight
contributes nothing, and a repeated entry of `outidxs` adds its weights.

Model each parent slot's local partial as carrying an independent, equiprobable sign. Distinct
paths then contribute uncorrelated terms, so `E[J[k,j]^2] = Ũ⁽²⁾[k,j]` and
`h[j]^2 = E[(J'*abs.(W)*J)[j,j]]`: `h` is the root-mean-square column norm of `sqrt.(abs.(W))*J`,
the column (MINPACK) scaling of the Gram matrix in expectation. The entries of `J` for distinct
outputs are uncorrelated too, so off-diagonal weights contribute nothing to this expectation; a
general `W::AbstractMatrix` is an error. On an interval tape the magnitudes are the largest over
the box, so `h` bounds the pointwise `h` at every point of the box.

`h` is a scale, **not a cover**: where several paths reach one entry of `J` with the same sign it
can fall below that column's norm. Per entry `Ũ⁽²⁾ <= Ũ.^2` (a sum of squares is at most the square of
the sum), where `Ũ` is the absolute-sensitivity surrogate ([`jacobian_surrogate`](@ref)), so

    h[j] <= h_gn[j] = sqrt(∑_k abs(w[k]) * Ũ[k,j]^2) <= gramcover(t, outidxs, w)[j].

Unweighted, `h_gn` is the Gauss–Newton cover of [`cover_nls`](@ref). Where each entry of `J` has a
single path, `Ũ⁽²⁾ = Ũ.^2 = abs2.(J)` on a point tape, and `h` equals `h_gn` and the column norms
of `sqrt.(abs.(W))*J`. Rescaling the inputs, `x -> D*x`, gives `h -> inv(D)*h`; rescaling the
outputs by a diagonal `C` together with `w -> w ./ C.^2` leaves `h` unchanged.

`h` comes from one reverse sweep with squared local partials (the variance channel of
[`dependency_sweep`](@ref)) seeded `abs(w[k])` at output node `outidxs[k]`, so it costs `O(L)` for
a tape of length `L`, where `h_gn` needs one channel per output.

# Examples

```jldoctest
julia> r(x) = [x[1] * x[2], exp(x[1]) - x[3], x[2] + 2x[3]];

julia> t, outidxs = tape_function(r, [0.5, 2.0, -1.0]);

julia> round.(DerivativeCovers.gramvariance(t, outidxs); digits=4)   # column norms of J
3-element Vector{Float64}:
 2.592
 1.118
 2.2361

julia> round.(DerivativeCovers.gramvariance(t, outidxs, [1.0, 0.0, 4.0]); digits=4)
3-element Vector{Float64}:
 2.0
 2.0616
 4.0

julia> t2, o2 = tape_function(x -> [x[1] * x[1]], [3.0]);   # two paths, J = [6.0;;]

julia> round.(DerivativeCovers.gramvariance(t2, o2); digits=4)   # sqrt(3^2 + 3^2)
1-element Vector{Float64}:
 4.2426
```
"""
gramvariance(t::Tape, outidxs::AbstractVector{<:Integer}) =
    _output_variance_sweep(t, outidxs, Iterators.repeated(1, length(outidxs)))

function gramvariance(t::Tape, outidxs::AbstractVector{<:Integer}, w::AbstractVector)
    _check_diagonal_weight(outidxs, w)
    return _output_variance_sweep(t, outidxs, (abs(w[k]) for k in eachindex(w)))
end

gramvariance(t::Tape, outidxs::AbstractVector{<:Integer}, W::Diagonal) =
    gramvariance(t, outidxs, W.diag)

gramvariance(::Tape, ::AbstractVector{<:Integer}, W::AbstractMatrix) = throw(
    ArgumentError(
        "gramvariance takes diagonal weights only (a vector or a Diagonal), got a " *
            "$(typeof(W)); off-diagonal weights have zero expected contribution, so pass " *
            "diag(W) if its diagonal is what you mean"
    )
)

# `sqrt.(∑_k seeds[k] * Ũ⁽²⁾[k, :])` from the squared channel of `dependency_sweep`, with the
# linear channel left unseeded.
function _output_variance_sweep(t::Tape{T}, outidxs, seeds) where {T}
    theta2 = _output_seeds(t, outidxs, seeds)
    _, vbar = dependency_sweep(t, zeros(magtype(T), nnodes(t)), theta2)
    return sqrt.(vbar[1:t.ninputs])
end
