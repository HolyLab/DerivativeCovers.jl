# Composition of covers and declared sub-maps.
#
# Magnitude covers compose without a joint tape:
# (a₂b₂')(a₁b₁') = (b₂⋅a₁)a₂b₁'. A `submapnode` records exact Jacobian rows and declared
# curvature covers, keeping the outer tape independent of the sub-map's internal node count.

"""
    compose_jacobian_cover(a2, b2, a1, b1) -> (; a, b)

Cover of a product of covered factors: if `abs.(J2) <= a2*b2'` (`m × q`) and
`abs.(J1) <= a1*b1'` (`q × n`), then `abs.(J2*J1) <= (b2⋅a1) * a2*b1'`, so the returned pair
`a = sqrt(b2⋅a1)*a2`, `b = sqrt(b2⋅a1)*b1` covers `J2*J1` — the Jacobian of a composition
`F2 ∘ F1` wherever each factor's cover holds at the matching point. For the composite cover to hold
over a set of points `x`, such as a box, `(a1, b1)` must hold at every `x` in the set and
`(a2, b2)` at every `F1(x)`. The inner product `b2⋅a1` is
invariant under rescalings of the input, middle, and output spaces (`b2` and `a1` carry inverse
middle-space units), and its square root is split onto both sides so the rule commutes with a
rescaling of either factor.

Equality holds entrywise for sign-definite rank-1 factors covered exactly; in general the rule
pays the same signed-cancellation looseness as any magnitude-level bound on a product.

The arithmetic is that of the inputs' element type. Pass them as [`Upward`](@ref) vectors for a
composite that holds exactly rather than up to rounding, as covers over a box should.
"""
function compose_jacobian_cover(
        a2::AbstractVector, b2::AbstractVector,
        a1::AbstractVector, b1::AbstractVector
    )
    axes(b2, 1) == axes(a1, 1) ||
        throw(DimensionMismatch("b2 and a1 must share the middle space: $(axes(b2)) vs $(axes(a1))"))
    for (name, v) in (("a2", a2), ("b2", b2), ("a1", a1), ("b1", b1))
        all(>=(0), v) || throw(ArgumentError("cover vectors must be nonnegative; $name is not"))
    end
    c = sqrt(dot(b2, a1))
    return (; a = c .* a2, b = c .* b1)
end

"""
    compose_hessian_cover(hg, gamma, a, b, S=nothing) -> (; h_sum, h_quad)

Symmetric covers of the Hessian of a composite `f = g ∘ F` at `x`, with `y = F(x)`, from covers
of the pieces:

- `hg`: symmetric cover of `∇²g(y)` (`abs.(∇²g) <= hg*hg'`), one entry per output of `F`;
- `gamma`: entrywise bound `gamma[k] >= abs(∂g/∂y_k)` at `y`;
- `(a, b)`: Jacobian cover of `F` at `x` (`abs.(J_F) <= a*b'`);
- `S`: curvature covers with `abs.(∇²F_k) <= S[k, :]*S[k, :]'` (as [`declare_submap`](@ref)
  returns); `nothing` declares `F` linear.

The result covers `∇²f` at every `x` where all four inputs hold. At a single point, exact gradient
magnitudes qualify as `gamma`. For a cover over a set of points, such as a box, each input must
hold over the whole set: `(a, b)` and `S` at every `x` in it, and `hg` and `gamma` at every
`F(x)`; the [`magnitude`](@ref)s of an interval gradient enclosing `∂g/∂y` over the set qualify as
`gamma`. The arithmetic is that of the inputs' element type; pass them as [`Upward`](@ref) vectors
for a composite that holds exactly rather than up to rounding, as covers over a box should.

The chain rule `∇²f = J_F'*∇²g*J_F + Σ_k (∂g/∂y_k)*∇²F_k` bounds entrywise as
`abs.(∇²f) <= (a⋅hg)² * b*b' + Σ_k gamma[k] * S[k,:]*S[k,:]'` — a sum of rank-1 magnitude atoms
`u_0 = (a⋅hg)*b` and `u_k = sqrt(gamma[k])*S[k,:]`. Both merges are returned: the sum merge
`h_sum = Σ_α u_α` (cross terms are nonnegative slack) and the quadrature merge
`h_quad[i] = sqrt(Σ_α u_α[i]²)` (Cauchy–Schwarz), with `h_quad <= h_sum` coordinatewise.

For `g = 0.5*norm(r)^2` this generic rule gives `norm(a, 1)*b` for the Gauss–Newton term: the
sum-merge Gram cover with each row of the surrogate `Ũ` replaced by its rank-1 bound `a[k]*b`. It
is therefore at least the tape's `gramcover(t, outidxs) = Ũ'*ones(m)`, which is the default
`h_gn` of [`cover_nls`](@ref) and in turn at least its per-residual quadrature
(`ngroups=nothing`); NLS consumers should prefer `cover_nls`.
"""
function compose_hessian_cover(
        hg::AbstractVector, gamma::AbstractVector, a::AbstractVector,
        b::AbstractVector, S::Union{Nothing, AbstractMatrix} = nothing
    )
    axes(hg, 1) == axes(gamma, 1) == axes(a, 1) ||
        throw(
        DimensionMismatch(
            "hg, gamma, and a must share the output space: " *
                "$(axes(hg)) vs $(axes(gamma)) vs $(axes(a))"
        )
    )
    for (name, v) in (("hg", hg), ("gamma", gamma), ("a", a), ("b", b))
        all(>=(0), v) || throw(ArgumentError("cover vectors must be nonnegative; $name is not"))
    end
    h_sum = dot(a, hg) .* b
    hsq = h_sum .^ 2
    if S !== nothing
        axes(S) == (axes(a, 1), axes(b, 1)) ||
            throw(DimensionMismatch("S is $(axes(S)); expected ($(axes(a, 1)), $(axes(b, 1))) (outputs × inputs)"))
        all(>=(0), S) || throw(ArgumentError("curvature covers S must be nonnegative"))
        for k in axes(S, 1)
            g = sqrt(gamma[k])
            g == 0 && continue
            for j in axes(S, 2)
                u = g * S[k, j]
                h_sum[j] += u
                hsq[j] += u^2
            end
        end
    end
    return (; h_sum, h_quad = sqrt.(hsq))
end

"""
    declare_submap(F, x0::AbstractVector; scales=nothing, ratio=nothing) -> (; y, J, S)

The declaration [`submapnode`](@ref) consumes, distilled from `F`'s own scratch tape at `x0`:
exact values `y = F(x0)`, the exact signed Jacobian `J` from per-output reverse sweeps, and
curvature covers `S` whose row `k` is the sum-merge cover of `∇²F_k`, so
`abs.(∇²F_k) <= S[k, :]*S[k, :]'` by [`cover_hessian`](@ref)'s guarantee. `F` may return a
single `TVar` (declared as one output) or a vector of them; a linear output gets an exactly
zero row of `S`.

Costs `O(q*L_F)` time — one `q`-channel adjoint sweep over `F`'s tape plus one absolute sweep per
output — once per declaration, and `O(q*n)` memory for `J` and `S`, which places it at diagnostic
tier alongside [`submapnode`](@ref); the scratch tape is discarded afterwards. The declaration
transforms covariantly under rescalings of `F`'s
arguments and outputs because each ingredient inherits the covariance of the underlying sweeps,
including the zero-parent-value fallback exception of the local tie-break.

`scales` and `ratio` govern the split behind the rows of `S`, as in [`cover_hessian`](@ref). The
scratch tape's inputs are `F`'s arguments, so `scales` is expressed in *those* units, not in the
recording tape's. The two splits are otherwise independent: the recording tape consumes `S` as a
fixed bound, so its covers stay valid whatever choice stands behind the declaration.

`y` and `J` are in the element type of `x0` (as [`Tape`](@ref) assigns it) and `S` in its
[`magtype`](@ref). The declaration holds wherever `x0` does: at a point for numeric `x0`, and at
every point of the box for an interval `x0`, whose `y` and `J` then enclose the values and
Jacobian over the box and whose `S` covers the curvature there. A tape whose values stand for sets
accepts only a declaration of the second kind (see [`submapnode`](@ref)).
"""
function declare_submap(F, x0::AbstractVector; scales = nothing, ratio = nothing)
    t, out = tape_function(F, x0)
    outidxs = out isa Integer ? [out] : out
    q, n = length(outidxs), t.ninputs
    y = t.vals[outidxs]
    # Each sweep row belongs to one output, so the same traversal provides its Jacobian row and
    # local covers.
    Vbar = adjoints(t, outidxs)
    J = Vbar[:, 1:n]
    S = Matrix{magtype(eltype(t))}(undef, q, n)
    for k in 1:q
        _, theta = local_covers(t, Vbar[k, :]; scales, ratio)
        S[k, :] = abs_reverse(t, theta)[1:n]
    end
    return (; y, J, S)
end

"""
    submap(F, xs::AbstractVector{<:CoverVar}; scales=nothing, ratio=nothing) -> Vector{<:CoverVar}

Record `F` as a single declared call on the tape of `xs`: evaluate [`declare_submap`](@ref) at
the parent values and embed the result via [`submapnode`](@ref). The recording tape gains one
node per output of `F` — independent of how many operations `F` performs internally — at the
price that exact Hessian-vector products ([`hvp`](@ref)) become unavailable on that tape. A
scalar-valued `F` yields a one-element vector. `scales` and `ratio` are passed to
[`declare_submap`](@ref) and govern only the declaration's curvature covers.

The declaration is recorded on the parent values themselves, so on a tape whose values stand for
sets, such as intervals over a box, it holds over the same sets as the rest of the tape.

The declaration it embeds is dense in the sub-map's outputs and arguments, so this keeps the outer
tape on one tape at diagnostic tier. [`compose_jacobian_cover`](@ref) and
[`compose_hessian_cover`](@ref) cover the same composite from covers of the factors alone.
"""
function submap(F, xs::AbstractVector{<:CoverVar}; scales = nothing, ratio = nothing)
    isempty(xs) && throw(ArgumentError("cannot record a sub-map call with no arguments"))
    d = declare_submap(F, [value(x) for x in xs]; scales, ratio)
    return submapnode(xs, d.y, d.J, d.S)
end
