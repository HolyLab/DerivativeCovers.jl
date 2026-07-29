```@meta
CurrentModule = DerivativeCoverProblems
DocTestSetup = quote
    using DerivativeCovers, DerivativeCoverProblems
end
```

# Problem families

This companion package provides parameterized test problems with reference Hessians and
Jacobians. Use them to measure cover tightness and to study how program structure affects it.

A generator returns a [`CoverProblem`](@ref), which contains a function `f`, an evaluation point
`x0`, and, when available, a closed-form reference Hessian from [`exact_hessian`](@ref). Pass
`f` and `x0` to the usual cover functions, then compare the result with the reference:

```jldoctest problems
julia> p = quadratic(6; rank=2, form=:factored);

julia> c = cover_hessian(p.f, p.x0);

julia> H = exact_hessian(p);

julia> all(abs.(H) .<= c.h_sum * c.h_sum')
true

julia> round(DerivativeCovers.cover_ratio(H, c.h_sum); digits=3)
0.879
```

The ratio never exceeds `1`, and values closer to `1` indicate tighter covers. If `n`, `rank`,
and the random seed are fixed, changing `form` represents the same quadratic with a different
computational graph. Differences in the ratio therefore isolate the effect of graph structure:

```jldoctest problems
julia> ratio(p) = round(DerivativeCovers.cover_ratio(exact_hessian(p), cover_hessian(p.f, p.x0).h_sum); digits=3);

julia> [ratio(quadratic(6; rank=2, form=form)) for form in (:entrywise, :matvec, :factored)]
3-element Vector{Float64}:
 0.051
 0.145
 0.879
```

The entrywise form sends all `n^2` entries of `A` through separate nonlinear nodes that
reconverge on the same inputs. The factored form instead uses one node for each of the `rank`
layers, with no overlap. This reconvergence explains the difference in cover tightness.

```@index
Pages = ["problems.md"]
```

```@autodocs
Modules = [DerivativeCoverProblems]
```
