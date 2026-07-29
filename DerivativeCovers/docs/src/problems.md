```@meta
CurrentModule = DerivativeCoverProblems
DocTestSetup = quote
    using DerivativeCovers, DerivativeCoverProblems
end
```

# Problem families

Parameterized families of programs, with reference Hessians and Jacobians, for measuring how
loose a cover is and what makes it so.

A generator returns a [`CoverProblem`](@ref): the program `f`, the point `x0`, and — when a
closed form exists — the reference Hessian reached through [`exact_hessian`](@ref). Cover a
problem the way any other program is covered, and score the result against the reference:

```jldoctest problems
julia> p = quadratic(6; rank=2, form=:factored);

julia> c = cover_hessian(p.f, p.x0);

julia> H = exact_hessian(p);

julia> all(abs.(H) .<= c.h_sum * c.h_sum')
true

julia> round(DerivativeCovers.cover_ratio(H, c.h_sum); digits=3)
0.879
```

The ratio is what the corpus exists to measure: it never exceeds `1` — the cover is guaranteed —
and it approaches `1` as the cover approaches the tightest one the Hessian admits. Holding `n`,
`rank`, and the random seed fixed while varying `form` writes the *same* quadratic as three
different programs, so the spread in the ratio isolates the cost of graph structure alone:

```jldoctest problems
julia> ratio(p) = round(DerivativeCovers.cover_ratio(exact_hessian(p), cover_hessian(p.f, p.x0).h_sum); digits=3);

julia> [ratio(quadratic(6; rank=2, form=form)) for form in (:entrywise, :matvec, :factored)]
3-element Vector{Float64}:
 0.051
 0.145
 0.879
```

Reconvergence is the driver: the entrywise form routes every one of the `n^2` entries of `A`
through its own nonlinear node onto the same inputs, while the factored form gives each of the
`rank` layers a node of its own and nothing overlaps.

```@index
Pages = ["problems.md"]
```

```@autodocs
Modules = [DerivativeCoverProblems]
```
