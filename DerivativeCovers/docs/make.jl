# `[sources]` resolves the two packages of this repository. MatrixCovers is unregistered and lives
# outside it, so building here needs `Pkg.develop(PackageSpec(path=<path to MatrixCovers>))` first.

using DerivativeCovers
using DerivativeCoverProblems
using Documenter

DocMeta.setdocmeta!(DerivativeCovers, :DocTestSetup, :(using DerivativeCovers); recursive=true)
DocMeta.setdocmeta!(DerivativeCoverProblems, :DocTestSetup, :(using DerivativeCoverProblems); recursive=true)

makedocs(;
    modules=[DerivativeCovers, DerivativeCovers.Diagnostics, DerivativeCoverProblems],
    authors="Tim Holy <tim.holy@gmail.com> and contributors",
    sitename="DerivativeCovers.jl",
    checkdocs=:all,
    format=Documenter.HTML(;
        canonical="https://HolyLab.github.io/DerivativeCovers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Quantities and extension points" => "reference.md",
        "Diagnostics" => "diagnostics.md",
        "Problem families" => "problems.md",
        "Internals" => "internals.md",
    ],
)

deploydocs(;
    repo="github.com/HolyLab/DerivativeCovers.jl",
    devbranch="main",
)
