# `[sources]` resolves the two packages of this repository.

using DerivativeCovers
using DerivativeCoverProblems
using Documenter
using IntervalArithmetic   # loaded here so the doctests' own `using` cannot emit precompilation output

DocMeta.setdocmeta!(DerivativeCovers, :DocTestSetup, :(using DerivativeCovers); recursive=true)
DocMeta.setdocmeta!(DerivativeCoverProblems, :DocTestSetup, :(using DerivativeCoverProblems); recursive=true)

makedocs(;
    modules=[DerivativeCovers, DerivativeCoverProblems],
    authors="Tim Holy <tim.holy@gmail.com> and contributors",
    sitename="DerivativeCovers.jl",
    checkdocs=:all,
    format=Documenter.HTML(;
        canonical="https://HolyLab.github.io/DerivativeCovers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Introduction" => "index.md",
        "Computing covers" => "covers.md",
        "Using a cover" => "using.md",
        "Scale covariance" => "covariance.md",
        "Tightness" => "tightness.md",
        "Covers over a box" => "intervals.md",
        "Programs and new operations" => "extending.md",
        "Reference" => [
            "Computing and using covers" => "reference/covers.md",
            "Extension interface" => "reference/extending.md",
            "DerivativeCoverProblems" => "reference/problems.md",
        ],
        "Internals" => "internals.md",
    ],
)

deploydocs(;
    repo="github.com/HolyLab/DerivativeCovers.jl",
    devbranch="main",
)
