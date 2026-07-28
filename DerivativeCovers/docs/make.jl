using DerivativeCovers
using DerivativeCoverProblems
using Documenter

DocMeta.setdocmeta!(DerivativeCovers, :DocTestSetup, :(using DerivativeCovers); recursive=true)
DocMeta.setdocmeta!(DerivativeCoverProblems, :DocTestSetup, :(using DerivativeCoverProblems); recursive=true)

makedocs(;
    modules=[DerivativeCovers, DerivativeCovers.Diagnostics, DerivativeCoverProblems],
    authors="Tim Holy <tim.holy@gmail.com> and contributors",
    sitename="DerivativeCovers.jl",
    format=Documenter.HTML(;
        canonical="https://HolyLab.github.io/DerivativeCovers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Problem families" => "problems.md",
        "Diagnostics" => "diagnostics.md",
    ],
)

deploydocs(;
    repo="github.com/HolyLab/DerivativeCovers.jl",
    devbranch="main",
)
