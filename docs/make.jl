using DerivativeCovers
using Documenter

DocMeta.setdocmeta!(DerivativeCovers, :DocTestSetup, :(using DerivativeCovers); recursive=true)

makedocs(;
    modules=[DerivativeCovers],
    authors="Tim Holy <tim.holy@gmail.com> and contributors",
    sitename="DerivativeCovers.jl",
    format=Documenter.HTML(;
        canonical="https://HolyLab.github.io/DerivativeCovers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/HolyLab/DerivativeCovers.jl",
    devbranch="main",
)
