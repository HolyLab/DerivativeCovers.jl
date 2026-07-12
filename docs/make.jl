using CoverAD
using Documenter

DocMeta.setdocmeta!(CoverAD, :DocTestSetup, :(using CoverAD); recursive=true)

makedocs(;
    modules=[CoverAD],
    authors="Tim Holy <tim.holy@gmail.com> and contributors",
    sitename="CoverAD.jl",
    format=Documenter.HTML(;
        canonical="https://HolyLab.github.io/CoverAD.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/HolyLab/CoverAD.jl",
    devbranch="main",
)
