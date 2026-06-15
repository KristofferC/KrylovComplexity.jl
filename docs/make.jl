using Documenter
using KrylovComplexity

makedocs(
    sitename = "KrylovComplexity.jl",
    authors  = "shanoether",
    modules  = [KrylovComplexity],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical  = "https://shanoether.github.io/KrylovComplexity.jl",
    ),
    pages = [
        "Home"          => "index.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
    warnonly  = false,
)

deploydocs(
    repo = "github.com/shanoether/KrylovComplexity.jl",
    devbranch = "main",
)