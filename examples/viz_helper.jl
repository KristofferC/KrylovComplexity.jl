# Standalone visualization helpers for KrylovComplexity.jl results.
#
# Plotting is intentionally not part of the package to keep the installation
# lightweight. Include this file at the top of any script that needs plots:
#
#   include("examples/viz_helper.jl")   # also calls `using CairoMakie`
#
# Functions provided:
#   plot_b(b; title)                          → Figure   (b-coefficient scatter/line plot)
#   plot_complexity(t, K, S=nothing; K_sat)   → Figure   (K(t) and optionally S(t))
#
# Requires the examples environment (already includes CairoMakie):
#   First time: julia --project=examples -e 'using Pkg; Pkg.develop(PackageSpec(path=".")); Pkg.instantiate()'
#   Then run scripts with: julia --project=examples examples/<script>.jl
using CairoMakie

"""
    plot_b(b; title) -> Figure

Scatter + line plot of Lanczos coefficients b_n vs index n.
"""
function plot_b(b::AbstractVector; title::String="Lanczos coefficients")
    fig = Figure()
    ax  = Axis(fig[1,1]; xlabel="n", ylabel="b_n", title=title)
    scatter!(ax, 1:length(b), Float64.(b); markersize=6)
    lines!(ax,   1:length(b), Float64.(b); linewidth=1, color=(:black, 0.3))
    return fig
end

"""
    plot_complexity(t, K, S=nothing; title, K_sat) -> Figure

Plot Krylov complexity K(t) and optionally entropy S(t) side by side.
If `K_sat` is provided, a dashed horizontal line at that value is added to
the K(t) panel.
"""
function plot_complexity(t::AbstractVector, K::AbstractVector,
                         S::Union{AbstractVector,Nothing}=nothing;
                         title::String="Krylov complexity",
                         K_sat=nothing)
    if S === nothing
        fig = Figure()
        ax  = Axis(fig[1,1]; xlabel="t", ylabel="K(t)", title=title)
        lines!(ax, t, K; label="K(t)")
        if K_sat !== nothing
            hlines!(ax, [Float64(K_sat)]; linestyle=:dash, color=:gray, label="Kmax")
            axislegend(ax; position=:rb)
        end
    else
        fig = Figure(size=(900, 380))
        ax1 = Axis(fig[1,1]; xlabel="t", ylabel="K(t)", title="Krylov Complexity")
        ax2 = Axis(fig[1,2]; xlabel="t", ylabel="S(t)", title="Krylov Entropy")
        lines!(ax1, t, K; label="K(t)")
        if K_sat !== nothing
            hlines!(ax1, [Float64(K_sat)]; linestyle=:dash, color=:gray, label="Kmax")
            axislegend(ax1; position=:rb)
        end
        lines!(ax2, t, S)
    end
    return fig
end
