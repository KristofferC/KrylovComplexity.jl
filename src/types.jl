"""
    KCResult{T<:AbstractFloat}

Container returned by `simulate`. Holds all outputs from a single Krylov complexity run.

Fields
- `b`             : Lanczos coefficients at working precision T
- `t`             : time grid (Float64)
- `K`             : Krylov complexity K(t) (Float64)
- `S`             : Krylov entropy S(t) (Float64)
- `Kdim`          : Krylov dimension (>0 = converged, -1 = max_iter reached)
- `reortho_count` : total number of reorthogonalization passes performed
- `wall_time`     : elapsed wall time in seconds
"""
struct KCResult{T<:AbstractFloat}
    b             :: Vector{T}
    t             :: Vector{Float64}
    K             :: Vector{Float64}
    S             :: Vector{Float64}
    Kdim          :: Int
    reortho_count :: Int
    wall_time     :: Float64
end

function Base.show(io::IO, r::KCResult{T}) where T
    status = r.Kdim > 0 ? "converged (Kdim=$(r.Kdim))" : (r.Kdim == -1 ? "max_iter reached without convergence" : "divergence" )
    print(io, "KCResult{$T}: $status, $(length(r.b)) b-coefficients, ",
              "reortho=$(r.reortho_count), wall_time=$(round(r.wall_time, digits=3))s")
end
