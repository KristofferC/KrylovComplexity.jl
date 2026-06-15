using KrylovComplexity
using CairoMakie

# Convergence study: CER vs nCER for cSYK N=8 at three precision levels.
#
# CER  (remove_degeneracies=true)  — Constraint Eigenbasis Representation: degenerate
#   Liouvillian frequency blocks are collapsed and zero-weight transitions discarded.
#   This shrinks the effective Krylov dimension K_eff avoid spurious leakage in other symmetry sectors
#
# nCER (remove_degeneracies=false) — degenerate eigenvalues of the Hamiltonian and eigenspace where ty
#
# For each precision (Float64, Double64, BigFloat-256) the script runs both CER and nCER
# Lanczos, prints a comparison table (Krylov dimension, wall time, MARE),
# and saves a plot of all six b-coefficient sequences to csyk_N8_cer.png.
#
# Reortho: all runs use :dpro.
#
# Reference: arXiv:2507.06286
#
# Run (from the KrylovComplexity.jl/ directory):
#   First time: julia --project=examples -e 'using Pkg; Pkg.develop(PackageSpec(path=".")); Pkg.instantiate()'
#   Then:       julia --project=examples examples/csyk_convergence_arxiv_250706286.jl
H, O = build_system(Dict("model" => "csyk", "N" => 8, "precision_build" => 1024, "J" => 1.0))

# Double64
t_cer_double  = @elapsed b_cer_double,  _, oc_cer_double  = lanczos(H, O; precision=:double, reortho=:dpro, remove_degeneracies=true)
t_ncer_double = @elapsed b_ncer_double, _, oc_ncer_double = lanczos(H, O; precision=:double, reortho=:dpro, remove_degeneracies=false)

# Float64
t_cer_f64     = @elapsed b_cer_f64,     _, oc_cer_f64     = lanczos(H, O; precision=:f64,    reortho=:dpro, remove_degeneracies=true)
t_ncer_f64    = @elapsed b_ncer_f64,    _, oc_ncer_f64    = lanczos(H, O; precision=:f64,    reortho=:dpro, remove_degeneracies=false)

# BigFloat 256-bit
t_cer_bf256   = @elapsed b_cer_bf256,   _, oc_cer_bf256   = lanczos(H, O; precision=256,     reortho=:dpro, remove_degeneracies=true)
t_ncer_bf256  = @elapsed b_ncer_bf256,  _, oc_ncer_bf256  = lanczos(H, O; precision=256,     reortho=:dpro, remove_degeneracies=false)

# ── Table: CER vs nCER per precision ─────────────────────────────────────────
# MARE(a,b) = (1/K) Σ |a_n - b_n| / |a_n|   where K = min(len(a), len(b))
function mare(a, b)
    n  = min(length(a), length(b))
    af = Float64.(a[1:n])
    bf = Float64.(b[1:n])
    sum(abs.(af .- bf) ./ af) / n
end

println("\nCER vs nCER comparison — cSYK N=8:")
println("  Reortho: :dpro for all precisions and both CER/nCER")
println()
println("  $(rpad("precision", 14)) $(rpad("K(cer)", 8)) $(rpad("K(ncer)", 8)) " *
        "$(rpad("t_cer(s)", 10)) $(rpad("t_ncer(s)", 11)) $(rpad("oc_cer", 8)) $(rpad("oc_ncer", 9)) MARE")
println("  " * "─"^85)
for (name, b_cer, b_ncer, t_cer, t_ncer, oc_cer, oc_ncer) in [
    ("Float64",     b_cer_f64,    b_ncer_f64,    t_cer_f64,    t_ncer_f64,    oc_cer_f64,    oc_ncer_f64),
    ("Double64",    b_cer_double, b_ncer_double, t_cer_double, t_ncer_double, oc_cer_double, oc_ncer_double),
    ("BigFloat256", b_cer_bf256,  b_ncer_bf256,  t_cer_bf256,  t_ncer_bf256,  oc_cer_bf256,  oc_ncer_bf256),
]
    println("  $(rpad(name,14)) $(rpad(length(b_cer)+1,8)) $(rpad(length(b_ncer)+1,8)) " *
            "$(rpad(round(t_cer,digits=2),10)) $(rpad(round(t_ncer,digits=2),11)) " *
            "$(rpad(oc_cer,8)) $(rpad(oc_ncer,9)) $(round(mare(b_cer, b_ncer), sigdigits=4))")
end

# ── Cross-precision MARE (CER, using bf256 as reference) ─────────────────────
println()
println("  Cross-precision MARE (CER, bf256 as reference):")
println("  $(rpad("f64   vs bf256", 22)) $(round(mare(b_cer_f64,    b_cer_bf256), sigdigits=4))")
println("  $(rpad("double vs bf256", 22)) $(round(mare(b_cer_double, b_cer_bf256), sigdigits=4))")
println("  $(rpad("f64   vs double", 22)) $(round(mare(b_cer_f64,    b_cer_double), sigdigits=4))")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = Figure(size=(900, 500))
ax  = Axis(fig[1,1]; xlabel="n", ylabel="bₙ", title="cSYK N=8 — CER vs nCER")

lines!(ax, 1:length(b_cer_f64),    Float64.(b_cer_f64);    label="f64    cer",  color=:red)
lines!(ax, 1:length(b_ncer_f64),   Float64.(b_ncer_f64);   label="f64    ncer", color=:red,   linestyle=:dash)
lines!(ax, 1:length(b_cer_double), Float64.(b_cer_double);  label="double cer",  color=:blue)
lines!(ax, 1:length(b_ncer_double),Float64.(b_ncer_double); label="double ncer", color=:blue,  linestyle=:dash)
lines!(ax, 1:length(b_cer_bf256),  Float64.(b_cer_bf256);  label="bf256  cer",  color=:green)
lines!(ax, 1:length(b_ncer_bf256), Float64.(b_ncer_bf256); label="bf256  ncer", color=:green, linestyle=:dash)

axislegend(ax; position=:rt)
save("csyk_N8_cer.png", fig)
fig
