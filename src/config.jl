"""
    load_config(cfg) -> Dict{String,Any}

Accept either a path to a YAML file or a `Dict` and return a flat
`Dict{String,Any}`.  When a `Dict` is passed it is returned as-is.
"""
load_config(path::AbstractString) = YAML.load_file(path; dicttype=Dict{String,Any})
load_config(cfg::Dict)            = cfg

"""
    simulate(cfg; kwargs...) -> KCResult

Run the full pipeline (build system, Lanczos, complexity) from a YAML file
path or a `Dict`.  Keyword arguments override config file values.

Recognised keys (beyond model-specific ones):
- `precision`          : Lanczos working precision (`:f64`, `:double`, integer bits)
- `reortho`            : reorthogonalization method (`:none`, `:fro`, `:dro`, `:pro`, `:dpro`)
- `remove_degeneracies`: collapse degenerate Liouvillian blocks (default `true`)
- `eps_reortho_factor` : PRO trigger threshold factor (default: 1.0 * sqrt(eps(T)))
- `eps_factor`         : convergence threshold factor
- `tol_factor`         : degeneracy tolerance factor
- `max_iter`           : maximum Lanczos iterations
- `time_max`           : integration time (default 1000.0)
- `dt_factor`          : initial RK4 step = dt_factor / max(b) (default 0.1)
- `kc_out_resolution`  : number of output time points (default 2000)
- `output_dir`         : directory for output files (default `"results/"`)
- `save`               : save full KCResult to `output_dir/result.jld2`
- `save_b`             : save b only to `output_dir/b.jld2` (ignored if `save` is true)
"""
function simulate(cfg::Union{Dict, AbstractString}; kwargs...)
    config = merge(load_config(cfg), Dict(string(k) => v for (k,v) in kwargs))

    t0 = time()
    H, O = build_system(config)

    prec_raw            = get(config, "precision", "f64")
    precision           = prec_raw isa String ? Symbol(prec_raw) : prec_raw   # "f64" -> :f64, 256 -> 256
    reortho             = Symbol(get(config, "reortho", "pro"))
    remove_degeneracies = get(config, "remove_degeneracies", true)
    eps_reortho_factor  = get(config, "eps_reortho_factor", nothing)
    eps_factor          = get(config, "eps_factor", nothing)
    tol_factor          = get(config, "tol_factor", nothing)
    max_iter            = get(config, "max_iter", nothing)
    b, Kdim, reortho_count = lanczos(H, O;
        precision=precision, reortho=reortho,
        remove_degeneracies=remove_degeneracies,
        eps_reortho_factor=eps_reortho_factor,
        eps_factor=eps_factor,
        tol_factor=tol_factor,
        max_iter=max_iter)

    time_max  = Float64(get(config, "time_max",  1000.0))
    dt_factor = Float64(get(config, "dt_factor", 0.1))
    # number of time points in the K(t) / S(t) output arrays
    n_out     = Int(get(config, "kc_out_resolution", 2000))
    t, K, S = krylov_complexity_entropy(b; time_max=time_max, dt_factor=dt_factor, n_out=n_out)

    result = KCResult(b, t, K, S, Kdim, reortho_count, time() - t0)

    out_dir   = get(config, "output_dir", "results/")
    do_save   = get(config, "save",   false)
    do_save_b = get(config, "save_b", false)

    if do_save
        mkpath(out_dir)
        save_result(result, joinpath(out_dir, "result.jld2"))
    elseif do_save_b
        mkpath(out_dir)
        save_b(result.b, joinpath(out_dir, "b.jld2"))
    end

    return result
end
