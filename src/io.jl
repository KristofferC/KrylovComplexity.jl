"""
    setup_logger(log_dir, file_level, console_level) -> TeeLogger

Create a logger that writes Debug+ to a timestamped file in `log_dir` and
Info+ to stderr.  Useful for long runs where you want a persistent log.
"""
function setup_logger(log_dir::String="logs/",
    file_level = Logging.Debug, console_level=Logging.Info)
    mkpath(log_dir)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    path = joinpath(log_dir, "kc_$(timestamp).log")

    file_logger = MinLevelLogger(FileLogger(path; append=false), file_level)
    console_logger = ConsoleLogger(stderr, console_level)

    return TeeLogger(console_logger, file_logger)
end

"""
    save_b(b, path)

Save Lanczos coefficients to a JLD2 file.  BigFloat vectors are saved with
their precision recorded so `load_b` can reconstruct them exactly.
"""
function save_b(b::AbstractVector, path::AbstractString)
    bits = eltype(b) <: BigFloat ? precision(first(b)) : nothing
    JLD2.jldsave(path; b=b, bf_precision=bits)
    @info "Saved b" path=path n=length(b) T=eltype(b)
end

"""
    load_b(path) -> Vector

Load Lanczos coefficients from a JLD2 file written by `save_b`.  BigFloat
vectors are reconstructed at the stored precision.
"""
function load_b(path::AbstractString)
    bits = JLD2.load(path, "bf_precision")
    b = if bits !== nothing
        setprecision(BigFloat, bits) do
            JLD2.load(path, "b")
        end
    else
        JLD2.load(path, "b")
    end
    @info "Loaded b" path=path n=length(b) T=eltype(b)
    return b
end

"""
    save_result(r, path)

Save a `KCResult` to a JLD2 file (all fields, BigFloat-compatible).
"""
function save_result(r::KCResult, path::AbstractString)
    bits = eltype(r.b) <: BigFloat ? precision(first(r.b)) : nothing
    JLD2.jldsave(path; b=r.b, bf_precision=bits,
                 t=r.t, K=r.K, S=r.S,
                 Kdim=r.Kdim, reortho_count=r.reortho_count, wall_time=r.wall_time)
    @info "Saved result" path=path
end

"""
    load_result(path) -> KCResult

Load a `KCResult` from a JLD2 file written by `save_result`.
"""
function load_result(path::AbstractString)
    data = JLD2.load(path)
    bits = get(data, "bf_precision", nothing)
    b = if bits !== nothing
        setprecision(BigFloat, bits) do
            JLD2.load(path, "b")
        end
    else
        data["b"]
    end
    @info "Loaded result" path=path Kdim=data["Kdim"] T=eltype(b)
    return KCResult(b, data["t"], data["K"], data["S"],
                    data["Kdim"], data["reortho_count"], data["wall_time"])
end
