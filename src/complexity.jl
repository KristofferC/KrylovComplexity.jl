
"""
    _rk4_step!(phi, J, dt, k1, k2, k3, k4, tmp)

In-place RK4 step for dphi/dt = -i J phi.  All work arrays are pre-allocated;
no allocation occurs during the step.
"""
function _rk4_step!(φ, J, dt, k1, k2, k3, k4, tmp)
    mul!(k1, J, φ);  @. k1 = -im * k1
    @. tmp = φ + (dt/2) * k1;  mul!(k2, J, tmp);  @. k2 = -im * k2
    @. tmp = φ + (dt/2) * k2;  mul!(k3, J, tmp);  @. k3 = -im * k3
    @. tmp = φ +  dt    * k3;  mul!(k4, J, tmp);  @. k4 = -im * k4
    @. φ  += (dt/6) * (k1 + 2k2 + 2k3 + k4)
end


"""
    _rk4(b, dt, time_max, n_out) -> (t, K, S)

Evolve the Krylov chain Schrodinger equation with fixed step `dt` and record
K(t) and S(t) at `n_out` evenly spaced output times up to `time_max`.
`b` must be a `Vector{Float64}`.
"""
function _rk4(b::Vector{Float64}, dt::Float64, time_max::Float64, n_out::Int)
    K       = length(b) + 1
    J       = Tridiagonal(b, zeros(K), b)
    n_steps = round(Int, time_max / dt)
    stride  = max(1, div(n_steps, n_out))
    n_rec   = div(n_steps, stride) + 1

    φ   = zeros(ComplexF64, K);  φ[1] = 1.0
    k1  = similar(φ); k2 = similar(φ)
    k3  = similar(φ); k4 = similar(φ); tmp = similar(φ)
    p   = Vector{Float64}(undef, K)
    t_out = Vector{Float64}(undef, n_rec)
    K_out = Vector{Float64}(undef, n_rec)
    S_out = Vector{Float64}(undef, n_rec)

    j = 1
    @inbounds for i in 0:n_steps
        if i % stride == 0 && j <= n_rec # save only n_out points no need of saving all the smalle time steps
            p .= abs2.(φ)
            t_out[j] = i * dt
            Kc = 0.0;  for n in 1:K;  Kc += (n-1) * p[n];  end
            Sc = 0.0;  for pj in p;   pj > 0.0 && (Sc -= pj * log(pj));  end
            K_out[j] = Kc
            S_out[j] = Sc
            j += 1
        end
        i == n_steps && break
        _rk4_step!(φ, J, dt, k1, k2, k3, k4, tmp)
        φ ./= sqrt(sum(abs2, φ))
    end
    return t_out[1:j-1], K_out[1:j-1], S_out[1:j-1]
end


"""
    krylov_complexity_entropy(b; time_max, dt_factor, n_out, tol, max_halvings) -> (t, K, S)

Compute Krylov complexity K(t) and Krylov entropy S(t) from the Lanczos
coefficients `b` by solving the chain Schrodinger equation with RK4.

The initial time step is `dt = dt_factor / maximum(b)`.  If the RK4 solution
does not satisfy the norm-conservation tolerance `tol`, the step is halved up
to `max_halvings` times.

Returns three `Vector{Float64}` of length `n_out`: the time grid `t`,
complexity `K`, and entropy `S`.
"""
function krylov_complexity_entropy(b::Vector{T};
    time_max::Real    = 1000.0,
    dt_factor::Real   = 0.1,
    n_out::Int        = 2000,
    tol::Real         = 1e-7,
    max_halvings::Int = 10) where T<:AbstractFloat

    @info "Krylov complexity start" K_dim=length(b)+1 time_max=time_max dt_factor=dt_factor n_out=n_out

    b64 = Float64.(b)
    dt  = Float64(dt_factor) / maximum(b64)
    t, C_K, S_K = _rk4(b64, dt, Float64(time_max), n_out)

    for h in 1:max_halvings
        dt /= 2
        t_fine, K_fine, S_fine = _rk4(b64, dt, Float64(time_max), n_out)
        n   = min(length(C_K), length(K_fine))
        err = sqrt(sum(abs2, K_fine[1:n] .- C_K[1:n]))/n
        t, C_K, S_K = t_fine, K_fine, S_fine
        if err < tol
            @info "Krylov complexity converged" halvings=h dt=dt err=err K_sat=C_K[end]
            return t, C_K, S_K
        end
        @info "RK4 did not converge -> halving the time step" halving=h err=err dt=dt
    end

    @warn "Krylov complexity did not converge after $max_halvings halvings" dt=dt
    return t, C_K, S_K
end
