import KrylovComplexity:
_resolve_eps,
build_liouvillian_eigenbasis,
_reortho_fro!, _reortho_dro!,
_lanczos_vectorial!,
lanczos,
frobenius_norm

@testset "Lanczos" begin

@testset "build_liouvillian_eigenbasis" begin
    N = 4
    H, O = build_system(Dict("model"=>"ising", "N"=>N, "hz"=>0.5, "precision_build"=>1024))

    E_eff, V_eff = build_liouvillian_eigenbasis(H,O)
    @test issorted(real.(E_eff))
    @test all(V_eff .>= 0)
    @test length(E_eff) == length(V_eff)
    @test length(E_eff) < size(H,1)^2
    @test precision(real(first(E_eff))) == 1024
    @test precision(real(first(V_eff))) == 1024

    # without degeneracy removal: more frequencies kept
    E_all, V_all = build_liouvillian_eigenbasis(H, O; remove_degeneracies=false)
    @test length(E_all) >= length(E_eff)

    commutator = Matrix(H)*Matrix(O) - Matrix(O)*Matrix(H)   # = [H, O]
    lhs = frobenius_norm(commutator)^2                   # frobenius norm squared
    rhs = sum((E_all .* V_all).^2) /size(H,1)               # Σ ω²|Õ|²
    @test lhs ≈ rhs  atol=1e-8                               # Parseval type identity
    @test frobenius_norm(Matrix(O))^2 ≈ sum(V_all.^2)/size(H,1)  atol=1e-8 #
end

@testset "Lanczos FRO — Ising N=4 integrable" begin
    N = 4
    H, O = build_system(Dict("model"=>"ising", "N"=>N, "precision_build" => 1024))  # hz=0: integrable
    b_f64, _, _ = lanczos(H, O; precision=:f64, reortho=:fro)
    b_d, _, _ = lanczos(H, O; precision=:double, reortho=:fro)
    b_1024, _, _ = lanczos(H, O; precision=1024, reortho=:fro)

    @test b_f64 isa Vector{Float64}
    @test b_d isa Vector{Double64}
    @test b_1024 isa Vector{BigFloat}
    @test precision((first(b_1024))) == 1024
    @test all(b_f64 .> 0)                                       # all hoppings positive
    @test all(b_d .> 0)                                         # all hoppings positive
    @test all(b_1024 .> 0)                                      # all hoppings positive


    K_max = size(H,1)^2 - size(H,1) + 1
    @test length(b_f64) < K_max            # converged before K_max (integrable)
    @test length(b_d) < K_max            # converged before K_max (integrable)
    @test length(b_1024) < K_max            # converged before K_max (integrable)

    @test length(b_f64) == length(b_1024)
    @test length(b_1024) == length(b_d)
    
    n = min(length(b_f64), length(b_d) ,length(b_1024))
    @test Float64.(b_1024[1:n]) ≈ b_f64[1:n] rtol=1e-10
    @test Float64.(b_1024[1:n]) ≈ Float64.(b_d[1:n]) rtol=1e-10
end


@testset "Tridiagonal eigenvalue test — Ising N=4" begin
    # Eigenvalues of SymTridiagonal(zeros, b) must match
    # a subset of eigenvalues of the full Liouvillian
    N = 4
    H, O = build_system(Dict("model"=>"ising", "N"=>N, "hz"=>0.5))
    b, _, _ = lanczos(H, O; precision=:f64, reortho=:fro)

    # Krylov tridiagonal eigenvalues
    K = length(b) + 1
    T_b = eltype(b)
    λ_krylov = sort(real.(eigvals(SymTridiagonal(zeros(T_b,(K)), b))))

    # Full Liouvillian eigenvalues (for small N this is feasible)
    D  = size(H,1)
    H_dense = Matrix(H)
    # L(v) = H*v - v*H, vectorized: L_vec = H⊗I - I⊗H^T
    L = kron(H_dense, I(D)) - kron(I(D), transpose(H_dense))
    λ_full = sort(real.(eigvals(L)))

    # Every Krylov eigenvalue should appear in the full spectrum
    for λ in λ_krylov
        @test minimum(abs.(λ_full .- λ)) < 1e-8
    end
end

@testset "FRO vs DRO consistency" begin
    H, O = build_system(Dict("model"=>"ising", "N"=>4, "hz"=>0.5))
    b_fro, _, count_fro = lanczos(H, O; precision=:f64, reortho=:fro)
    b_dro, _, count_dro = lanczos(H, O; precision=:f64, reortho=:dro)
    n = min(length(b_fro), length(b_dro))
    @test b_fro[1:n] ≈ b_dro[1:n]  rtol=1e-10
    @test count_dro >= count_fro    # DRO does more reorthogonalizations
end

@testset "Integrable < chaotic Krylov dimension" begin
    H_int, O_int = build_system(Dict("model"=>"ising", "N"=>4))          # hz=0
    H_cha, O_cha = build_system(Dict("model"=>"ising", "N"=>4, "hz"=>0.5))
    b_int, Kdim_int, _ = lanczos(H_int, O_int; precision=:f64, reortho=:fro)
    b_cha, Kdim_cha, _ = lanczos(H_cha, O_cha; precision=:f64, reortho=:fro)
    @test Kdim_int < Kdim_cha       # chaotic systems have larger Krylov dimension
end

end  # Lanczos