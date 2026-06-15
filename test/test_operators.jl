# imports
import KrylovComplexity: pauli_x, pauli_y, pauli_z, id2,
    kron_list, build_site_operator,
    frobenius_inner, frobenius_norm,
    ising_hamiltonian, ising_operator, build_system, GOE_matrix, GUE_matrix, xyz_hamiltonian, xyz_operator,
    xxz_hamiltonian, xxz_operator, _reverse_bits, _parity_basis, _project_to_sector, _block_magnetization,
    _csyk_bit, _csyk_flip, _csyk_parity_left, _csyk_annihilate, _csyk_create, _csyk_basis, csyk_hamiltonian, csyk_operator

# ----------------

@testset "Operators" begin

@testset "Pauli Matrices" begin
    # correct type
    @test eltype(pauli_x(Float64)) == ComplexF64
    @test eltype(pauli_x(Double64)) == Complex{Double64}

    M = setprecision(BigFloat, 1024) do
        p = pauli_x(BigFloat)
        @test eltype(p) == Complex{BigFloat}
        @test precision(BigFloat) == 1024
        Matrix(p)
    end
    @test precision(real(M[1,2])) == 1024

    # correct values
    @test Matrix(pauli_x(Float64)) ≈ [0 1; 1 0]
    @test Matrix(pauli_y(Float64)) ≈ [0 -im; im 0]
    @test Matrix(pauli_z(Float64)) ≈ [1 0; 0 -1]
    # Hermitian
    @test pauli_x(Float64) ≈ pauli_x(Float64)'
end

@testset "Frobenius" begin
    A = Matrix(pauli_z(Float64))
    @test frobenius_inner(A,A) ≈ 1.0
    @test frobenius_norm(A)   ≈ 1.0
end

# ----------------
# test ising model
@testset "Ising Hamiltonian" begin
    N = 4
    H = ising_hamiltonian(N)
    @test size(H) == (2^N,2^N)
    @test H ≈ H'                  # Hermitian
    @test tr(H) ≈ 0.0 atol=1e-12  #"Should be traceless: Pauli matrices are traceless and tr(kron(A,B)) = tr(A)tr(B)"

    # obc has less bounds than pbc
    H_obc = ising_hamiltonian(N; bc=:obc)
    @test nnz(H_obc) >= nnz(H)  #"PBC should have fewer nnz elements as it has translation symmetry -> more blocks -> more 0"
end

@testset "Ising Operator" begin
    N = 4
    O = ising_operator(N)
    @test size(O) == (2^N,2^N)
    @test O ≈ O'
    @test tr(O) ≈ 0.0 atol=1e-12
end

@testset "Build system" begin
    N = 4
    cfg = Dict("model" => "ising", "N" => N, "precision_build" => 1024)
    H, O = build_system(cfg)
    @test size(H) == (2^N,2^N)
    @test size(O) == (2^N,2^N)
    @test H ≈ H'                     # Hermitian
    @test frobenius_norm(Matrix(O)) ≈ 1.0 atol=1e-12
    @test O ≈ O'
    @test precision(real(first(nonzeros(H)))) == 1024
    @test precision(real(first(nonzeros(O)))) == 1024
end

@testset "build_system errors" begin
    @test_throws ErrorException build_system(Dict("N"=>4))                  #"Missing model in cfg"
    @test_throws ErrorException build_system(Dict("model"=>"ising"))        #"Missing N in cfg"
    @test_throws ErrorException build_system(Dict("model"=>"ising", "N"=>4,
                                                    "wrong_key" => 1.0))    #"Unknown key in cfg"
    @test_throws ErrorException build_system(Dict("model"=>"unknown_model", "N"=>4)) #"Unknown model in cfg"
    @test_throws ErrorException build_system(Dict("model"=>"ising", "N"=>4, "precision_build"=>-1)) #"precision_build needs to be positive"
end

# ----------------
# test  xyz
@testset "XYZ Hamiltonian" begin
    N = 4
    H = xyz_hamiltonian(N; bc=:pbc)                    # integrable: c=0
    @test size(H) == (2^N, 2^N)
    @test H ≈ H'
    @test tr(H) ≈ 0.0  atol=1e-12

    H_chaotic = xyz_hamiltonian(N; c=1.0, bc=:pbc)     # chaotic: c≠0, spectrum should differ
    @test H_chaotic ≈ H_chaotic'
    @test !(H ≈ H_chaotic)                    # must actually be a different matrix

    H_obc = xyz_hamiltonian(N; bc=:obc)
    @test nnz(H_obc) >= nnz(H)
end

@testset "build_system XYZ" begin
    H, O = build_system(Dict("model"=>"xyz", "N"=>4, "precision_build" => 1024))
    @test size(H) == (16,16)
    @test H ≈ H'
    @test frobenius_norm(Matrix(O)) ≈ 1.0  atol=1e-12
    @test precision(real(first(nonzeros(H)))) == 1024
    @test precision(real(first(nonzeros(O)))) == 1024
end


# ----------------
# test GOE


@testset "GOE Matrix" begin
    N = 6
    D = 2^N
    rng = MersenneTwister(0)
    sparsity=1.0
    H = GOE_matrix(N, rng, sparsity=sparsity)
    @test size(H) == (D,D)
    @test H ≈ transpose(H)                  # Hermitian
    @test eltype(H) <: Real
    @test mean(nonzeros(H)) ≈ 0 atol=1e-1 # 0 average
    @test std(nonzeros(H)) > 0.0         # there is variance
    @test var(diag(H)) ≈ 2/D atol=1e-1 # statisitc, can be that this test fails
    nondiag = [H[i,j] for i in 1:D for j in 1:D if i != j]
    @test var(nondiag) ≈ 1/D atol=1e-1
    lambdas = eigvals(Hermitian(Matrix(H)))
    @test maximum(abs, lambdas) < 2.1 # bounded eigenvalues in Wigner semicircle
end

@testset "GUE Matrix" begin
    N = 6
    D = 2^N
    rng = MersenneTwister(0)
    H = GUE_matrix(N, rng)
    @test size(H) == (2^N,2^N)
    @test H ≈ H'               # Hermitian
    @test eltype(H) <: Complex
    @test mean(nonzeros(H)) ≈ 0 atol=1e-1 # 0 average
    @test std(nonzeros(H)) > 0.0         # there is variance
    @test var(diag(H)) ≈ 2/D atol=1e-1 # statisitc, can be that this test fails
    nondiag = [H[i,j] for i in 1:D for j in 1:D if i != j]
    @test var(real(nondiag)) ≈ 1/D atol=1e-1
    @test var(imag(nondiag)) ≈ 1/D atol=1e-1
    lambdas = eigvals(Hermitian(Matrix(H)))
    @test maximum(abs, lambdas) < 3.0 # bounded eigenvalues in Wigner semicircle
end


@testset "Buidling Random Matrix" begin
    N = 4
    D = 2^N
    cfg = Dict("model" => "goe", "N" => N, "precision_build" => 1024)
    H, O = build_system(cfg)
    @test size(H) == (D,D)
    @test size(O) == (D,D)
    @test H ≈ H'                     # Hermitian
    @test O ≈ O'
    @test frobenius_norm(Matrix(O)) ≈ 1.0 atol=1e-12
    @test precision(real(first(nonzeros(H)))) == 1024
    @test precision(real(first(nonzeros(O)))) == 1024

    cfg = Dict("model" => "gue", "N" => N, "precision_build" => 1024)
    H, O = build_system(cfg)
    @test size(H) == (D,D)
    @test size(O) == (D,D)
    @test H ≈ H'                     # Hermitian
    @test O ≈ O'
    @test frobenius_norm(Matrix(O)) ≈ 1.0 atol=1e-12
    @test precision(real(first(nonzeros(H)))) == 1024
    @test precision(real(first(nonzeros(O)))) == 1024

    cfg = Dict("model" => "gue", "N" => N, "sparsity" => 0.5, "precision_build" => 1024)
    H, O = build_system(cfg)
    @test nnz(H) < D^2
end

@testset "XXZ_helper" begin
    @test _reverse_bits(128,8) == 1
    @test _reverse_bits(1,8) == 128
    @test _reverse_bits(10,4) == 5
    @test _reverse_bits(5,4) == 10
    
    basis = _block_magnetization(10,5)
    @test length(basis) == binomial(10,5)
    Vp, Vm = _parity_basis(10, basis, Float64)
    @test size(Vp, 2) == binomial(10, 5) ÷ 2  # = 126

    basis = _block_magnetization(12,3)
    @test length(basis) == binomial(12,3)

    Vp, Vm = _parity_basis(12, basis, Float64)
    @test size(Vp,1) == length(basis)
    @test size(Vm,1) == length(basis)
    @test Vp' * Vp ≈ I
    @test Vm' * Vm ≈ I
    @test norm(Vp' * Vm) < 1e-12
end


@testset "XXZ Hamiltonian" begin
    N = 10
    D = 2^N
    H = xxz_hamiltonian(N; J=1.0, Jzz=0.47, d=1.0, c=0.0, magnetization=5, parity=1, bc=:obc)
    @test size(H) == (126,126)              # Sec 4.2.1
    @test H ≈ H'                            # Hermitian
end


@testset "XXZ Operator" begin
    N = 10
    D = 2^N
    O = xxz_operator(N; oper_site=5, magnetization=5, parity=1)
    @test size(O) == (126,126)              # Sec 4.2.1
    @test O ≈ O'                            # Hermitian
end

@testset "Buidling XXZ System" begin
    N = 10
    D = 126
    cfg = Dict("model" => "xxz", "N" => N, "precision_build" => 1024,
            "J" => 1.0, "Jzz" => 0.47, "d" => 1.0, "c" => 0.0, "magnetization" => 5, "parity" => 1, "bc" => "obc", "oper_site" => 5)
    H, O = build_system(cfg)
    @test size(H) == (D,D)
    @test size(O) == (D,D)
    @test H ≈ H'                     # Hermitian
    @test O ≈ O'
    @test frobenius_norm(Matrix(O)) ≈ 1.0 atol=1e-12
    @test precision(real(first(nonzeros(H)))) == 1024
    @test precision(real(first(nonzeros(O)))) == 1024
    @test isfinite(tr(H))
end

@testset "cSYK helper" begin
    @test _csyk_bit(8,4) == 1
    @test _csyk_bit(255-8,4) == 0

    @test _csyk_flip(9,4) == 1
    @test _csyk_flip(1,4) == 9

    @test _csyk_parity_left(15,2) == 1
    @test _csyk_parity_left(15,3) == 0
    @test _csyk_parity_left(15,4) == 1
    @test _csyk_parity_left(13,4) == 0
    
    @test _csyk_annihilate(15,3) == (11,1)
    @test _csyk_annihilate(15,4) == (7,-1)
    @test _csyk_annihilate(13,2) == (nothing, 0)

    @test _csyk_create(13,2) == (15, -1)
    @test _csyk_create(15,5) == (31, 1)
    @test _csyk_create(15,1) == (nothing, 0)

    @test length(_csyk_basis(8,4)) == binomial(8, 4)
end

@testset "cSYK Hamiltonian" begin
    N = 8
    D = 70
    H = csyk_hamiltonian(8; Q=4, seed=0,T=Float64)
    @test H ≈ H'
    @test size(H) == (D,D)
end

@testset "cSYK Operator" begin
    N = 8
    D = 70
    O = csyk_operator(8; site1=7, site2=8, T=Float64)
    @test O ≈ O'
    @test size(O) == (D,D)
end

@testset "Building cSYK System" begin
    N = 8
    D = 70
    cfg = Dict("model" => "csyk", "N" => N, "precision_build" => 1024, "J" => 1.0, "seed" => 0)
    H, O = build_system(cfg)
    @test size(H) == (D,D)
    @test size(O) == (D,D)
    @test H ≈ H'                     # Hermitian
    @test O ≈ O'
    @test frobenius_norm(Matrix(O)) ≈ 1.0 atol=1e-12
    @test precision(real(first(nonzeros(H)))) == 1024
    @test precision(real(first(nonzeros(O)))) == 1024
    @test isfinite(tr(H))
    @test precision(real(first(nonzeros(H)))) == 1024
    @test precision(real(first(nonzeros(O)))) == 1024
end

end  # Operators