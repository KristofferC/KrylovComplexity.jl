import KrylovComplexity: frobenius_inner

@testset "Consistency — Mathematica" begin

@testset "Ising Integrable N=4 — Mathematica" begin
    b_ref = parse.(Float64, split(
        readlines(joinpath(@__DIR__, "data/integrable_ising_n4_p300_hx105_hz0/b_coefficients.csv"))[1], ","))

    H, O = build_system(Dict("model"=>"ising", "N"=>4,
                             "J"=>1.0, "hx"=>1.05, "hz"=>0.0,
                             "oper"=>"Sx", "oper_site"=>1,
                             "precision_build"=>256))
    b, _, _ = lanczos(H, O; precision=256, reortho=:fro)

    n = min(length(b), length(b_ref))
    @test n == length(b_ref)              # same Krylov dimension
    @test b[1:n] ≈ b_ref[1:n]  rtol=1e-12
end


@testset "Ising Chaotic N=4 — Mathematica" begin
    b_ref = parse.(Float64, split(
        readlines(joinpath(@__DIR__, "data/chaotic_ising_n4_p1024_hx105_hz05/b_coefficients.csv"))[1], ","))

    H, O = build_system(Dict("model"=>"ising", "N"=>4,
                             "J"=>1.0, "hx"=>1.05, "hz"=>0.5,
                             "oper"=>"Sx", "oper_site"=>1,
                             "precision_build"=>256))
    b, _, _ = lanczos(H, O; precision=256, reortho=:fro)

    n = min(length(b), length(b_ref))
    @test n == length(b_ref)              # same Krylov dimension
    @test b[1:n] ≈ b_ref[1:n]  rtol=1e-2
end

# -----

@testset "XYZ Integrable N=4 — Mathematica" begin
    b_ref = parse.(Float64, split(
        readlines(joinpath(@__DIR__, "data/integrable_xyz_n4_p300_Jx-035_Jy05_Jz-01_hz00/b_coefficients.csv"))[1], ","))

    H, O = build_system(Dict("model"=>"xyz", "N"=>4,
                             "Jx"=>-0.35, "Jy"=>0.5, "Jz"=>-0.1,
                             "hz"=>0.0, "c" => 0.0, "oper"=>"Sx", 
                             "oper_site"=>1, "precision_build"=>256))
    b, _, _ = lanczos(H, O; precision=256, reortho=:fro)

    n = min(length(b), length(b_ref))
    @test n == length(b_ref)              # same Krylov dimension
    @test b[1:n] ≈ b_ref[1:n]  rtol=1e-12
end


@testset "XYZ Chaotic N=4 — Mathematica" begin
    b_ref = parse.(Float64, split(
        readlines(joinpath(@__DIR__, "data/chaotic_xyz_n4_p300_Jx-035_Jy05_Jz-01_hz00/b_coefficients.csv"))[1], ","))

    H, O = build_system(Dict("model"=>"xyz", "N"=>4,
                             "Jx"=>-0.35, "Jy"=>0.5, "Jz"=>-0.1,
                             "hz"=>-0.8, "c" => 0.0, "oper"=>"Sx", 
                             "oper_site"=>1, "precision_build"=>256))
    b, _, _ = lanczos(H, O; precision=256, reortho=:fro)

    n = min(length(b), length(b_ref))
    @test n == length(b_ref)              # same Krylov dimension
    @test b[1:n] ≈ b_ref[1:n]  rtol=1e-2
end

@testset "b[1] = norm of commutator" begin
    H, O = build_system(Dict("model"=>"ising","N"=>4,"hz"=>0.5,"precision_build"=>256))
    b, _, _ = lanczos(H, O; precision=256, reortho=:fro)
    # b[1]^2 = <[H,O]|[H,O]> / <O|O> in Hilbert-Schmidt inner product
    comm = Matrix(H)*Matrix(O) - Matrix(O)*Matrix(H)
    b1_expected = sqrt(frobenius_inner(comm, comm) / frobenius_inner(Matrix(O), Matrix(O)))
    @test b[1] ≈ Float64(b1_expected)  rtol=1e-8
end

end  # Consistency — Mathematica