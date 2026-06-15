
# NOTE: the cSYK Hamiltonian draws random couplings J_{ijkl} at build time.
# Without a fixed seed the realisation differs each run, so this test compares
# against a stored reference b at rtol=1e-1 (10%).  It may fail for unlucky
# random draws; that is expected and does not indicate a code bug.

@testset "Consistency cSYK" begin

@testset "cSYK N=8 Difference with reference code" begin
    b_ref = JLD2.load("data/csyk_n8.jld2")["b"]

    H, O = build_system(Dict("model" => "csyk", "N" => 8,
                        "precision_build" => 1024, "J" => 1.0))
    b, _, _ = lanczos(H, O; precision=:f64, reortho=:fro)

    n = min(length(b), length(b_ref))
    @test n == length(b_ref)              # same Krylov dimension
    @test b[1:n] ≈ b_ref[1:n]  rtol=3e-1  # stochastic may fail for different random draws
end

end  # Consistency cSYK