@testset "Consistency — XXZ" begin

@testset "XXZ N=8 Difference with reference code" begin
    b_ref = JLD2.load("data/xxz_n8.jld2")["b"]

    H, O = build_system(Dict("model" => "xxz", "N" => 8,
            "precision_build" => 1024, "J" => 1.0,
            "Jzz" => 1.81, "d" => 0.5, "c" => 0.0,
            "magnetization" => 5, "parity" => 1,
            "bc" => "obc", "oper_site" => 4))

    b, _, _ = lanczos(H, O; precision=:f64, reortho=:fro)
    n = min(length(b), length(b_ref))
    @test n == length(b_ref)
    @test b[1:n]/5 ≈ b_ref[1:n]  rtol=1e-1
end

end  # Consistency — XXZ

