@testset "Krylov complexity" begin
    H, O = build_system(Dict("model"=>"ising", "N"=>4,
                            "J"=>1.0, "hx"=>1.05, "hz"=>0.5,
                            "oper"=>"Sx", "oper_site"=>1,
                            "precision_build"=>256))        
    b, Kdim, _ = lanczos(H, O; precision=:f64, reortho=:fro)
    t, K, S = krylov_complexity_entropy(b; time_max=100.0, dt_factor=0.01)

    @testset "K_C: Initial conditions K(0)=S(0)=0" begin
        @test t[1] == 0.0
        @test K[1] == 0.0
        @test S[1] == 0.0
    end

    @testset "Physical bounds" begin
        @test all(K .>= 0.0)
        K_max_theoretical = (length(b) + 1) / 2
        @test all(K .<= K_max_theoretical + 0.1)   # small tolerance for overshoot near plateau
        @test all(S .>= 0.0)
    end

    @testset "Plateau ≈ (Kdim-1)/2 — cSYK N=6" begin
        # SYK as maximally chaotic system
        H, O = build_system(Dict("model" => "csyk", "N" => 6, 
                        "precision_build" => 1024, "J" => 1.0))      
        b, Kdim, _ = lanczos(H, O; precision=:f64, reortho=:fro)
        t, K, S = krylov_complexity_entropy(b; time_max=1000.0, dt_factor=1.0e-3)
        l = Int(floor(size(K)[1]/4))
        K_sat = mean(K[l:end])          # average after exponential growth
        K_expected = (Kdim - 1) / 2
        @test K_sat ≈ K_expected  rtol=0.250  # 25% tolerance on saturation value (small system)
    end
end