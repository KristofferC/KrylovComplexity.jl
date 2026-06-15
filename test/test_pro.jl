@testset "PRO and optimization test" begin
    @testset "PRO reorthogonalization" begin
        H, O = build_system(Dict("model" => "ising", "N" => 4, "hz" => 0.5))

        b_fro,  n_fro, oc_fro = lanczos(H, O; precision=:f64, reortho=:fro)
        b_pro,  n_pro, oc_pro = lanczos(H, O; precision=:f64, reortho=:pro)
        b_dpro, n_dpro, oc_dpro = lanczos(H, O; precision=:f64, reortho=:dpro)

        @test n_fro == n_pro
        @test n_fro == n_dpro

        @test oc_fro >= oc_pro
        @test oc_dpro == 2*oc_pro

        @test length(b_pro)  == length(b_fro)
        @test length(b_dpro) == length(b_fro)
        @test b_pro  ≈ b_fro  atol=1e-8
        @test b_dpro ≈ b_fro  atol=1e-8
    end


    @testset "PRO threading" begin
        H, O = build_system(Dict("model" => "ising", "N" => 4, "hz" => 0.5))

        b_fro, _, _ = lanczos(H, O; precision=:f64, reortho=:fro)
        b_pro, _, _ = lanczos(H, O; precision=:f64, reortho=:pro)

        @info "Running PRO threading test" nthreads=Threads.nthreads()
        @test b_pro ≈ b_fro atol=1e-8

        if Threads.nthreads() == 1
            @warn "Thread count is 1 — run with julia --threads N to exercise the threaded W recurrence"
        end
    end
end