@testset "Config and IO" begin

    @testset "load_config — Dict passthrough" begin
        cfg = Dict("model" => "ising", "N" => 4)
        @test load_config(cfg) === cfg
    end

    @testset "load_config — YAML file" begin
        mktempdir() do dir
            path = joinpath(dir, "test.yaml")
            write(path, """
            model: ising
            N: 4
            hx: 1.05
            hz: 0.5
            """)
            cfg = load_config(path)
            @test cfg["model"] == "ising"
            @test cfg["N"]     == 4
            @test cfg["hx"]    == 1.05
        end
    end

    result = simulate(Dict("model" => "ising", "N" => 4,
                           "J" => 1.0, "hx" => 1.05, "hz" => 0.5,
                           "precision" => 200, "precision_build" => 1024,
                           "reortho" => "fro", "time_max" => 100.0,
                           "dt_factor" => 0.1))

    @testset "simulate — Ising N=4" begin
        @test result isa KCResult
        @test result.Kdim > 0
        @test result.K[1] ≈ 0.0
        @test result.S[1] ≈ 0.0
        @test maximum(result.K) > 0.0
        @test result.wall_time > 0.0
    end

    @testset "simulate — kc_out_resolution" begin
        r = simulate(Dict("model" => "ising", "N" => 4, "hz" => 0.5,
                          "kc_out_resolution" => 50))
        @test length(r.t) == 51
        @test length(r.K) == 51
    end

    @testset "simulate — remove_degeneracies kwarg passthrough" begin
        r_rd  = simulate(Dict("model" => "ising", "N" => 4, "hz" => 0.5,
                              "remove_degeneracies" => true))
        r_nrd = simulate(Dict("model" => "ising", "N" => 4, "hz" => 0.5,
                              "remove_degeneracies" => false))
        @test length(r_nrd.b) >= length(r_rd.b)
    end

    @testset "simulate — save: true writes result.jld2" begin
        mktempdir() do dir
            simulate(Dict("model" => "ising", "N" => 4, "hz" => 0.5,
                          "save" => true, "output_dir" => dir))
            @test isfile(joinpath(dir, "result.jld2"))
        end
    end

    @testset "simulate — save_b: true writes b.jld2" begin
        mktempdir() do dir
            simulate(Dict("model" => "ising", "N" => 4, "hz" => 0.5,
                          "save_b" => true, "output_dir" => dir))
            @test isfile(joinpath(dir, "b.jld2"))
            @test !isfile(joinpath(dir, "result.jld2"))
        end
    end

    @testset "simulate — save: true takes priority over save_b" begin
        mktempdir() do dir
            simulate(Dict("model" => "ising", "N" => 4, "hz" => 0.5,
                          "save" => true, "save_b" => true, "output_dir" => dir))
            @test isfile(joinpath(dir, "result.jld2"))
            @test !isfile(joinpath(dir, "b.jld2"))
        end
    end

    @testset "save_b / load_b — Float64 round-trip" begin
        b = [1.5, 2.3, 3.1, 2.8, 1.9]
        mktempdir() do dir
            path = joinpath(dir, "b.jld2")
            save_b(b, path)
            b2 = load_b(path)
            @test b2 == b
            @test eltype(b2) == Float64
        end
    end

    @testset "save_b / load_b — BigFloat round-trip" begin
        b = setprecision(BigFloat, 200) do; BigFloat[1, 2, 3]; end
        mktempdir() do dir
            path = joinpath(dir, "b_big.jld2")
            save_b(b, path)
            b2 = load_b(path)
            @test b2 == b
            @test eltype(b2) == BigFloat
            @test precision(b2[1]) == 200
        end
    end

    @testset "save_b / load_b — Double64 round-trip" begin
        b = Double64[1.5, 2.3, 3.1]
        mktempdir() do dir
            path = joinpath(dir, "b_d64.jld2")
            save_b(b, path)
            b2 = load_b(path)
            @test b2 == b
            @test eltype(b2) == Double64
        end
    end

    @testset "save_result / load_result — fields preserved" begin
        mktempdir() do dir
            path = joinpath(dir, "result.jld2")
            save_result(result, path)
            r2 = load_result(path)
            @test r2.K    == result.K
            @test r2.S    == result.S
            @test r2.Kdim == result.Kdim
            @test r2.b    == result.b
            @test precision(r2.b[1]) == 200
        end
    end

end
