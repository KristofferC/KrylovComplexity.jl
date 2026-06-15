using KrylovComplexity
using Test
using LinearAlgebra
using SparseArrays
using Random
using DoubleFloats
using GenericLinearAlgebra
using Statistics
using YAML
using JLD2
@testset "KrylovComplexity" begin
  include("test_operators.jl")
  include("test_lanczos.jl")
  include("test_pro.jl")
  include("test_consistency_mathematica.jl")
  include("test_consistency_csyk.jl")
  include("test_consistency_xxz.jl")
  include("test_complexity.jl")
  include("test_config_io.jl")
end