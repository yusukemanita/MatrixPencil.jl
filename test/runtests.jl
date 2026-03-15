using MatrixPencil
using Test
using LinearAlgebra
using Statistics
using Random

@testset "MatrixPencil.jl" begin
    include("test_mpm.jl")
    include("test_stabilization.jl")
    include("test_clustering.jl")
    include("test_t0scan.jl")
end
