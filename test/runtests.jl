using PartitionedLS
using Test

X = [[1.0 2.0 3.0]
     [3.0 3.0 4.0]
     [8.0 1.0 3.0]
     [5.0 3.0 1.0]]

y = [1.0
     1.0
     2.0
     3.0]

P = [[1 0]
     [1 0]
     [0 1]]

@testset "Opt" begin
     result = fit(Opt, X, y, P, η=0.0)
     opt = result.opt
     y_pred = predict(result.model, X)

     @info "Opt:", opt

    @test opt ≈ 0.0 atol=1e-6
    @test sum(y_pred - y)^2 ≈ 0.0 atol=1e-6
end

@testset "Alt" begin
    result = fit(Alt, X, y, P, η=0.0)
    opt = result.opt
    y_pred = predict(result.model, X)

     @info "Alt:", opt

    @test opt ≈ 0.0 atol=1e-6
    @test sum(y_pred - y)^2 ≈ 0.0 atol=1e-6
end

@testset "BnB" begin
     result = fit(BnB, X, y, P)
     opt = result.opt
     y_pred = predict(result.model, X)

     @info "BnB:", opt

     @test opt ≈ 0.0 atol=1e-6
     @test sum(y_pred - y)^2 ≈ 0.0 atol=1e-6
end
