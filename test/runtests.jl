using PartitionedLS
using Test
using MLJBase

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

@testset "Testing PartitionedLS" verbose=true begin

     @testset "Standard interface" verbose=true begin
          for alg in [Opt, Alt, BnB]
               @testset "Testing $alg" begin
                    if alg == Alt
                         result = fit(alg, X, y, P, η=0.0, ϵ=1e-6, T=100, rgn=123)
                    else
                         result = fit(alg, X, y, P, η=0.0)
                    end
                    
                    result = fit(alg, X, y, P, η=0.0)
                    opt = result[3].opt
                    y_pred = predict(result[1], X)

                    @test opt ≈ 0.0 atol=1e-6
                    @test sum(y_pred - y)^2 ≈ 0.0 atol=1e-6
               end
          end
     end

     @testset "MLJ interface" verbose = true  begin
          for alg in [Opt, Alt, BnB]
               @testset "Testing $alg" begin
                    model = PartLS(P=P, Optimizer=alg, rgn=123)
                    mach = machine(model, X, y)
                    fit!(mach, verbosity=0)

                    opt = report(mach).opt
                    y_pred = predict(mach, X)

                    @test opt ≈ 0.0 atol = 1e-6
                    @test sum(y_pred - y)^2 ≈ 0.0 atol = 1e-6
               end
          end
     end

     @testset "MLJ interface with Table data" begin 
          for alg in [Opt, BnB]
               @testset "Testing $alg" begin
                    X, y = make_regression(1000, 10, rng=123)
                    P = [[ones(5); zeros(5)] [zeros(5); ones(5)]]

                    model = PartLS(P=P, Optimizer=alg, rgn=123)
                    mach = machine(model, X, y)
                    fit!(mach, verbosity=0)

                    opt = report(mach).opt
                    y_pred = predict(mach, X)

                    @test opt ≈ 36.840 atol = 1e-3
                    @test sum(y_pred - y)^2 ≈ 0.0 atol = 1e-6
               end
          end

          X, y = make_regression(1000, 10, rng=123)
          P = [[ones(5); zeros(5)] [zeros(5); ones(5)]]

          model = PartLS(P=P, Optimizer=Alt)
          mach = machine(model, X, y)
          fit!(mach, verbosity=0)

          opt = report(mach).opt
          y_pred = predict(mach, X)

          @test true
     end

     @testset "Alt algorithm determinism" begin
          X, y = make_regression(1000, 10, rng=123)
          P = [[ones(5); zeros(5)] [zeros(5); ones(5)]]

          last = nothing
          for _ in 1:10
               model = PartLS(P=P, Optimizer=Alt, ϵ=1e-3, T=100, rgn=123)
               mach = machine(model, X, y)
               fit!(mach, verbosity=0)

               opt = report(mach).opt
               y_pred = predict(mach, X)

               if last === nothing
                    last = opt
               else
                    @test opt ≈ last atol=1e-6
               end
          end
     end
end
