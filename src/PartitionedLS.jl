module PartitionedLS

using Convex
export fit, predict, Opt, Alt, OptNNLS, AltNNLS, BnB

import Base.size
using LinearAlgebra
using ECOS
using NonNegLeastSquares

function get_ECOSSolver()
  ECOS.Optimizer(verbose=0)
end

"""
Returns a vector of length n with all components set to zero except
for the last one, which is set to 1.

E.g., vec1(5) -> [0 0 0 0 1]
"""
function vec1(n)
  result = zeros(1, n)
  result[n] = 1
  result
end

function homogeneousCoords(X, P::Array{Int,2})
  Xo = hcat(X, ones(size(X, 1), 1))
  Po::Matrix{Int} = vcat(hcat(P, zeros(size(P, 1))), vec1(size(P, 2) + 1))

  Xo,Po
end


"""
predict(model::Tuple, X::Array{Float64,2})

returns the predictions of the given model on examples in X
"""
function predict(model, X::Array{Float64,2})
  (_, α, β, t, P) = model
  X * (P .* α) * β .+ t
end

include("PartitionedLSAlt.jl")
include("PartitionedLSOpt.jl")
include("PartitionedLSBnB.jl")

end
