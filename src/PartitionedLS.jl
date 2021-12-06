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


# # Returns a vector of length n with all components set to zero except
# for the last one, which is set to 1.

# E.g., vec1(5) -> [0 0 0 0 1]
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
# predict(α::Vector{Float64}, β::Vector{Float64}, t::Float64, P::Matrix{Int}, X::Matrix{Float64})::Vector{Float64}

## Result
the prediction for the partitioned least squares problem with solution α, β, t over the dataset X and partition matrix P
"""
function predict(α::Vector{Float64}, β::Vector{Float64}, t::Float64, P::Matrix{Int}, X::Matrix{Float64})::Vector{Float64}
  X * (P .* α) * β .+ t
end


"""
# predict(model::Tuple, X::Array{Float64,2})

Make predictions for the datataset `X` using the PartialLS model `model`.

## Arguments
  - `model` is a Tuple in the form returned by fit functions, it shall contains the following elements:
    - `opt`: the optimal value of the objective attained by the fit function
    - `α`: the values of the α variables
    - `β`: the values of the β variables
    - `t`: the value of the t variable
    - `P`: the partition matrix
  - `X`: the data containing the examples for which the predictions are sought
  
## Return
 the predictions of the given model on examples in X. 
"""
function predict(model, X::Array{Float64,2})
  (_, α, β, t, P) = model
  predict(α, β, t, P, X)
end


include("PartitionedLSAlt.jl")
include("PartitionedLSOpt.jl")
include("PartitionedLSBnB.jl")

end
