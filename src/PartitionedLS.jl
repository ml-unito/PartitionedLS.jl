module PartitionedLS

using Convex
export fit, predict, Opt, Alt, OptNNLS, AltNNLS, BnB, regularizingMatrix

import Base.size
using LinearAlgebra
using ECOS
using NonNegLeastSquares

function get_ECOSSolver()
  ECOS.Optimizer(verbose = 0)
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

  Xo, Po
end

function regularizeProblem(X, y, P, η)
  if η == 0
    return X, y
  end
  
  Xn = X
  yn = y
  for k in 1:size(P, 2)
    v = P[:, k] .== 1
    v = reshape(convert(Vector{Float64}, v), 1, :)
    Xn = vcat(Xn, sqrt(η) .* v)
    yn = vcat(yn, zeros(1))
  end

  return Xn, yn
end

"""
  Incorporates in the datamatrix X additional rows to force a regularizing term into
  an unregularized objective.

  Assume that the objective has the form of |X w - y |^2, the idea is to add additional
  size(w) rows to X (an y) such that:

  - X_{N+j,j} = √μ
  - X_{N+j,j'} = 0   if j≂̸j'
  - y_{N+j} = 0
"""
function regularizingMatrix(X, y, μ)
  lenw = size(X, 2)

  newRows = I(lenw) .* sqrt(μ)
  newX = vcat(X, newRows)
  newY = vcat(y, zeros(lenw))

  return newX, newY
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
