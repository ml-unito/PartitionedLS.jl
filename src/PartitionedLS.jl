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


"""
Rewrites X and P in homogeneous coordinates. The result is a tuple (Xo, Po) where Xo is the
homogeneous version of X and Po is the homogeneous version of P.

## Arguments
  - `X`: the data matrix
  - `P`: the partition matrix

## Return
  - `Xo`: the homogeneous version of X
  - `Po`: the homogeneous version of P
"""
function homogeneousCoords(X, P::Array{Int,2})
  Xo = hcat(X, ones(size(X, 1), 1))
  Po::Matrix{Int} = vcat(hcat(P, zeros(size(P, 1))), vec1(size(P, 2) + 1))

  Xo, Po
end

"""
Adds regularization terms to the problem. The regularization terms are added to the
objective function as a sum of squares of the α variables. The regularization
parameter η controls the strength of the regularization.

## Arguments
  - `X`: the data matrix
  - `y`: the target vector
  - `P`: the partition matrix
  - `η`: the regularization parameter

## Return
  - `Xn`: the new data matrix
  - `yn`: the new target vector

## Main idea
K new rows are added to the data matrix X, row ``k \\in \\{1 \\dots K\\}`` is a vector of zeros except for
the components that corresponds to features belonging to the k-th partition, which is set to sqrt(η). 
The target vector y is extended with K zeros.

The point of this change is that when the objective function is evaluated as ``math \\|Xw - y\\|^2``, the new part of
  the matrix contributes to the loss with a factor of  ``η \\sum \\|w_i\\|^2`` . This is equivalent to adding a regularization
  term to the objective function.
"""
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
