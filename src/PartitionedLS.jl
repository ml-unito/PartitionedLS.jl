module PartitionedLS

export fit, predict, PartLS, PartLSFitResult, Opt, Alt, BnB, regularizingMatrix

import Base.size
using LinearAlgebra
using NonNegLeastSquares
using DocStringExtensions
using MLJModelInterface
using MLJBase
using Tables

import MLJModelInterface.fit
import MLJModelInterface.fitted_params
import MLJModelInterface.predict


"""
    $(TYPEDEF)

The PartLSFitResult struct represents the solution of the partitioned least squares problem. 
  It contains the values of the α and β variables, the intercept t and the partition matrix P.

## Fields
$(TYPEDFIELDS)
"""
struct PartLSFitResult
  """
  The values of the α variables. For each partition ``k``, it holds the values of the α variables
  are such that ``\\sum_{i \\in P_k} \\alpha_{k} = 1``.
  """
  α::Vector{Float64}
  """
  The values of the β variables. For each partition ``k``, ``\\beta_k`` is the coefficient that multiplies the
  features in the k-th partition.
  """
  β::Vector{Float64}
  """
  The intercept term of the model.
  """
  t::Float64
  """
  The partition matrix. It is a binary matrix where each row corresponds to a partition and each column
  corresponds to a feature. The element ``P_{k, i} = 1`` if feature ``i`` belongs to partition ``k``.
  """
  P::Matrix{Int}
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
    $(TYPEDSIGNATURES)

## Result
the prediction for the partitioned least squares problem with solution α, β, t over the dataset X and partition matrix P
"""
function predict(α::Vector{Float64}, β::Vector{Float64}, t::Float64, P::Matrix{Int}, X::Matrix{Float64})
  X * (P .* α) * β .+ t
end


"""
    $(TYPEDSIGNATURES)


Make predictions for the datataset `X` using the PartialLS model `model`.

## Arguments
  - `model`: a [PartLSFitResult](@ref)
  - `X`: the data containing the examples for which the predictions are sought
  
## Return
 the predictions of the given model on examples in X. 
"""
function predict(model::PartLSFitResult, X::Array{Float64,2})
  (; α, β, t, P) = model
  predict(α, β, t, P, X)
end


include("PartitionedLSAlt.jl")
include("PartitionedLSOpt.jl")
include("PartitionedLSBnB.jl")


MLJModelInterface.@mlj_model mutable struct PartLS <: MLJModelInterface.Deterministic
  Optimizer::Union{Type{Opt},Type{Alt},Type{BnB}} = Opt
  P::Matrix{Int} = 0.5::(all(_[i, j] == 0 || _[i, j] == 1 for i in range(1, size(_, 1)) for j in range(1, size(_, 2))))
  η::Float64 = 0.0::(_ >= 0)
end

function MLJModelInterface.fit(m::PartLS, verbosity, X, y)
  X = MLJBase.matrix(X)
  return PartitionedLS.fit(m.Optimizer, X, y, m.P, η=m.η)
end

function MLJModelInterface.fitted_params(model::PartLS, fitresult)
  return fitresult
end

function MLJModelInterface.predict(model::PartLS, fitresult, X)
  X = MLJBase.matrix(X)
  return PartitionedLS.predict(fitresult, X)
end

end
