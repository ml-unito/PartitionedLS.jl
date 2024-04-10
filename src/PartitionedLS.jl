module PartitionedLS

export fit, predict, PartLS, PartLSFitResult, Opt, Alt, BnB, regularizeProblem, homogeneousCoords
import MLJModelInterface

import Base.size
using LinearAlgebra
using NonNegLeastSquares
using DocStringExtensions
using Tables
using Random

import MLJModelInterface.fit
import MLJModelInterface.fitted_params
import MLJModelInterface.predict

const MMI = MLJModelInterface

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
  α::Vector{AbstractFloat}
  """
  The values of the β variables. For each partition ``k``, ``\\beta_k`` is the coefficient that multiplies the
  features in the k-th partition.
  """
  β::Vector{AbstractFloat}
  """
  The intercept term of the model.
  """
  t::(AbstractFloat)
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
  - `X`: any matrix or table with `Continuous` element scitype. 
         Check column scitypes of a table `X` with `schema(X)`. 
  - `P`: the partition matrix

## Return
  - `Xo`: the homogeneous version of X
  - `Po`: the homogeneous version of P
"""
function homogeneousCoords(X::Matrix{T}, P::Matrix{Int})::Tuple{Matrix{T}, Matrix{Int}} where T
  Xo = hcat(X, ones(size(X, 1), 1))
  Po::Matrix{Int} = vcat(hcat(P, zeros(size(P, 1))), vec1(size(P, 2) + 1))

  Xo, Po
end

"""
Adds regularization terms to the problem. The regularization terms are added to the
objective function as a sum of squares of the α variables. The regularization
parameter η controls the strength of the regularization.

## Arguments
  - `X`: any matrix or table with `Continuous` element scitype. 
         Check column scitypes of a table `X` with `schema(X)`.
  - `y`: any vector with `Continuous` element scitype. Check scitype with `scitype(y)`. 
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
function regularizeProblem(X::Matrix{T}, y::AbstractVector{T}, P::Matrix{Int}, η::AbstractFloat)::Tuple{Matrix{T}, Vector{T}} where T
  if η == 0
    return X, y
  end

  Xn = X
  yn = y
  for k in 1:size(P, 2)
    v = P[:, k] .== 1
    v = reshape(convert(Vector{<:AbstractFloat}, v), 1, :)
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
function predict(α::AbstractVector{<:AbstractFloat}, β::AbstractVector{<:AbstractFloat}, t::AbstractFloat, P::Matrix{Int}, X::Matrix{<:AbstractFloat})
  X * (P .* α) * β .+ t
end


"""
    $(TYPEDSIGNATURES)


Make predictions for the datataset `X` using the PartialLS model `model`.

## Arguments
  - `model`: a [PartLSFitResult](@ref)
  - `X`: any matrix or table with `Continuous` element scitype containing the 
         examples for which the predictions are sought. Check column scitypes 
         of a table `X` with `schema(X)`.

## Return
 the predictions of the given model on examples in X.
"""
function predict(model::PartLSFitResult, X::Array{<:AbstractFloat,2})
  (; α, β, t, P) = model
  predict(α, β, t, P, X)
end


include("PartitionedLSAlt.jl")
include("PartitionedLSOpt.jl")
include("PartitionedLSBnB.jl")

"""
    PartLS

A model type for fitting a partitioned least squares model to data. Both an MLJ and native
interfacew are provided.

# MLJ Interface

From MLJ, the type can be imported using

PartLS = @load PartLS pkg=PartitionedLS

Construct an instance with default hyper-parameters using the syntax `model =
PartLS()`. Provide keyword arguments to override hyper-parameter defaults, as in
`model = PartLS(P=...)`.


## Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where

  - `X`: any matrix or table with `Continuous` element scitype. 
         Check column scitypes of a table `X` with `schema(X)`.
         
Train the machine using `fit!(mach)`.

## Hyper-parameters

- `Optimizer`: the optimization algorithm to use. It can be `Opt`, `Alt` or `BnB` (names
  exported by `PartitionedLS.jl`).

- `P`: the partition matrix. It is a binary matrix where each row corresponds to a
  partition and each column corresponds to a feature. The element `P_{k, i} = 1` if
  feature `i` belongs to partition `k`.

- `η`: the regularization parameter. It controls the strength of the regularization.

- `ϵ`: the tolerance parameter. It is used to determine when the Alt optimization
  algorithm has converged. Only used by the `Alt` algorithm.

- `T`: the maximum number of iterations. It is used to determine when to stop the Alt
  optimization algorithm has converged. Only used by the `Alt` algorithm.

- `rng`: the random number generator to use.

  - If `nothing`, the global random number generator `rand` is used.

  - If an integer, the global number generator `rand` is used after seeding it with the
    given integer.

  - If an object of type `AbstractRNG`, the given random number generator is used.

## Operations

- `predict(mach, Xnew)`: return the predictions of the model on new data `Xnew`


## Fitted parameters

The fields of `fitted_params(mach)` are:

- `α`: the values of the α variables. For each partition `k`, it holds the values of the α variables
  are such that ``\\sum_{i \\in P_k} \\alpha_{k} = 1``.
- `β`: the values of the β variables. For each partition `k`, `β_k` is the coefficient that multiplies the features in the k-th partition.
- `t`: the intercept term of the model.
- `P`: the partition matrix. It is a binary matrix where each row corresponds to a partition and each column
  corresponds to a feature. The element `P_{k, i} = 1` if feature `i` belongs to partition `k`.

## Examples

```julia
PartLS = @load PartLS pkg=PartitionedLS

X = [[1. 2. 3.];
     [3. 3. 4.];
     [8. 1. 3.];
     [5. 3. 1.]]

y = [1.;
     1.;
     2.;
     3.]

P = [[1 0];
     [1 0];
     [0 1]]


model = PartLS(P=P)
mach = machine(model, X, y) |> fit!

# predictions on the training set:
predict(mach, X)

```

# Native Interface

```
using PartitionedLS

X = [[1. 2. 3.];
     [3. 3. 4.];
     [8. 1. 3.];
     [5. 3. 1.]]

y = [1.;
     1.;
     2.;
     3.]

P = [[1 0];
     [1 0];
     [0 1]]


# fit using the optimal algorithm
result = fit(Opt, X, y, P, η = 0.0)
y_hat = predict(result.model, X)
```

For other `fit` keyword options, refer to the "Hyper-parameters" section for the MLJ
interface.

"""
MMI.@mlj_model mutable struct PartLS <: MMI.Deterministic
  Optimizer::Union{Type{Opt},Type{Alt},Type{BnB}} = Opt
  P::Matrix{Int} = Array{Int}(undef, 0,0)::(all(_[i, j] == 0 || _[i, j] == 1 for i in range(1, size(_, 1)) for j in range(1, size(_, 2))))
  η::AbstractFloat = 0.0::(_ >= 0)
  ϵ::AbstractFloat = 1e-6::(_ > 0)
  T::Int = 100::(_ > 0)
  rng::Union{Nothing,Int,AbstractRNG} = nothing
end


"""
    $(TYPEDSIGNATURES)

Fits a PartitionedLS Regression model to the given data and resturns the learnt model (see the Result section).
It conforms to the MLJ interface.

## Arguments
- `m`: A [`PartLS`](@ref) model to fit
- `verbosity`: the verbosity level
- `X`: any matrix or table with `Continuous` element scitype. 
       Check column scitypes of a table `X` with `schema(X)`.
- `y`: any vector with `Continuous` element scitype. Check scitype with `scitype(y)`. 

"""
function MMI.fit(m::PartLS, verbosity, X, y)
  X = MMI.matrix(X)
  y = vec(y)
  P = m.P

  if size(P, 1) == 0
    @warn "P is empty, using a single partition: this corresponds to resort to a standard least squares problem and is likely not what you want."
    P = ones(Int, size(X, 2), 1)
  end

  if m.Optimizer == Alt
    return PartitionedLS.fit(Alt, X, y, P, η=m.η, ϵ=m.ϵ, T=m.T, rng=m.rng)
  end

  return PartitionedLS.fit(m.Optimizer, X, y, P, η=m.η)
end

function MMI.fitted_params(model::PartLS, fitresult)
  return fitresult
end

"""
    $(TYPEDSIGNATURES)

Make predictions for the datataset `X` using the PartitionedLS model `model`.
It conforms to the MLJ interface.
"""
function MMI.predict(model::PartLS, fitresult, X)
  X = MMI.matrix(X)
  return PartitionedLS.predict(fitresult, X)
end

MMI.metadata_pkg.(PartLS,
    name = "PartitionedLS",
    uuid = "19f41c5e-8610-11e9-2f2a-0d67e7c5027f", # see your Project.toml
    url  = "https://github.com/ml-unito/PartitionedLS.jl.git",  # URL to your package repo
    julia = true,          # is it written entirely in Julia?
    license = "MIT",       # your package license
    is_wrapper = false,    # does it wrap around some other package?
)

# Then for each model,
MMI.metadata_model(PartLS,
    input_scitype   = Union{MMI.Table{AbstractVector{MMI.Continuous}}, AbstractMatrix{MMI.Continuous}},  # what input data is supported?
    target_scitype  = AbstractVector{MMI.Continuous},           # for a supervised model, what target?
    supports_weights = false,                                                  # does the model support sample weights?
        load_path    = "PartitionedLS.PartLS"
    )
end
