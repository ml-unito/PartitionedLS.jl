module TLLR

using Convex
using ECOS

import Base.size
export fit, fit_iterative, fit_iterative_slow, predict
using LinearAlgebra

"""
  indextobeta(b::Integer, K::Integer)::Array{Int64,1}

  returns 2 * bin(b,K) - 1

  where bin(b,K) is a vector of K elements containing the binary
  representation of b.
"""
function indextobeta(b::Integer, K::Integer)
  result::Array{Int64,1} = []
  for k = 1:K
    push!(result, 2(b % 2)-1)
    b >>= 1
  end

  result
end

"""
    fit(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; beta=randomvalues)

Fits a TLLRegression model to the given data and resturns the
learnt model (see the Result section).

# Arguments

* `X`: \$N × M\$ matrix describing the examples
* `y`: \$N\$ vector with the output values for each example
* `P`: \$M × K\$ matrix specifying how to partition the \$M\$ attributes into
    \$K\$ subsets. \$P_{m,k}\$ should be 1 if attribute number \$m\$ belongs to
    partition \$k\$.
* `verbose`: if true (or 1) the output of solver will be shown
* `η`: regularization factor, higher values implies more regularized solutions

# Result

A tuple of the form: `(opt, a, b, t, P)`

* `opt`: optimal value of the objective function (loss + regularization)
* `a`: values of the α variables at the optimal point
* `b`: values of the β variables at the optimal point
* `t`: the intercept at the optimal point
* `P`: the partition matrix (copied from the input)

The output model predicts points using the formula: f(X) = \$X * (P .* a) * b + t\$.

"""
function fit(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; verbose=0, η=1.0)
  # row normalization
  M,K = size(P)

  results = []

  for b in 0:(2^K-1)
    α = Variable(M, Positive())
    t = Variable()
    β = indextobeta(b,K)

    # loss = sumsquares(X * (P .* (α * ones(1,K))) * β + t - y) + η * sumsquares(α)
    loss = sumsquares(X * (P .* (α * ones(1,K))) * β + t - y) + η * (sumsquares(P' * α) + t^2)

    p = minimize(loss)
    Convex.solve!(p, ECOSSolver(verbose=verbose))

    @debug "iteration" b "optval:" p.optval
    push!(results,(p.optval, α.value, β, t.value, P))
  end

  optindex = argmin((z -> z[1]).(results))
  opt,a,b,t,_ = results[optindex]


  A = sum(P .* a, dims=1)
  a = sum((P .* a) ./ A, dims=2)
  b = b .* A'

  (opt, a, b, t, P)
end


"""
    fit_iterative(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; beta=randomvalues)

Fits a block least squares model by alternating the optimization of the α and β variables.

# Arguments

* `X`: \$N × M\$ matrix describing the examples
* `y`: \$N\$ vector with the output values for each example
* `P`: \$M × K\$ matrix specifying how to partition the \$M\$ attributes into
    \$K\$ subsets. \$P_{m,k}\$ should be 1 if attribute number \$m\$ belongs to
    partition \$k\$.
* `verbose`: if true (or 1) the output of solver will be shown
* `η`: regularization factor, higher values implies more regularized solutions

# Result

A tuple of the form: `(opt, a, b, t, P)`

* `opt`: optimal value of the objective function (loss + regularization)
* `a`: values of the α variables at the optimal point
* `b`: values of the β variables at the optimal point
* `t`: the intercept at the optimal point
* `P`: the partition matrix (copied from the input)

The output model predicts points using the formula: f(X) = \$X * (P .* a) * b + t\$.

"""
function fit_iterative(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; verbose=0, η=1, N=20)
  M,K = size(P)

  α = Variable(M, Positive())
  β = Variable(K)
  t = Variable()
  constraints =  P' * α == ones(K)

  loss = sumsquares(X * (P .* (α * ones(1,K))) * β + t - y) + η * (sumsquares(β) + t^2)
  p = minimize(loss, constraints)

  α.value = rand(Float32, M)
  β.value = (rand(Float32, K) .- 0.5) .* 20

  for i in 1:N
    fix!(β)
    Convex.solve!(p, ECOSSolver(verbose=verbose, warmstart=(i == 1 ? true : false)))
    free!(β)

    @debug "optval (β fixed)" p.optval  α.value  β.value

    fix!(α)
    Convex.solve!(p, ECOSSolver(verbose=verbose, warmstart=true))
    free!(α)

    @debug "optval (α fixed)"  p.optval α.value β.value
  end

  # # row normalization
  # M,K = size(P)

  # results = []

  # for b in 0:(2^K-1)
  #   α = Variable(M, Positive())
  #   t = Variable()
  #   β = indextobeta(b,K)

  #   loss = sumsquares(X * (P .* (α * ones(1,K))) * β + t - y)
  #   regularization = η * norm(α,2)
  #   p = minimize(loss + regularization)
  #   Convex.solve!(p, ECOSSolver(verbose=verbose))

  #   @debug "iteration" b "optval:" p.optval
  #   push!(results,(p.optval, α.value, β, t.value, P))
  # end

  # optindex = argmin((z -> z[1]).(results))
  # opt,a,b,t,_ = results[optindex]


  # A = sum(P .* a, dims=1)
  # a = sum((P .* a) ./ A, dims=2)
  # b = b .* A'

  (p.optval, α.value, β.value, t.value, P)
end



function fit_iterative_slow(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; verbose=0, η=1.0, N=20)
  M,K = size(P)

  α = Variable(M, Positive())
  β = Variable(K)
  t = Variable()
  constraints =  P' * α == ones(K)

  α.value = rand(Float32, M)
  β.value = (rand(Float32, K) .- 0.5) .* 20
  a = α.value
  b = β.value
  optval = 100000

  for i in 1:N
    α.value = a
    loss = sumsquares(X * (P .* (α * ones(1,K))) * b + t - y) + η * (norm(b,2)^2 + t^2)
    p = minimize(loss, constraints)
    solve!(p, ECOSSolver(verbose=verbose))
    a = α.value

    @debug "with b fixed | a: $(α.value) b: $b" p.optval

    β.value = b
    loss = sumsquares(X * (P .* (a * ones(1,K))) * β + t - y) + η * (sumsquares(β) + t^2)
    p = minimize(loss, constraints)
    solve!(p, ECOSSolver(verbose=verbose))
    b = β.value

    @debug "with a fixed | a: $a b: $(β.value)" p.optval

    optval = p.optval
  end

  (optval, a, b, t.value, P)
end

"""
  predict(model::Tuple, X::Array{Float64,2})

  returns the predictions of the given model on examples in X

  #see

    fit(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; beta=randomvalues)
"""
function predict(model, X::Array{Float64,2})
  (_, α, β, t, P) = model
  X * (P .* α) * β .+ t
end

end
