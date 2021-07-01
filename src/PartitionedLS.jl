module PartitionedLS

using Convex
export fit, fit_alternating_slow, predict, Opt, Alt, OptNNLS, AltNNLS

import Base.size
using LinearAlgebra
using ECOS
using NonNegLeastSquares

struct Opt end         # Optimal algorithm
struct Alt end         # Alternate Optimization approach
struct OptNNLS end     # Optimal algorithm using Non Negative Least Squares
struct AltNNLS end     # Alternate Optimization using Non Negative Least Squares
struct BnB end         # Brench and Bound approach

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

function get_ECOSSolver()
  ECOS.Optimizer(verbose=0)
end

function vec1(n)
  result = zeros(1,n)
  result[n] = 1
  result
end

"""
Returns the matrix obtained multiplying each element in X to the associated
weight in β. 
"""
function bmatrix(X, P, β)
  Pβ = P .* β'
  featuremul = sum(Pβ, dims=2)
  X .* featuremul'
end

function fit(::Type{OptNNLS}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; η = 0.0, get_solver = get_ECOSSolver, checkpoint = data -> Nothing, resume = init -> init, fake_run = false)
  if fake_run
    return ()
  end

  if η != 0.0
    @warn "PartitionedLS (Opt): fit called with NNLS option and η != 0. Assuming η==0"
  end

  @debug "Opt algorithm fitting  using non negative least square algorithm"
  
  # Rewriting the problem in homogenous coordinates
  Xo = hcat(X, ones(size(X,1),1))
  Po = vcat( hcat(P, zeros(size(P,1))), vec1(size(P,2)+1))
  
  M,K = size(Po)
  
  b_start, results = resume((-1, []))
  
  for b in (b_start+1):(2^K-1)
    @debug "Starting iteration $b/$(2^K-1)"
    β = indextobeta(b,K)
    Xb = bmatrix(Xo, Po, β)
    α = nonneg_lsq(Xb, y, alg=:nnls)
    optval = norm(Xo * (Po .* α) * β - y)

    if any(==(0), sum(Po .* α, dims=1))
      @warn "found group containing all zeros:" α = α β = β sumα = sum(Po .* α, dims=1)
    end
    
    result = (optval, α[1:(end-1)], β[1:(end-1)], β[end] * α[end], P)
    push!(results,result)
    
    checkpoint((b, results))
  end
  
  optindex = argmin((z -> z[1]).(results))
  opt,a,b,t,_ = results[optindex]
    
  A = sum(P .* a, dims=1)
  b = b .* A'

  A[A.==0.0] .= 1.0 # substituting all 0.0 with 1.0
  a = sum((P .* a) ./ A, dims=2)

  (opt, a, b, t, P)
end


"""
fit(::Type{Opt}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; beta=randomvalues)

Fits a PartialLS Regression model to the given data and resturns the
learnt model (see the Result section).

# Arguments

* `X`: \$N × M\$ matrix describing the examples
* `y`: \$N\$ vector with the output values for each example
* `P`: \$M × K\$ matrix specifying how to partition the \$M\$ attributes into
\$K\$ subsets. \$P_{m,k}\$ should be 1 if attribute number \$m\$ belongs to
partition \$k\$.
* `η`: regularization factor, higher values implies more regularized solutions
* `get_solver`: a function returning the solver to be used. Defaults to () -> ECOSSolver()

# Result

A tuple of the form: `(opt, a, b, t, P)`

* `opt`: optimal value of the objective function (loss + regularization)
* `a`: values of the α variables at the optimal point
* `b`: values of the β variables at the optimal point
* `t`: the intercept at the optimal point
* `P`: the partition matrix (copied from the input)

The output model predicts points using the formula: f(X) = \$X * (P .* a) * b + t\$.

"""
function fit(::Type{Opt}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; η = 1.0, get_solver = get_ECOSSolver, checkpoint = data -> Nothing, resume = init -> init, fake_run = false )
  if fake_run
    return ()
  end

  @debug "Regularization parameter set: Opt algorithm fitting using standard convex solver"
  
  # row normalization
  M,K = size(P)
  
  b_start, results = resume((-1, []))
  
  for b in (b_start+1):(2^K-1)
    @debug "Starting iteration $b/$(2^K-1)"
    α = Variable(M, Positive())
    t = Variable()
    β = indextobeta(b,K)
    
    loss = sumsquares(X * (P .* (α * ones(1,K))) * β + t - y) + η * (sumsquares(P' * α) + t*t)
    
    p = minimize(loss)
    Convex.solve!(p, get_solver())
    
    @debug "iteration $b optval:" p.optval
    push!(results,(p.optval, α.value, β, t.value, P))
    
    checkpoint((b, results))
  end
  
  optindex = argmin((z -> z[1]).(results))
  opt,a,b,t,_ = results[optindex]
  
  
  A = sum(P .* a, dims=1)
  a = sum((P .* a) ./ A, dims=2)
  b = b .* A'
  
  (opt, a, b, t, P)
end

function checkalpha(a, P)
  suma = sum(P .* a, dims=1)
  sumP = sum(P, dims=1)

  for k in 1:size(P,2)
    if suma[k] == 0.0
      for m in 1:size(P,1)
        if P[m,k] == 1
          a[m] = 1.0 / sumP[k]
        end
      end
    end
  end

  return a
end


function fit(::Type{AltNNLS}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; η = 0.0, N=20, get_solver = get_ECOSSolver,
  checkpoint = (data) -> Nothing, resume = (init) -> init, fake_run = false)
  if fake_run
    return ()
  end

  if η != 0.0
    @warn "PartitionedLS (Alt): fit called with NNLS option and η != 0. Assuming η==0"
  end

  # Rewriting the problem in homogenous coordinates
  Xo = hcat(X, ones(size(X,1),1))
  Po = vcat( hcat(P, zeros(size(P,1))), vec1(size(P,2)+1))
  
  M,K = size(Po)
    
  α = rand(Float32, M)
  β = (rand(Float32, K) .- 0.5) .* 10
  t = rand(Float32, 1)
  initvals = (0, α, β, t, Inf64)
  loss = (a, b) -> norm(Xo * (Po .* a) * b - y,2)
  
  i_start, α, β, t, optval = resume(initvals)
  
  for i in (i_start+1):N
    # nnls problem with fixed beta variables

    Poβ = sum(Po .* β', dims=2)
    Xoβ = Xo .* Poβ'
    α = nonneg_lsq(Xoβ, y, alg=:nnls)
    α = checkalpha(α, Po)

    sumα = sum(Po .* α, dims=1)
    Poα = sum(Po .* sumα, dims=2)
    α = α ./ Poα
    β = β .* sumα'


    if any(isnan, α)
      @warn "found α containing NaN values: $α iteration: $i β: $β"
    end

    @debug "optval (β fixed): $(loss(α, β))"

    # ls problem with fixed alpha variables

    Xoα = Xo * (Po .* α)
    β = Xoα \ y
    optval = loss(α, β)
    @debug "optval (α fixed)  $optval"
    
    checkpoint((i, α[1:end-1], β[1:end-1], β[end] * α[end], optval))
  end
  
  result = (optval, α[1:end-1], β[1:end-1], β[end] * α[end], P)
  @debug result

  result
end


"""
fit(::Type{Alt}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; beta=randomvalues)

Fits a PartitionedLS model by alternating the optimization of the α and β variables.

# Arguments

* `X`: \$N × M\$ matrix describing the examples
* `y`: \$N\$ vector with the output values for each example
* `P`: \$M × K\$ matrix specifying how to partition the \$M\$ attributes into
\$K\$ subsets. \$P_{m,k}\$ should be 1 if attribute number \$m\$ belongs to
partition \$k\$.
* `η`: regularization factor, higher values implies more regularized solutions
* `N`: number of alternating loops to be performed, defaults to 20.
* `get_solver`: a function returning the solver to be used. Defaults to () -> ECOSSolver()

# Result

A tuple of the form: `(opt, a, b, t, P)`

* `opt`: optimal value of the objective function (loss + regularization)
* `a`: values of the α variables at the optimal point
* `b`: values of the β variables at the optimal point
* `t`: the intercept at the optimal point
* `P`: the partition matrix (copied from the input)

The output model predicts points using the formula: f(X) = \$X * (P .* a) * b + t\$.

"""
function fit(::Type{Alt}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; 
  η=1.0, get_solver = get_ECOSSolver,  checkpoint = (data) -> Nothing, resume = (init) -> init, fake_run = false)

  if fake_run
    return ()
  end
  
  M,K = size(P)
  
  α = Variable(M, Positive())
  β = Variable(K)
  t = Variable()
  constraints =  P' * α == ones(K)
  
  loss = sumsquares(X * (P .* (α * ones(1,K))) * β + t - y) + η * (sumsquares(β) + t*t)
  p = minimize(loss, constraints)
  
  α.value = rand(Float32, M)
  β.value = (rand(Float32, K) .- 0.5) .* 10
  t.value = rand(Float32, 1)
  initvals = (0, α.value, β.value, t.value, p.optval)
  
  i_start, α.value, β.value, t.value, _ = resume(initvals)
  
  for i in (i_start+1):N
    fix!(β)
    Convex.solve!(p, get_solver())
    free!(β)
    
    @debug "optval (β fixed)" p.optval  α.value  β.value
    
    fix!(α)
    Convex.solve!(p, get_solver())
    free!(α)
    
    @debug "optval (α fixed)"  p.optval α.value β.value
    checkpoint((i, α.value, β.value, t.value, p.optval))
  end
  
  (p.optval, α.value, β.value, t.value, P)
end

function fit( ::Type{BnB}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; η=1.0, get_solver = get_ECOSSolver,  checkpoint = (data) -> Nothing, resume = (init) -> init, fake_run = false) 
  Xo = hcat(X, ones(size(X,1),1))
  Po = vcat( hcat(P, zeros(size(P,1))), vec1(size(P,2)+1))
  
  Q = Xo'Xo
  q = -2Xo'y
  q0 = y'y

  opt,α = fit_BnB(Q, q, q0, Po, Inf, get_solver = get_solver)
  β = sum(Po .* α, dims=1)
  α = sum(Po .* α ./ β, dims=2)

  return opt, α[1:end-1], β[1:end-1], β[end], P
end

function sum_max_0_αi_αj(P::Array{Float64,2}, α::Array{Float64,1})
  K = size(P)[2]   # number of partitions
  result = zeros(K)

  # forall k, sets result[k] = sum(max(0, -αi*αj)) forall i,j in partition k
  for k in 1:K
    is = findall(!=(0), P[:,k])   # list of indices of partition k
    for i in size(is)
      for j in (i+1):size(is)
        result[k] += max(0, -α[is[i]]α[is[j]])
      end
    end
  end

  return result
end

function lower_bound(Q::Array{Float64,2}, q::Array{Float64,1}, q0, P::Array{Int,2}, Σ::Array{Int}; get_solver = get_ECOSSolver) {
  M = size(Q)[1]
  α = Variable(M)
  constraints = []
  for σ in Σ
    if σ > 0
      push!(constraints, α[σ] >= 0)
    else 
      push!(constraints, α[σ] <= 0)
    end
  end

  loss = α' * Q * α + q' * α + q0
  p = minimize(loss, constraints)

  Convex.solve!(p, get_solver())

  return p.optval, α.value
}


function fit_BnB(Q::Array{Float64,2}, q::Array{Float64,1}, q0, P::Array{Int,2}, μ::Float64; get_solver = get_ECOSSolver)

  lb,α = lower_bound(Q,q,q_0,Σ, get_solver)

  if lb >= μ
    # no solution can be found in this branch
    return Inf64, []
  end

  ν = sum_max_0_αi_αj(P, α)
  if ν == zeros(len(ν))
    # optimal solution found
    return ν, α
  end

  k = argmax(ν)
  pk = findall(==(1), P[:,k])
  Σp = [Σ; pk]     # positive index i stands for αi >= 0
  Σm = [Σ; -pk]    # negative index i stands for αi <= 0

  μp,αp = fit_BnB(Q, q, q0, P, μ, Σp, get_solver = get_solver)
  μm,αm = fit_Bnb(Q, q, q0, P, min(μ, μp), Σm, get_solver = get_solver)

  i = argmin(μ, μp, μm)
  return [μ,μp,μm][i], [α, αp, αm][i]
end

# This function implements the same algorithm as fit_alternating, but it works around a bug 
# in the CVX library that prevented fix! and free! to work as intended. The current version
# of the library fixes the bug, so this function should not be called (just use fix(::Alt)).

function fit_alternating_slow(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; verbose=0, η=1.0, N=20)
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
    loss = sumsquares(X * (P .* (α * ones(1,K))) * b + t - y) + η * (norm(b,2)^2 + t*t)
    p = minimize(loss, constraints)
    solve!(p, ECOSSolver(verbose=verbose))
    a = α.value
    
    @debug "with b fixed | a: $(α.value) b: $b" p.optval
    
    β.value = b
    loss = sumsquares(X * (P .* (a * ones(1,K))) * β + t - y) + η * (sumsquares(β) + t*t)
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
