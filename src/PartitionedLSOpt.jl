struct Opt end         # Optimal algorithm
struct OptNNLS end     # Optimal algorithm using Non Negative Least Squares

"""
indextobeta(b::Integer, K::Integer)::Array{Int64,1}

returns 2 * bin(b,K) - 1

where bin(b,K) is a vector of K elements containing the binary
representation of b.
"""
function indextobeta(b::Integer, K::Integer)
    result::Array{Int64,1} = []
    for k = 1:K
        push!(result, 2(b % 2) - 1)
        b >>= 1
    end

    result
end

"""
Returns the matrix obtained multiplying each element in X to the associated
weight in β. 
"""
function bmatrix(X, P, β)
    Pβ = P .* β'
    featuremul = sum(Pβ, dims = 2)
    X .* featuremul'
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
function fit(::Type{Opt}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; η = 1.0, get_solver = get_ECOSSolver, checkpoint = data -> Nothing, resume = init -> init, fake_run = false)
    if fake_run
        return ()
    end

    @debug "Regularization parameter set: Opt algorithm fitting using standard convex solver"

    # row normalization
    M, K = size(P)

    b_start, results = resume((-1, []))

    for b = (b_start+1):(2^K-1)
        @debug "Starting iteration $b/$(2^K-1)"
        α = Variable(M, Positive())
        t = Variable()
        β = indextobeta(b, K)

        loss = sumsquares(X * (P .* (α * ones(1, K))) * β + t - y) + η * (sumsquares(P' * α) + t * t)

        p = minimize(loss)
        Convex.solve!(p, get_solver())

        @debug "iteration $b optval:" p.optval
        push!(results, (p.optval, α.value, β, t.value, P))

        checkpoint((b, results))
    end

    optindex = argmin((z -> z[1]).(results))
    opt, a, b, t, _ = results[optindex]


    A = sum(P .* a, dims = 1)
    a = sum((P .* a) ./ A, dims = 2)
    b = b .* A'

    (opt, a, b, t, P)
end


#
#  OptNNLS
#


function fit(::Type{OptNNLS}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; η = 0.0, get_solver = get_ECOSSolver, checkpoint = data -> Nothing, resume = init -> init, fake_run = false)
    if fake_run
        return ()
    end

    if η != 0.0
        @warn "PartitionedLS (Opt): fit called with NNLS option and η != 0. Assuming η==0"
    end

    @debug "Opt algorithm fitting  using non negative least square algorithm"

    # Rewriting the problem in homogenous coordinates
    Xo, Po = homogeneousCoords(X, P)
    M, K = size(Po)

    b_start, results = resume((-1, []))

    for b = (b_start+1):(2^K-1)
        @debug "Starting iteration $b/$(2^K-1)"
        β = indextobeta(b, K)
        Xb = bmatrix(Xo, Po, β)
        α = nonneg_lsq(Xb, y, alg = :nnls)
        optval = norm(Xo * (Po .* α) * β - y)



        result = (optval, α[1:(end-1)], β[1:(end-1)], β[end] * α[end], P)
        push!(results, result)

        checkpoint((b, results))
    end

    optindex = argmin((z -> z[1]).(results))
    opt, a, b, t, _ = results[optindex]

    A = sum(P .* a, dims = 1)
    b = b .* A'

    A[A.==0.0] .= 1.0 # substituting all 0.0 with 1.0
    a = sum((P .* a) ./ A, dims = 2)

    (opt, a, b, t, P)
end