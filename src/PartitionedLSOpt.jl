struct Opt end         # Optimal algorithm

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


function cleanupResult(::Type{Opt}, result, P)
    opt, a, b, t, _ = result

    A = sum(P .* a, dims = 1)
    b = b .* A'

    A[A.==0.0] .= 1.0 # substituting all 0.0 with 1.0
    a = sum((P .* a) ./ A, dims = 2)

    return opt, PartLSFitResult(vec(a), vec(b), t, P)
end

"""
    $(TYPEDSIGNATURES)

Fits a PartialLS Regression model to the given data and resturns the learnt model (see the Result section). 
It uses a coplete enumeration strategy which is exponential in K, but guarantees to find the optimal solution.

## Arguments

* `X`: \$N × M\$ matrix describing the examples
* `y`: \$N\$ vector with the output values for each example
* `P`: \$M × K\$ matrix specifying how to partition the \$M\$ attributes into \$K\$ subsets. \$P_{m,k}\$ should be 1 if attribute number \$m\$ belongs to
partition \$k\$.
* `η`: regularization factor, higher values implies more regularized solutions (default: 0.0)
* `returnAllSolutions`: if true an additional output is appended to the resulting tuple containing all solutions found during the algorithm.
* `nnlsalg`: the kind of nnls algorithm to be used during solving. Possible values are :pivot, :nnls, :fnnls (default: :nnls)

## Example

```julia
X = rand(100, 10)
y = rand(100)
P = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 0 1 1]
result = fit(Opt, X, y, P)
```

"""
function fit(::Type{Opt}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2};
    η=0.0, nnlsalg=:nnls, returnAllSolutions=false)

    @debug "Opt algorithm fitting  using non negative least square algorithm"

    # Rewriting the problem in homogenous coordinates
    Xo, Po = homogeneousCoords(X, P)
    Xo, yo = regularizeProblem(Xo, y, Po, η)
    _, K = size(Po)

    b_start, results = -1, []

    for b = (b_start+1):(2^K-1)
        @debug "Starting iteration $b/$(2^K-1)"
        β = indextobeta(b, K)
        Xb = bmatrix(Xo, Po, β)
        α = nonneg_lsq(Xb, yo, alg = nnlsalg)
        optval = norm(Xo * (Po .* α) * β - yo)

        result = (optval, α[1:(end-1)], β[1:(end-1)], β[end] * α[end], P)
        push!(results, result)
    end

    optindex = argmin((z -> z[1]).(results))
    opt, model = cleanupResult(Opt, results[optindex], P)

    if returnAllSolutions
        return (model, nothing, (;solutions = map((r) -> cleanupResult(Opt, r, P), results)))
    else
        return (model, nothing, nothing)
    end
end

