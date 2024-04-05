struct BnB end         # Branch and Bound approach

"""
    $(TYPEDSIGNATURES)

Implements the Branch and Bound algorithm to fit a Partitioned Least Squres model.

## Arguments

* `X`: \$N × M\$ matrix describing the examples
* `y`: \$N\$ vector with the output values for each example
* `P`: \$M × K\$ matrix specifying how to partition the \$M\$ attributes into \$K\$ subsets. \$P_{m,k}\$ should be 1 if attribute number \$m\$ belongs to
partition \$k\$.
* `η`: regularization factor, higher values implies more regularized solutions (default: 0.0)
* nnlsalg: the kind of nnls algorithm to be used during solving. Possible values are :pivot, :nnls, :fnnls (default: :nnls)

## Result

A tuple with the following fields:


1. a `PartLSFitResult` object containing the fitted model
2. a `nothing` object
3. a NamedTuple with fields: 
    - `opt` containing the optimal value of the objective function
    - `nopen` containing the number of open nodes in the branch and bound tree

"""
function fit(::Type{BnB}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; η = 0.0, nnlsalg = :nnls)
    Xo, Po = homogeneousCoords(X, P)
    Xo, yo = regularizeProblem(Xo, y, Po, η)
    Σ::Array{Int,1} = []

    opt, α, nopen = fit_BnB(Xo, yo, Po, Inf, Σ, nnlsalg = nnlsalg)
    β = sum(Po .* α, dims = 1)
    α = sum(Po .* α ./ β, dims = 2)

    return (PartLSFitResult(α[1:end-1], β[1:end-1], β[end], P), nothing, (; opt=opt, nopen=nopen))
end

function sum_max_0_αi_αj(P::Array{Int,2}, α::Array{Float64,1})
    K = size(P, 2)   # number of partitions
    result = zeros(K)

    # forall k, sets result[k] = sum(max(0, -αi*αj)) forall i,j in partition k
    for k = 1:K
        is = findall(!=(0), P[:, k])   # list of indices of partition k
        for i = 1:length(is)
            for j = (i+1):length(is)
                result[k] += max(0, -α[is[i]]α[is[j]])
            end
        end
    end

    return result
end

# using CSV
# using DataFrames
# function saveproblem(XX, y)
#     df = DataFrame(XX, :auto)
#     df.y = y

#     @warn "Writing temporary csv file -- this is a debug feature, should not be present in production"
#     CSV.write("nnls_problem.csv", df)
# end

function lower_bound(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}, Σ::Array{Int,1}, nnlsalg)
    posConstr = Σ[findall(>(0), Σ)]
    negConstr = -Σ[findall(<(0), Σ)]

    M = size(X, 2)
    Xp = copy(X)
    Xm = -copy(X)
    Xp[:, negConstr] .= 0
    Xm[:, posConstr] .= 0

    XX = [Xp Xm]

    @debug "Launching nonneg_lsq alg=$nnlsalg"
    αα = nonneg_lsq(XX, y, alg = nnlsalg)
    @debug "nonneg_lsq terminated"
    αp = αα[1:M]
    αn = αα[M+1:end]
    αp[negConstr] .= 0
    αn[posConstr] .= 0

    α = αp - αn

    return norm(XX * αα - y), α
end

function fit_BnB(X::Array{Float64,2}, y::Array{Float64,1}, P::Matrix{Int}, μ::Float64, Σ::Array{Int,1};
    depth = 0,
    nnlsalg = :pivot)::Tuple{Float64,Vector{Float64},Int}
    @debug "BnB new node"

    lb, α = lower_bound(X, y, P, Σ, nnlsalg)
    @debug "Lower bound: $lb"

    if lb >= μ
        # no solution can be found in this branch
        return (Inf64, [], 1)
    end

    ν = sum_max_0_αi_αj(P, α)

    if all(==(0), ν)
        # optimal solution found
        p★ = norm(X * α - y)

        @debug "Optimal solution for this node $p★"
        return (p★, α, 1)
    end

    k = argmax(ν)
    pk = findall(==(1), P[:, k])

    Σp = [Σ; pk]     # positive index i stands for αi >= 0
    Σm = [Σ; -pk]    # negative index i stands for αi <= 0

    μp, αp, nopenp = fit_BnB(X, y, P, μ, Σp, depth = depth + 1, nnlsalg = nnlsalg)
    μm, αm, nopenm = fit_BnB(X, y, P, min(μ, μp), Σm, depth = depth + 1, nnlsalg = nnlsalg)

    i = argmin([μ, μp, μm])

    res = ([μ, μp, μm][i], [α, αp, αm][i], (nopenp + nopenm + 1))

    @debug "Best upperbound for this node: $(res[1]) num open nodes so far: $(res[3])"
    return res
end