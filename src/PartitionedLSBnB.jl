struct BnB end         # Branch and Bound approach

function fit(::Type{BnB}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; η = 1.0, get_solver = get_ECOSSolver, checkpoint = (data) -> Nothing, resume = (init) -> init, fake_run = false)
    Xo, Po = homogeneousCoords(X, P)
    Σ::Array{Int,1} = []

    opt, α = fit_BnB(Xo, y, Po, Inf, Σ, get_solver = get_solver)
    β = sum(Po .* α, dims = 1)
    α = sum(Po .* α ./ β, dims = 2)

    return opt, α[1:end-1], β[1:end-1], β[end], P
end

function sum_max_0_αi_αj(P::Array{Int,2}, α::Array{Float64,1})
    K = size(P,2)   # number of partitions
    result = zeros(K)

    # forall k, sets result[k] = sum(max(0, -αi*αj)) forall i,j in partition k
    for k in 1:K
        is = findall(!=(0), P[:, k])   # list of indices of partition k
        for i in 1:length(is)
            for j in (i+1):length(is)
                result[k] += max(0, -α[is[i]]α[is[j]])
            end
        end
    end

    return result
end



function lower_bound(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}, Σ::Array{Int,1}, get_solver::typeof(get_ECOSSolver), nnlsalg)
    posConstr = Σ[findall(>(0), Σ)]
    negConstr = -Σ[findall(<(0), Σ)]

    M = size(X, 2)
    Xp = copy(X)
    Xm = -copy(X)
    Xp[:, negConstr] .= 0
    Xm[:, posConstr] .= 0

    XX = [Xp Xm]

    @debug "Launching nonneg_lsq"
    αα = nonneg_lsq(XX, y, alg=nnlsalg)
    @debug "nonneg_lsq terminated"
    αp = αα[1:M]
    αn = αα[M+1:end]
    αp[negConstr] .= 0
    αn[posConstr] .= 0

    α = αp - αn

    return norm(XX * αα - y), α
end

function fit_BnB(X::Array{Float64,2}, y::Array{Float64,1}, P::Matrix{Int}, μ::Float64, Σ::Array{Int,1}; 
                 get_solver = get_ECOSSolver, 
                 depth = 0,
                 nnlsalg = :pivot)
    @debug "BnB new node"

    lb, α = lower_bound(X, y, P, Σ, get_solver, nnlsalg)
    @debug "Lower bound: $lb"

    if lb >= μ
        # no solution can be found in this branch
        return Inf64, []
    end

    ν = sum_max_0_αi_αj(P, α)

    if all(==(0), ν)
        # optimal solution found
        p★ = norm(X * α - y), α
        
        @debug "Optimal solution for this node $p★"
        return p★
    end

    k = argmax(ν)
    pk = findall(==(1), P[:, k])

    Σp = [Σ; pk]     # positive index i stands for αi >= 0
    Σm = [Σ; -pk]    # negative index i stands for αi <= 0

    μp, αp = fit_BnB(X, y, P, μ, Σp, get_solver = get_solver, depth = depth + 1)
    μm, αm = fit_BnB(X, y, P, min(μ, μp), Σm, get_solver = get_solver, depth = depth + 1)

    i = argmin([μ, μp, μm])

    res = [μ, μp, μm][i], [α, αp, αm][i]

    @debug "Best upperbound for this node: $(res[1])"
    return res
end