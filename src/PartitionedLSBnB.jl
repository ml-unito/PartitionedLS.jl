struct BnB end         # Branch and Bound approach

function fit(::Type{BnB}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; η = 1.0, get_solver = get_ECOSSolver, checkpoint = (data) -> Nothing, resume = (init) -> init, fake_run = false)
    Xo = hcat(X, ones(size(X, 1), 1))
    Po::Array{Int,2} = vcat(hcat(P, zeros(size(P, 1))), vec1(size(P, 2) + 1))

    Q = Xo'Xo
    q = -2Xo'y
    q0 = y'y
    Σ::Array{Int,1} = []

    opt, α = fit_BnB(Q, q, q0, Po, Inf, Σ, get_solver = get_solver)
    β = sum(Po .* α, dims = 1)
    α = sum(Po .* α ./ β, dims = 2)

    return opt, α[1:end-1], β[1:end-1], β[end], P
end

function sum_max_0_αi_αj(P::Array{Float64,2}, α::Array{Float64,1})
    K = size(P)[2]   # number of partitions
    result = zeros(K)

    # forall k, sets result[k] = sum(max(0, -αi*αj)) forall i,j in partition k
    for k = 1:K
        is = findall(!=(0), P[:, k])   # list of indices of partition k
        for i in size(is)
            for j = (i+1):size(is)
                result[k] += max(0, -α[is[i]]α[is[j]])
            end
        end
    end

    return result
end

function lower_bound(Q::Array{Float64,2}, q::Array{Float64,1}, q0::Float64, P::Array{Int,2}, Σ::Array{Int,1}, get_solver::typeof(get_ECOSSolver))
    M = size(Q)[1]
    α = Variable(M)
    constraints::Array{Convex.Constraint} = []
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
end


function fit_BnB(Q::Array{Float64,2}, q::Array{Float64,1}, q0, P::Array{Int,2}, μ::Float64, Σ::Array{Int,1}; get_solver = get_ECOSSolver)

    lb, α = lower_bound(Q, q, q0, P, Σ, get_solver)

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
    pk = findall(==(1), P[:, k])
    Σp = [Σ; pk]     # positive index i stands for αi >= 0
    Σm = [Σ; -pk]    # negative index i stands for αi <= 0

    μp, αp = fit_BnB(Q, q, q0, P, μ, Σp, get_solver = get_solver)
    μm, αm = fit_Bnb(Q, q, q0, P, min(μ, μp), Σm, get_solver = get_solver)

    i = argmin(μ, μp, μm)
    return [μ, μp, μm][i], [α, αp, αm][i]
end