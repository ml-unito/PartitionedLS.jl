struct Alt end         # Alternate Optimization approach
struct AltNNLS end     # Alternate Optimization using Non Negative Least Squares


function checkalpha(a, P)
    suma = sum(P .* a, dims = 1)
    sumP = sum(P, dims = 1)

    for k = 1:size(P, 2)
        if suma[k] == 0.0
            for m = 1:size(P, 1)
                if P[m, k] == 1
                    a[m] = 1.0 / sumP[k]
                end
            end
        end
    end

    return a
end


"""
fit(::Type{Alt}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; beta=randomvalues)

Fits a PartitionedLS model by alternating the optimization of the α and β variables.

# Arguments

* `X`: \$N × M\$ matrix describing the examples
* `y`: \$N\$ vector with the output values for each example
* `P`: \$M × K\$ matrix specifying how to partition the \$M\$ attributes into \$K\$ subsets. \$P_{m,k}\$ should be 1 if attribute number \$m\$ belongs to
partition \$k\$.
* `η`: regularization factor, higher values implies more regularized solutions
* `T`: number of alternating loops to be performed, defaults to 20.
* `get_solver`: a function returning the solver to be used. Defaults to () -> ECOSSolver()
* `resume` and `checkpoint` allows for restarting an optimization from a given checkpoint. 

# Result

A tuple of the form: `(opt, a, b, t, P)`

* `opt`: optimal value of the objective function (loss + regularization)
* `a`: values of the α variables at the optimal point
* `b`: values of the β variables at the optimal point
* `t`: the intercept at the optimal point
* `P`: the partition matrix (copied from the input)

"""
function fit(::Type{Alt}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; 
        T=1000, η = 1.0, ϵ = 1e-6,
        get_solver = get_ECOSSolver)
    M, K = size(P)

    α = Variable(M, Positive())
    β = Variable(K)
    t = Variable()
    constraints = P' * α == ones(K)

    loss = sumsquares(X * (P .* (α * ones(1, K))) * β + t - y) + η * (sumsquares(β) + t * t)
    p = minimize(loss, constraints)

    α.value = rand(Float32, M)
    β.value = (rand(Float32, K) .- 0.5) .* 10
    t.value = rand(Float32, 1)
    initvals = (0, α.value, β.value, t.value, p.optval)

    i_start, α.value, β.value, t.value, _ = initvals

    oldoptval = 1e20
    optval = 1e10
    i = 1

    while i <= T && abs(oldoptval - optval) > ϵ * oldoptval
        fix!(β)
        Convex.solve!(p, get_solver())
        free!(β)

        @debug "optval (β fixed)" p.optval α.value β.value

        fix!(α)
        Convex.solve!(p, get_solver())
        free!(α)

        @debug "optval (α fixed)" p.optval α.value β.value
        
        oldoptval = optval
        optval = p.optval
        i += 1
    end

    @debug "Exiting with optimality gap: $(abs(oldoptval - optval))"

    (p.optval, α.value, β.value, t.value, P)
end

#
# AltNNLS
#

function fit(::Type{AltNNLS}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2};
    η = 0.0, ϵ = 1e-6, T = 1000, get_solver = get_ECOSSolver, nnlsalg = :pivot)
    if η != 0.0
        @warn "PartitionedLS (Alt): fit called with NNLS option and η != 0. Assuming η==0"
    end

    Xo, Po = homogeneousCoords(X, P)

    M, K = size(Po)

    α = rand(Float32, M)
    β = (rand(Float32, K) .- 0.5) .* 10
    t = rand(Float32, 1)
    initvals = (0, α, β, t, Inf64)
    loss = (a, b) -> norm(Xo * (Po .* a) * b - y, 2)

    i_start, α, β, t, optval = initvals

    oldoptval = 1e20
    optval = 1e10
    i = 1

    while i <= T && abs(oldoptval - optval) > ϵ * oldoptval
        # nnls problem with fixed beta variables

        Poβ = sum(Po .* β', dims = 2)
        Xoβ = Xo .* Poβ'
        α = nonneg_lsq(Xoβ, y, alg = nnlsalg)
        α = checkalpha(α, Po)

        sumα = sum(Po .* α, dims = 1)
        Poα = sum(Po .* sumα, dims = 2)
        α = α ./ Poα
        β = β .* sumα'


        if any(isnan, α)
            @warn "found α containing NaN values: $α iteration: $i β: $β"
        end

        @debug "optval (β fixed): $(loss(α, β))"

        # ls problem with fixed alpha variables

        Xoα = Xo * (Po .* α)
        β = Xoα \ y             # idiomatic Julia way to solve the least squares problem

        oldoptval = optval
        optval = loss(α, β)
        @debug "optval (α fixed)  $optval"

        i += 1
    end

    result = (optval, α[1:end-1], β[1:end-1], β[end] * α[end], P)

    @debug "Exiting with optimality gap: $(abs(oldoptval - optval))"

    result
end