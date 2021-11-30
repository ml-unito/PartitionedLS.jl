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
* `P`: \$M × K\$ matrix specifying how to partition the \$M\$ attributes into
\$K\$ subsets. \$P_{m,k}\$ should be 1 if attribute number \$m\$ belongs to
partition \$k\$.
* `η`: regularization factor, higher values implies more regularized solutions
* `N`: number of alternating loops to be performed, defaults to 20.
* `get_solver`: a function returning the solver to be used. Defaults to () -> ECOSSolver()
* `resume` and `checkpoint` allows for restarting an optimization from a given checkpoint. 

# Result

A tuple of the form: `(opt, a, b, t, P)`

* `opt`: optimal value of the objective function (loss + regularization)
* `a`: values of the α variables at the optimal point
* `b`: values of the β variables at the optimal point
* `t`: the intercept at the optimal point
* `P`: the partition matrix (copied from the input)

The output model predicts points using the formula: f(X) = \$X * (P .* a) * b + t\$.

"""
function fit(::Type{Alt}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}, N::Int;
    η = 1.0, get_solver = get_ECOSSolver, checkpoint = (data) -> Nothing, resume = (init) -> init, fake_run = false)

    if fake_run
        return ()
    end

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

    i_start, α.value, β.value, t.value, _ = resume(initvals)

    for i = (i_start+1):N
        fix!(β)
        Convex.solve!(p, get_solver())
        free!(β)

        @debug "optval (β fixed)" p.optval α.value β.value

        fix!(α)
        Convex.solve!(p, get_solver())
        free!(α)

        @debug "optval (α fixed)" p.optval α.value β.value
        checkpoint((i, α.value, β.value, t.value, p.optval))
    end

    (p.optval, α.value, β.value, t.value, P)
end

#
# AltNNLS
#

function fit(::Type{AltNNLS}, X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; η = 0.0, N = 20, get_solver = get_ECOSSolver,
    checkpoint = (data) -> Nothing, resume = (init) -> init, fake_run = false)
    if fake_run
        return ()
    end

    if η != 0.0
        @warn "PartitionedLS (Alt): fit called with NNLS option and η != 0. Assuming η==0"
    end

    # Rewriting the problem in homogenous coordinates
    Xo = hcat(X, ones(size(X, 1), 1))
    Po = vcat(hcat(P, zeros(size(P, 1))), vec1(size(P, 2) + 1))

    M, K = size(Po)

    α = rand(Float32, M)
    β = (rand(Float32, K) .- 0.5) .* 10
    t = rand(Float32, 1)
    initvals = (0, α, β, t, Inf64)
    loss = (a, b) -> norm(Xo * (Po .* a) * b - y, 2)

    i_start, α, β, t, optval = resume(initvals)

    for i = (i_start+1):N
        # nnls problem with fixed beta variables

        Poβ = sum(Po .* β', dims = 2)
        Xoβ = Xo .* Poβ'
        α = nonneg_lsq(Xoβ, y, alg = :nnls)
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
        optval = loss(α, β)
        @debug "optval (α fixed)  $optval"

        checkpoint((i, α[1:end-1], β[1:end-1], β[end] * α[end], optval))
    end

    result = (optval, α[1:end-1], β[1:end-1], β[end] * α[end], P)
    @debug result

    result
end