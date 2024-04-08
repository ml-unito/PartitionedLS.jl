using CSV

struct Alt end         # Alternate Optimization approach

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
    $(TYPEDSIGNATURES)

Fits a PartitionedLS model by alternating the optimization of the α and β variables. This version uses 
an optimization strategy based on non-negative-least-squaes solvers. This formulation is faster and 
more numerically stable with respect to `fit(Alt, ...)``.

## Arguments

* `X`: \$N × M\$ matrix describing the examples
* `y`: \$N\$ vector with the output values for each example
* `P`: \$M × K\$ matrix specifying how to partition the \$M\$ attributes into \$K\$ subsets. \$P_{m,k}\$ should be 1 if attribute number \$m\$ belongs to partition \$k\$.
* `η`: regularization factor, higher values implies more regularized solutions. Default is 0.0.
* `T`: number of alternating loops to be performed. Default is 100.
* `ϵ`: minimum relative improvement in the objective function before stopping the optimization. Default is 1e-6
* `nnlsalg`: specific flavour of nnls algorithm to be used, possible values are `:pivot`, `:nnls`, `:fnnls`. Default is :nnls

## Result

A Tuple with the following fields:

1. a `PartLSFitResult` object containing the fitted model
2. a `nothing` object
3. a NamedTuple with a field `opt` containing the optimal value of the objective function
"""
function fit(::Type{Alt}, X::Array{Float64,2}, y::AbstractArray{Float64,1}, P::Array{Int,2};
    η = 0.0, ϵ = 1e-6, T = 100, nnlsalg = :nnls, rng = nothing)

    Xo, Po = homogeneousCoords(X, P)
    Xo, yo = regularizeProblem(Xo, y, Po, η)

    M, K = size(Po)

    if rng === nothing
        rng = rand
    elseif isa(rng, Int)
        Random.seed!(rng)
        rng = rand
    end

    α = rng(Float32, M)
    β = (rng(Float32, K) .- 0.5) .* 10

    initvals = (0, α, β, Inf64)
    loss = (a, b) -> norm(Xo * (Po .* a) * b - yo, 2)

    i_start, α, β, optval = initvals

    oldoptval = 1e20
    optval = 1e10
    i = 1

    while i <= T && abs(oldoptval - optval) > ϵ * oldoptval
        # nnls problem with fixed beta variables

        Poβ = sum(Po .* β', dims = 2)
        Xoβ = Xo .* Poβ'

        # αvars = Variable(size(Xoβ, 2))
        # αloss = square(norm(Xoβ * αvars - yo))
        # constraints = [Po' * αvars == ones(size(Po, 2)), αvars >= 0]
        # problem = minimize(αloss, constraints)
        # solve!(problem, () -> ECOS.Optimizer())
        # α = αvars.value

        α = nonneg_lsq(Xoβ, y, alg = nnlsalg)
        α = checkalpha(α, Po)

        @debug sum(abs.(Po' * α)) / length(α)

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
        β = Xoα \ yo             # idiomatic Julia way to solve the least squares problem

        oldoptval = optval
        optval = loss(α, β)
        @debug "optval (α fixed)  $optval"

        i += 1
    end

    result = (PartLSFitResult(α[1:end-1], β[1:end-1], β[end] * α[end], P), nothing, (;opt=optval))

    @debug "Exiting with optimality gap: $(abs(oldoptval - optval))"

    result
end
