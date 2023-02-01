using Pkg
Pkg.activate("PartitionedLSenv", shared=true)

using PartitionedLS

X = [[1.0 2.0 3.0]
     [3.0 3.0 4.0]
     [8.0 1.0 3.0]
     [5.0 3.0 1.0]]

y = [1.0
     1.0
     2.0
     3.0]

P = [[1 0]
     [1 0]
     [0 1]]

# io = open("log.txt", "w+")
# logger = SimpleLogger(io)
# global_logger(logger)

# @info "Calling Opt"
# result_opt = fit(Opt, X, y, P, η = 0.0)

# @info "Calling Opt nnls"
# result_opt_nnls = fit(OptNNLS, X, y, P)
# opt, α, β, t, _ = result_opt_nnls
# @info "OptNNLS:", opt, α, β, t

# @info "Calling Alt"
# result_alt = fit(Alt, X, y, P, η = 0.0)


@info "Calling Alt nnls"
result_alt_nnls = fit(AltNNLS, X, y, P)
opt, a, b, t, P = result_alt_nnls

@info "Calling BnB"
opt, α, β, t, _ = fit(BnB, X, y, P)
@info "BnB:", opt, α, β, t

# @info "Opt" norm(predict(result_opt, X) - y, 2)
# @info "Alt" norm(predict(result_alt, X) - y, 2)
# @info "Alt nnls" norm(predict(result_alt_nnls, X) - y, 2)