using PartitionedLS
using ECOS
using LinearAlgebra
using Logging

X = [[1. 2. 3.]; 
     [3. 3. 4.]; 
     [8. 1. 3.]; 
     [5. 3. 1.]]

y = [1.; 
     1.; 
     2.; 
     3.]

P = [[1 0]; 
     [1 0]; 
     [0 1]]

# io = open("log.txt", "w+")
# logger = SimpleLogger(io)
# global_logger(logger)

@info "Calling Opt"
result_opt = fit(Opt, X, y, P, η = 0.0)

@info "Calling Opt nnls"
result_opt_nnls = fit(OptNNLS, X, y, P)
opt, a, b, t, P = result_opt_nnls

@info "Calling Alt"
result_alt = fit(Alt, X, y, P, η = 0.0)

@info "Calling Alt nnls"
result_alt_nnls = fit(AltNNLS, X, y, P)
opt, a, b, t, P = result_alt_nnls

@info "Opt" norm(predict(result_opt, X) - y, 2)
@info "Opt nnls" norm(predict(result_opt_nnls, X) - y, 2)
@info "Alt" norm(predict(result_alt, X) - y, 2)
@info "Alt nnls" norm(predict(result_alt_nnls, X) - y, 2)