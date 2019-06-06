using TLLR: fit_iterative, fit_iterative_slow, fit

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


result = fit_iterative_slow(X, y, P, verbose=0)
# result_opt = fit(X, y, P, verbose=0)