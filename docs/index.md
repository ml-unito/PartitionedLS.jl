---
layout: default
title: Home
---

# Partitioned Least Squares

Linear least squares is one of the most widely used regression methods in all the sciences. The simplicity of the model allows this method to be used when data is scarce and it is usually appealing to practitioners that need to gather some insight into the problem by in- specting the values of the learnt parameters. This library implements a variant of the linear least squares model that allows practitioners to partition the input features into groups of variables that they require to contribute similarly to the final result. The library provides two alternative methods to attack the problem: one non-exact method (implemented by PartitionedLS.fit_alternating) based on an alternating least square approach; and one (implemented by PartitionedLS.fit) exact method based on a reformulation of the problem using an exponential number of sub-problems whose minimum is guaranteed to be the optimal solution. In most practical cases the PartitionedLS.fit is both faster and more accurate.

# To install this library

Just add it as a dependency to your Julia environment. Launch julia from the main directory of your project and enter the following commands:

```julia
# Opens the package manager REPL
]

# Activate you local environment (can be skipped if you want to install the library globally)
activate .

# Adds the library to the environment
add git@github.com:ml-unito/PartitionedLS.git
```

# To use this library

You will need a matrix P describing the partitioning of your variables, e.g.:

```julia
P = [[1 0]; 
     [1 0]; 
     [0 1]]
```

specifies that the first and the second variable belongs to the first partition, while the third variable belongs to the second.

You then just give your data to the `fit` function and use the `predict` function to make predictions. 

A complete example:

```
using PartitionedLS: fit_alternating, fit

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

result = fit_alternating(X, y, P, verbose=0)
result_opt = fit(X, y, P, verbose=0)

predictions = predict(result_opt, X)
```
