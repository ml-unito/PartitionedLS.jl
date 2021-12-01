---
layout: default
title: Home
---

  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Partitioned Least Squares

Linear least squares is one of the most widely used regression methods among scientists in many fields. The simplicity of the model allows this method to be used when data is scarce and it is usually appealing to practitioners that need to gather some insight into the problem by inspecting the values of the learnt parameters. PartitionedLS is a variant of the linear least squares model allowing practitioners to partition the input features into groups of variables that they require to contribute similarly to the final result. 

## The model

The Partitioned Least Squares model is formally defined as:

$$
\begin{gather*}
\text{minimize}_{\mathbf{\alpha}, \mathbf{\beta}} \| \mathbf{X} \times (\mathbf{P} \circ \mathbf{\alpha}) \times \mathbf{\beta} - \mathbf{y} \|_2^2 \\
\begin{aligned}
\quad s.t.\quad  &\mathbf{\alpha}  \succeq 0\\
                    &\mathbf{P}^T \times \mathbf{\alpha} = \mathbf{1}.
\end{aligned}
\end{gather*}
$$

where: 

- $$\mathbf{X}$$ is $$N \times M$$ data matrix;
- $$\mathbf{P}$$ is a user-defined partition matrix having $$K$$ columns (one for each element of the partition), $$M$$ rows, and containing $$1$$ in $$P_{i,j}$$ if the $$i$$-th attribute belongs to the $$j$$-th partition and $$0$$ otherwise;
- $$\mathbf{\beta}$$ is a vector weighting the importance of each set of attributes in the partition;
- $$\mathbf{\alpha}$$ is a vector weighting the importance of each attribute within one of the sets in the partition. Note that the constraints imply that for each set in the partition the weights of the corresponding $$\alpha$$ variables are all positive and sum to $$1$$.

The PartitionedLS problem is non-convex and NP-complete. The library provides two algorithms to solve the problem anyway: an iterative algorithm based on the Alternating Least Squares approach and an optimal algorithm that guarantees requiring however exponential time in the cardinality of the partition (i.e., it is mainly useful when $$K$$ is small).

More details can be found in the paper [Partitioned Least Squares](https://arxiv.org/abs/2006.16202).

## To install this library

Just add it as a dependency to your Julia environment. Launch julia from the main directory of your project and enter the following commands:

```julia
# Opens the package manager REPL
]

# Activate you local environment (can be skipped if you want to install the library globally)
activate .

# Adds the library to the environment
add git@github.com:ml-unito/PartitionedLS.git
```

## To use this library

You will need a matrix P describing the partitioning of your variables, e.g.:

```julia
P = [[1 0]; 
     [1 0]; 
     [0 1]]
```

specifies that the first and the second variable belongs to the first partition, while the third variable belongs to the second.

You then just give your data to the `fit` function and use the `predict` function to make predictions. 

A complete example:

```julia
using PartitionedLS

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


# fit using the optimal algorithm and using generic solver
result_opt = fit(Opt, X, y, P, η = 0.0)

# fit using the optimal algorithm using a non negative least squares
# solver (numerical solutions should be more stable)
result_opt_nnls = fit(OptNNLS, X, y, P)

# fit using the iterative algorithm and a generic solver
result_alt = fit(Alt, X, y, P, η = 0.0)

# fit using the iterative algorithm using a non negative least squares
# solver (numerical solutions should be more stable)
result_alt_nnls = fit(AltNNLS, X, y, P)

# Make predictions on the given data matrix. The function works
# with results returned by anyone of the solvers.
predict(result_opt, X)
```

## Performances

<iframe src="BnB.html"></iframe>