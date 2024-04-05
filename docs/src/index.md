# Partitioned Least Squares

Linear least squares is one of the most widely used regression methods among scientists in many fields. The simplicity of the model allows this method to be used when data is scarce and it is usually appealing to practitioners that need to gather some insight into the problem by inspecting the values of the learnt parameters. PartitionedLS is a variant of the linear least squares model allowing practitioners to partition the input features into groups of variables that they require to contribute similarly to the final result.

An example of analysing a dataset using PartitionedLS is given [here](examples/example.md)

## The model

The Partitioned Least Squares model is formally defined as:

```math
\begin{gather*}
\text{minimize}_{\mathbf{\alpha}, \mathbf{\beta}} \| \mathbf{X} \times (\mathbf{P} \circ \mathbf{\alpha}) \times \mathbf{\beta} - \mathbf{y} \|_2^2 \\
\begin{aligned}
\quad s.t.\quad  &\mathbf{\alpha}  \succeq 0\\
                    &\mathbf{P}^T \times \mathbf{\alpha} = \mathbf{1}.
\end{aligned}
\end{gather*}
```

where: 

- ``\mathbf{X}`` is ``N \times M`` data matrix;
- ``\mathbf{P}`` is a user-defined partition matrix having ``K`` columns (one for each element of the partition), ``M`` rows, and containing ``1`` in ``P_{i,j}`` if the ``i``-th attribute belongs to the ``j``-th partition and ``0`` otherwise;
- ``\mathbf{\beta}`` is a vector weighting the importance of each set of attributes in the partition;
- ``\mathbf{\alpha}`` is a vector weighting the importance of each attribute within one of the sets in the partition. Note that the constraints imply that for each set in the partition the weights of the corresponding ``\alpha`` variables are all positive and sum to ``1``.

The PartitionedLS problem is non-convex and NP-complete. The library provides two algorithms to solve the problem anyway: an iterative algorithm based on the Alternating Least Squares approach and an optimal algorithm that guarantees requiring however exponential time in the cardinality of the partition (i.e., it is mainly useful when ``K`` is small).

More details can be found in the paper [Partitioned Least Squares](https://arxiv.org/abs/2006.16202).

## To install this library

Just add it as a dependency to your Julia environment. Launch julia from the main directory of your project and enter the following commands:

```julia
# Opens the package manager REPL
]

# Activate you local environment (can be skipped if you want to install the library globally)
activate .

# Adds the library to the environment
add PartitionedLS
```

## To use this library

You will need a matrix P describing the partitioning of your variables, e.g.:

```julia
P = [[1 0]; 
     [1 0]; 
     [0 1]]
```

specifies that the first and the second variable belongs to the first partition, while the third variable belongs to the second.

You have then the choice to use either the standard interface or the MLJ interface. 

### Standard interface

The standard interface defines a `fit` function for each of the implemented algorithms. The function returns a tuple containing:
- a `PartLSFitResult` object containing the model and the parameters found by the algorithm;
- `nothing` (this is mandated by the MLJ interface, but it is not used in this case).
- a NamedTuple containing some additional information.

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


# fit using the optimal algorithm 
result = fit(Opt, X, y, P, η = 0.0)


# Make predictions on the given data matrix. The function works
# with results returned by anyone of the solvers.
predict(result[1], X)
```

### MLJ interface

The MLJ interface is a allows you to use the library in a more MLJ-like fashion. The interface is defined by the [`PartLS`](@ref) model, which can be used in the MLJ framework. The model can be used in the same way as any other MLJ model.

A complete example:

```julia
using MLJ
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

# Define the model

model = PartLS(P=P, Optimizer=Opt, η=0.0)

# Fit the model
mach = machine(model, X, y)
fit!(mach)

# Make predictions
predict(mach, X)
```


## API Documentation
```@docs
PartLS
PartLSFitResult
PartitionedLS.fit
PartitionedLS.predict
PartitionedLS.homogeneousCoords
PartitionedLS.regularizeProblem
```
