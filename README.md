# Partitioned Least Squares

![Static Badge](https://img.shields.io/badge/Docs-%E2%9C%93-green) ![Static Badge](https://img.shields.io/badge/Tests-%E2%9C%93-green)

Linear least squares is one of the most widely used regression methods among scientists in many fields. The simplicity of the model allows this method to be used when data is scarce and it is usually appealing to practitioners that need to gather some insight into the problem by inspecting the values of the learnt parameters. 

PartitionedLS is a **variant** of the linear least squares model allowing practitioners to **partition the input features into groups of variables** that they require to contribute with the same sign to the final result. 

For instance, when analyzing complex chemical compounds, it is possible to group together fine-grained features to obtain a partition which refers to high-level properties of the compound (such as structural, interactive and bond-forming among others), and knowing how much each high-level property contributes to the result of the analysis is often of great practical value. The PartitionedLS squares problem allows a practitioner to specify how to group the variables together. 

Then, the target value for a new sample $x$ will be predicted as:
$$
f(x) = \sum_{k=1}^{K} \beta_i \sum_{m \in P_k} \alpha_m x_m + t
$$
where: 
- $K$ is the number of groups;
- $P_k$ is the set of indices of the variables in the $k$-th group;
- $\sum_{m \in P_k} \alpha_m = 1$
- we have a $\beta$ variable for each group, specifying how much (and in which direction) the group contributes to the final result;
- an $\alpha$ variable for each feature in the group, specifying how much the feature contributes to the group.

The optimization problem is solved by minimizing the sum of the squared residuals, with the additional constraint that the sum of the $\alpha$ variables in each group is equal to 1.

While the interpretability angle is important, it is worth noticing that the constraint posed by the algorithm also act as an inductive bias, which can lead the optimization to a better solution when it matches the structure of the underlying phenomenon.

# Basic usage

The library is very easy to use. Here is an example:

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
y_hat = predict(result.model, X)
```

You can choose between three algorithms to solve the problem: `Opt`, `Alt`, and `BnB`. The `Opt` algorithm is the optimal one, but it is exponential in the number of partitions. The `Alt` algorithm is an iterative one, based on the Alternating Least Squares approach. The `BnB` algorithm is a variant of the `Opt` algorithm that is often faster in practice and still guarantees the optimal solution.


# To learn more

Theoretical details about PartitionedLS can be found on [this paper](https://arxiv.org/abs/2006.16202) and [here](https://ml-unito.github.io/PartitionedLS.jl/jldocs/build/index.html) you can find the documentation of the exported functions.

Instructions about how to intall the library and a complete example can be found
on the [github pages of this project](https://ml-unito.github.io/PartitionedLS.jl/).

