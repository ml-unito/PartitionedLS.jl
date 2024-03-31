# Partitioned Least Squares

Linear least squares is one of the most widely used regression methods among scientists in many fields. The simplicity of the model allows this method to be used when data is scarce and it is usually appealing to practitioners that need to gather some insight into the problem by inspecting the values of the learnt parameters. 

PartitionedLS is a **variant** of the linear least squares model allowing practitioners to **partition the input features into groups of variables** that they require to contribute with the same sign to the final result. 

For instance, when analyzing complex chemical compounds, it is possible to group together fine-grained features to obtain a partition which refers to high-level properties of the compound (such as structural, interactive and bond-forming among others), and knowing how much each high-level property contributes to the result of the analysis is often of great practical value. The PartitionedLS squares problem allows a practitioner to specify how to group the variables together. The solution will contain: 
- a $\beta$ variable for each group, specifying how strong is the correlation of that group of variables with the solution and the direction of the correlation;
- an $\alpha$ variable for each features, with the constraint that the set of $\alpha$ variables corresponding to a given group will be all positive and sum to $1$.
The idea is that, with this formulation, the solution will be easily interpretable, as the user will have knowledge of the "importance" of each group and the importance of each variable within the group.

While the interpretability angle is important, it is worth noticing that the constraint posed by the algorithm also act as an inductive bias, which can lead the optimization to a better solution when it matches the structure of the underlying phenomenon.

# To learn more

Theoretical details about PartitionedLS can be found on [this paper](https://arxiv.org/abs/2006.16202).

Instructions about how to intall the library and a complete example can be found
on the [github pages of this project](https://ml-unito.github.io/PartitionedLS.jl/).
