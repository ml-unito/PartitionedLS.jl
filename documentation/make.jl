using Pkg;
Pkg.activate("PartitionedLSenv", shared=true)
push!(LOAD_PATH,"../src/")

using PartitionedLS
using Documenter
makedocs(
         sitename = "PartitionedLS.jl",
         modules  = [PartitionedLS],
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
    repo="github.com/boborbt/PartitionedLS.jl",
)