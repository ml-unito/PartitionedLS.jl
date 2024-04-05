using PartitionedLS
using MLJModelInterface
using Documenter

format = Documenter.HTML(sidebar_sitename=false)

makedocs(
    format=format,
    sitename="PartitionedLS.jl",
    modules = [PartitionedLS]
)

# makedocs(
#          sitename = "PartitionedLS.jl",
#          modules  = [PartitionedLS],
#          pages = [
#                 "Home" => "index.md"
#                ])
# deploydocs(;
#     repo="github.com/boborbt/PartitionedLS.jl",
# )
