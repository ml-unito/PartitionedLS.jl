using PartitionedLS
using Documenter

format = Documenter.HTML(sidebar_sitename=false)

makedocs(
    format=format,
    sitename="PartitionedLS.jl"
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
