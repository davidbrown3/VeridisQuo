using Plots
using ColorSchemes

# Exporting FEM tool functions
export calc_R, calc_KLocal, calc_IGlobal, plot_structure

# Function to generate rotation matricies
calc_R(θ) = [
            cos(θ) -sin(θ) 0. 0.
            sin(θ) cos(θ) 0. 0.
            0. 0. cos(θ) -sin(θ)
            0. 0. sin(θ) cos(θ)
            ]

# Generating local stiffness matrix
# Operating on [x1, y1, x2, y2]
calc_KLocal(e::Edge,L) =    [
                            e.E*e.A/L 0. -e.E*e.A/L 0.;
                            0. 0. 0. 0.;
                            -e.E*e.A/L 0. e.E*e.A/L 0.;
                            0. 0. 0. 0.
                            ]

# Indexing an edge relative to global list of DOFs
function calc_IGlobal(a::Int64, b::Int64, NodeA::Node, NodeB::Node)

    LI = Array{CartesianIndex{2},2}(undef,4,4)
    base = [NodeA.iX, NodeA.iY, NodeB.iX, NodeB.iY]

    for j1 = 1:4, j2 = 1:4
        LI[j2, j1] = CartesianIndex(base[j2], base[j1])
    end

    return LI

end

# Setting plot backends
plotlyjs()

function plot_structure(xyNodes::Array{Array{Float64,2},1}, FEdges::Array{Array{Float64,2},1}, Structure::Structure, inc::Int64=100)

    @gif for i in 1:100:size(xyNodes, 1)
        Fs = FEdges[i][1,:]
        FMax = max(maximum(Fs), -minimum(Fs))
        plt = scatter(xyNodes[i][:,1], xyNodes[i][:,2], legend=false)
        for (j,Edge) in enumerate(Structure.Edges)
            col = get(ColorSchemes.coolwarm, FEdges[i][1,j]/FMax/2+0.5)
            plot!(plt, xyNodes[i][[Edge.nodeA, Edge.nodeB], 1], xyNodes[i][[Edge.nodeA, Edge.nodeB], 2],
                    legend=false, linecolor=col, linewidth=2)
        end
    end

end
