module femlib

export Edge, Node
export calc_R, calc_KLocal, calc_IGlobal
export solveFEM

# Defining an Edge type
struct Edge
    nodeA::Int64
    nodeB::Int64
    A::Float64
    E::Float64
    LI::Array{CartesianIndex{2},2}
end

# Defining a Node type
mutable struct Node
    X::Any # Cartesian Position (Compatable with AD types)
    Y::Any
    bX::Bool # Is DOF free?
    bY::Bool
    iX::Int64 # What is global index of DOF
    iY::Int64
end

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

include("./femsolvers.jl")

end
