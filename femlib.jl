module femlib

# Exporting fundamental FEM types
export Edge, Node, Structure, LoadSimulation

# Exporting FEM solver functions
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

# Defining a Structure type
struct Structure
    Nodes::Array{Node,1}
    Edges::Array{Edge,1}
end

struct LoadSimulation
    Structure::Structure
    Loads::Array{Float64,2}
end

include("./femtools.jl")
include("./femsolvers.jl")

end
