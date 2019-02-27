module optlib

#include("./femlib.jl")
#using Main.optlib.femlib
using Main.femlib

# Exporting types
export Problem

# Exporting functions
export optimiseFEM

# Defining a Problem type
struct Problem
    Sim::LoadSimulation
    CostFcn::Function
end

function optimiseFEM(Prob::Problem)

    FEdge, xDisplacement = solveFEM(Prob.Sim)
    cost = Prob.CostFcn(FEdge, xDisplacement)
    
    return cost, FEdge, xDisplacement

end

end
