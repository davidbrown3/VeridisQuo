include("./femlib.jl")
include("./optlib.jl")

using JLD
using Flux
using Plots
using Flux.Tracker
using ProgressMeter
using Main.femlib
using Main.optlib

## Problem Setup

# Defining Nodes
xyNodes =   param([
            0. 0.
            1. 2.
            2. 0.
            3. 2.
            4. 0.
            5. 2.
            6. 0.
            7. 2.
            8. 0.
            ])

# Defining Loads
Loads   = zeros(size(xyNodes, 1), 2)
Loads[5, :] = [0. -10.]

# Defining edges
edges   =   [
            1 2
            1 3
            2 3
            2 4
            3 4
            3 5
            4 5
            4 6
            5 6
            5 7
            6 7
            6 8
            7 8
            7 9
            8 9
            ]

bFixed  = [true, false, false, false, false, false, false, false, true]
bFree   = map(!, bFixed)

A       = 0.02 # Cross sectional area of beam
E       = 30e6 # Stiffness coefficient

Nodes   = [Node(x,y,b,b,i*2-1,i*2) for (i,(x,y,b)) in enumerate(zip(xyNodes[:,1], xyNodes[:,2], bFree))]
Edges   = map((a,b)->Edge(a, b, A, E, calc_IGlobal(a,b,Nodes[a],Nodes[b])), edges[:,1], edges[:,2])
Truss   = Structure(Nodes, Edges)
Sim     = LoadSimulation(Truss, Loads)

# Optimisation
Targets = 0. # Target displacement
CostFcn = (F,x) -> x[5] - Targets
Prob    = Problem(Sim, CostFcn)

# Selecting parameters for optimiation
θ       = Params([xyNodes])

# Configuring backprop
opt     = ADAM(1e-3)
NEpochs = Int64(5e3)

# Configuring log pre-allocation
log_xyNodes = Array{Array{Float64,2},1}(undef, NEpochs)
log_FEdges  = Array{Array{Float64,2},1}(undef, NEpochs)
log_cost    = Array{Float64,1}(undef, NEpochs)

# Calculating bNodeFree matrix
bNodeFree   = vcat(map(x->[x.bX, x.bY], Nodes)'...)

@showprogress for i = 1:NEpochs

    # Calculating forward pass
    Cost, FEdge, xDisplacement = optimiseFEM(Prob)

    # Calculating gradients
    gradfcn = Tracker.gradient(()->-optimiseFEM(Prob)[1], θ)
    grads = gradfcn[xyNodes] .* bNodeFree

    # SGD
    Tracker.update!(opt, xyNodes, grads)
    for (ix, Node) in enumerate(Sim.Structure.Nodes)
        Node.X = xyNodes[ix,1]
        Node.Y = xyNodes[ix,2]
    end

    # Storing data
    log_xyNodes[i] = map(x->x.data, xyNodes)
    log_FEdges[i] = map(x->x.data, FEdge)
    log_cost[i] = Cost.data

end

# Plotting progress
plt = plot(log_cost,
            ylabel = "Center node displacement",
            xlabel = "# Epoch",
            legend=false,
            linewidth=2
            )
png(plt, "./TrainingLog")

display(plt)

# Animating data
plot_structure(log_xyNodes, log_FEdges, Truss, 100)

# Saving data
jldopen("/tmp/fem_optimisation.jld", "w") do file
    #addrequire(file, femlib)
    write(file, "xyNodes", log_xyNodes)
    write(file, "FEdges", log_FEdges)
    #write(file, "Sim", Sim)
end
