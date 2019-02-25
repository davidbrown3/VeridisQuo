include("./femlib.jl")

using Plots
using JLD
using ColorSchemes
using Flux
using Flux.Tracker
using ProgressMeter
using Main.femlib

# Setting plot backends
plotlyjs()

## Problem Setup

# Defining Nodes
xyNodes = param([
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
NNodes = size(xyNodes, 1)

Nodes = [Node(x,y,true,true,i*2-1,i*2) for (i,(x,y)) in enumerate(zip(xyNodes[:,1], xyNodes[:,2]))]

iNodeFixed = [1, 9]
bNodeFree = ones(NNodes,2)
for i in iNodeFixed
    Nodes[i].bX = false
    Nodes[i].bY = false
    bNodeFree[i,:] *= 0.
end

iNodeOptimise = 5

# Defining edges
edges = [
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

A = 0.02
E = 30e6
Edges = map((a,b)->Edge(a,b,A,E,calc_IGlobal(a,b,Nodes[a],Nodes[b])), edges[:,1], edges[:,2])

# Defining Loads
loads = zeros(NNodes, 2)
loads[5, :] = [0. -10.] #TODO: Relate learning rate to force

function optimiseFEM(Nodes, loads, targets)

    FEdge, xDisplacement = solveFEM(Nodes, Edges, loads)
    cost = xDisplacement[iNodeOptimise]

    return cost

end

θ = Params([xyNodes])
target = 0. # Target displacement
gradfcn = Tracker.gradient(()->optimiseFEM(Nodes, loads, target), θ)

NEpochs = Int64(1e5)

storage = Array{Array{Float64,2},1}(undef, NEpochs)
storage2 = Array{Array{Float64,2},1}(undef, NEpochs)

opt = ADAM()

@showprogress for i = 1:NEpochs

    # Calculating edge properties
    FEdge, xDisplacement = solveFEM(Nodes, Edges, loads)

    # Calculating gradients
    grads = gradfcn[xyNodes] .* bNodeFree

    # SGD
    Tracker.update!(opt, xyNodes, grads)
    for (ix, Node) in enumerate(Nodes)
        Node.X = xyNodes[ix,1]
        Node.Y = xyNodes[ix,2]
    end

    # Storing data
    storage[i] = map(x->x.data, xyNodes)
    storage2[i] = map(x->x.data, FEdge)

end

# Saving cached data
save("/tmp/myfile.jld", "storage", storage, "storage2", storage2, "edges", edges)

# Loading cached data
cached = load("/tmp/myfile.jld")
storage = cached["storage"]
storage2 = cached["storage2"]
edges = cached["edges"]
nodes = storage[end]

@gif for i in 1:100:size(storage,1)
    Fs = storage2[i][1,:]
    FMax = max(maximum(Fs), -minimum(Fs))
    plt = scatter(storage[i][:,1], storage[i][:,2], legend=false)
    for j = 1:size(edges, 1)
        col = get(ColorSchemes.coolwarm, storage2[i][1,j]/FMax/2+0.5)
        plot!(plt, storage[i][edges[j,:],1], storage[i][edges[j,:],2], legend=false, linecolor=col)
    end

end
