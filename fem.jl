using ForwardDiff
using Plots
using JLD
using ColorSchemes

nodes = [
    0. 0.
    1. 2.
    2. 0.
    3. 2.
    4. 0.
    5. 2.
    6. 0.
    7. 2.
    8. 0.
]

# Enumerate each nodes DOFs: [x1, y1, x2, y2, ..., xn, yn] for nodes i→n
NNodes = size(nodes, 1)
iDOF = reshape((1:NNodes*2), 2, NNodes)'

nodesMin = [0. 0.]
nodesMax = [8.0 5.0]

iNodes = 1:NNodes
iNodeFixed = [1, 9]
iNodeFree = setdiff(iNodes, iNodeFixed)

iNodeOptimise = 5

NFixed = length(iNodeFixed)
NFree = length(iNodeFree)

NNodes = size(nodes, 1)

NDOFNode = 2
NDOFSystem = NDOFNode * NNodes

# Defining a function to convert iNodes to iDOF
iNode2iDOF(iNode) = reshape(reduce(vcat, # Long winded way of flattening array of arrays → matrix
                        map(i->[2*i-1 2*i], iNode) # Enumerating DOF for each node
                        )', :, 1) # Transposing for correct unravelling in reshape

# TODO: Why is there the need to [:]
iDOFFixed = iNode2iDOF(iNodeFixed)[:]
iDOFFree = iNode2iDOF(iNodeFree)[:]

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

NEdges = size(edges, 1)

loads = zeros(NNodes, 2)

#TODO: Relate learning rate to force
loads[5, :] = [0. -10.]

# Indexing loads by DOF as opposed to node
loadsDOF = reshape(loads',:,1)[:]

Es = 30.e6 * ones(NEdges, 1)

#TODO: Move this out of the loop
LIs = Array{Array{CartesianIndex{2},2},1}(undef, NEdges)
for i = 1:NEdges
    LI = Array{CartesianIndex{2},2}(undef,4,4)
    base = [edges[i,1]*2-1 edges[i,1]*2 edges[i,2]*2-1 edges[i,2]*2]
    for j1 = 1:4, j2 = 1:4
        LI[j2, j1] = CartesianIndex(base[j2], base[j1])
    end
    LIs[i] = LI
end

function solveFEM(nodes)

    # Calculate length of each edge
    Ls = map((i,j)->(sqrt((nodes[i,1]-nodes[j,1])^2 + (nodes[i,2]-nodes[j,2])^2)),
                        edges[:,1], edges[:,2])

    # Calculate local→global angular deflection for each edge
    θs = map((i,j)->atan(nodes[j,2]-nodes[i,2], nodes[j,1]-nodes[i,1]),
                        edges[:,1], edges[:,2])

    # Generating transformation matrix for each edge
    GenR = θ->[
                cos(θ) -sin(θ) 0. 0.
                sin(θ) cos(θ) 0. 0.
                0. 0. cos(θ) -sin(θ)
                0. 0. sin(θ) cos(θ)
            ]

    R_Local = map(GenR, θs)

    # Build local stiffness matrix
    # Operating on [x1, y1, x2, y2]
    KEdge_Local = map((E,A,L)->[[E*A/L 0. -E*A/L 0.]; [0. 0. 0. 0.]; [-E*A/L 0. E*A/L 0.]; [0. 0. 0. 0.]],
                                Es, As, Ls)

    # Transforming stiffness matrix to global coordinates
    KEdge_Global = map((R, K)-> R * K * R', R_Local, KEdge_Local)

    # For compatability with autodiff
    ADType = typeof(nodes[1])
    KSystem = zeros(ADType, NDOFSystem, NDOFSystem)

    for i = 1:NEdges
        KSystem[LIs[i]] += KEdge_Global[i]
    end

    KSystem_FixedFixed = KSystem[iDOFFixed, iDOFFixed]
    KSystem_FixedFree = KSystem[iDOFFixed, iDOFFree]
    KSystem_FreeFixed = KSystem[iDOFFree, iDOFFixed]
    KSystem_FreeFree = KSystem[iDOFFree, iDOFFree]

    FFree = loadsDOF[iDOFFree]
    xFixed = zeros(size(iDOFFixed))
    xFree = KSystem_FreeFree \ (FFree - KSystem_FreeFixed * xFixed)
    FFixed = KSystem_FixedFixed * xFixed + KSystem_FixedFree * xFree

    x = zeros(ADType, NDOFSystem)
    x[iDOFFixed] = xFixed
    x[iDOFFree] = xFree

    # Calculating force in edge
    FEdge = zeros(ADType, 4, NEdges)
    for i = 1:NEdges
        xVec = reshape(x[iDOF[edges[i,:],:]]',4,1)
        FEdge[:,i] = KEdge_Local[i] * xVec
    end

    # Calculating node euclidian displacement
    xDisplacement = sum(x[iDOF].^2,dims=2).^0.5

    return FEdge, xDisplacement

end

function optimiseFEM(nodes)

    FEdge, xDisplacement = solveFEM(nodes)
    cost = xDisplacement[iNodeOptimise]

end

NEpochs = Int64(1e5)
learn = 1.
momentum = 0.9
grads = zeros(size(nodes))

As = 0.02 * ones(NEdges, 1)

solution = solveFEM(nodes)
println(solution)

storage = Array{Array{Float64,2},1}(undef, NEpochs)
storage2 = Array{Array{Float64,2},1}(undef, NEpochs)

for i = 1:NEpochs

    # Monitoring progress
    if mod(i,1000) == 0
        println(i)
    end

    # Setting As as global so it can be modified within for loop
    global nodes
    global grads

    #
    FEdge, xDisplacement = solveFEM(nodes)

    # DIY momentum
    grads_prev = grads
    grads = ForwardDiff.gradient(optimiseFEM, nodes)

    # Running gradient descent (Fixing the prescribed nodes)
    nodes[2:end-1,:] -= (grads[2:end-1,:] * learn + momentum * grads_prev[2:end-1,:])

    # Storing data
    storage[i] = copy(nodes)
    storage2[i] = copy(FEdge)

end

# Saving cached data
save("/tmp/myfile.jld", "storage", storage, "storage2", storage2, "edges", edges)

# Loading cached data
cached = load("/tmp/myfile.jld")
storage = cached["storage"]
storage2 = cached["storage2"]
edges = cached["edges"]
nodes = storage[end]

solution = solveFEM(nodes)
println(nodes)
println(solution)

Fs = map(x->x[1,:], storage2)
FMax = max(maximum(map(x->maximum(x), Fs)), -minimum(map(x->minimum(x), Fs)))

@gif for i in 1:1000:size(storage,1)

    plt = scatter(storage[i][:,1], storage[i][:,2], legend=false)

    for j = 1:size(edges, 1)
        col = get(ColorSchemes.coolwarm, storage2[i][1,j]/FMax/2+0.5)
        plot!(plt, storage[i][edges[j,:],1], storage[i][edges[j,:],2], legend=false, linecolor=col)
    end

end

plt = scatter(storage[end][:,1], storage[i][:,2], legend=false)
for j = 1:size(edges, 1)
    plot!(plt, storage[end][edges[j,:],1], storage[end][edges[j,:],2], legend=false)
end

# plt = scatter(nodes[:,1], nodes[:,2], legend=false)
# for j = 1:size(edges, 1)
#     plot!(plt, nodes[edges[j,:],1], nodes[edges[j,:],2], legend=false)
# end
# display(plt)
