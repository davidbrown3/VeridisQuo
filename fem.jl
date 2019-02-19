# Introduction to Finite Element Analysis Using MATLAB and Abaqus

using ForwardDiff
# using ReverseDiff
#using Flux.Tracker: grad, update!
#using Flux, Flux.Tracker
using Statistics: mean
# using Zygote
using Plots
using JLD

# Setting plot backends
plotlyjs()

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

nodesMin = [0. 0.]
nodesMax = [8.0 5.0]

# Enumerate each nodes DOFs: [x1, y1, x2, y2, ..., xn, yn] for nodes i→n

NNodes = size(nodes, 1)
iNodes = 1:NNodes
iNodeFixed = [1, 9]
iNodeFree = setdiff(iNodes, iNodeFixed)

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
loads[5, :] = [0. -10.]

# Indexing loads by DOF as opposed to node
# TODO: Why [:]
loadsDOF = reshape(loads',:,1)[:]

Es = 30.e6 * ones(NEdges, 1)

function solveFEM(nodes)

    # Calculate length
    Ls = map((i,j)->(sqrt((nodes[i,1]-nodes[j,1])^2 + (nodes[i,2]-nodes[j,2])^2)),
                        edges[:,1], edges[:,2])

    # Calculate local→global transformation matrix
    θs = map((i,j)->atan(nodes[j,2]-nodes[i,2], nodes[j,1]-nodes[i,1]),
                        edges[:,1], edges[:,2])

    GenR = θ->[
                cos(θ) -sin(θ) 0. 0.
                sin(θ) cos(θ) 0. 0.
                0. 0. cos(θ) -sin(θ)
                0. 0. sin(θ) cos(θ)
            ]

    R_Local = map(GenR, θs)

    # Operating on [x1, y1, x2, y2]

    # Build local stiffness matrix
    # Array of arrays during autodiff

    KEdge_Local = map((E,A,L)->[[E*A/L 0. -E*A/L 0.]; [0. 0. 0. 0.]; [-E*A/L 0. E*A/L 0.]; [0. 0. 0. 0.]],
                                Es, As, Ls)

    KEdge_Global = map((R, K)-> R * K * R', R_Local, KEdge_Local)

    # For compatability with autodiff
    #KSystem = zeros(typeof(As[1), NDOFSystem, NDOFSystem)
    ADType = typeof(nodes[1])
    KSystem = zeros(ADType, NDOFSystem, NDOFSystem)

    for i = 1:NEdges
        LIs = Array{CartesianIndex{2},2}(undef,4,4)
        base = [edges[i,1]*2-1 edges[i,1]*2 edges[i,2]*2-1 edges[i,2]*2]
        for j1 = 1:4, j2 = 1:4
            LIs[j2, j1] = CartesianIndex(base[j2], base[j1])
        end
        KSystem[LIs] += KEdge_Global[i]
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

    return -x[10]

end

#out = ForwardDiff.gradient(solveFEM, As)
#println(out)

# TODO: Can not get Zygote library working
# out = Zygote.gradient(solveFEM, As)
# println(out)

# TODO: Can not get Flux library working
#fluxF(A) = Tracker.gradient(solveFEM, A)[1]
#fluxF(As)

#plot(x=map(x->x+1,nodes[:,1]), y=nodes[:,2], Geom.point)

# TODO: Can not get Flux optimisation components to work
#opt = Descent(0.1)

NEpochs = Int64(1e5)
learn = 1.
momentum = 0.9
grads = zeros(size(nodes))

As = 0.02 * ones(NEdges, 1)

solution = solveFEM(nodes)
#println(As)
println(solution)

storage = Array{Array{Float64,2},1}(undef, NEpochs)

for i = 1:NEpochs

    if mod(i,100)==0
        println(i)
    end

    # Setting As as global so it can be modified within for loop
    global nodes
    global grads

    # DIY momentum
    grads_prev = grads

    #TODO: print solution periodically
    #solution = solveFEM(nodes)
    grads = ForwardDiff.gradient(solveFEM, nodes)

    # Running gradient descent
    #TODO: Fixing the prescribed nodes
    nodes[2:end-1,:] -= (grads[2:end-1,:] * learn + momentum * grads_prev[2:end-1,:])

    # Storing data
    storage[i] = copy(nodes)

end

# Saving compiled data
save("/tmp/myfile.jld", "nodes", storage)

#TODO: Use activation function to limit variable limits

solution = solveFEM(nodes)
println(nodes)
println(solution)

@gif for i in 1:100:size(storage,1)

    plt = scatter(storage[i][:,1], storage[i][:,2])

    for j = 1:size(edges, 1)
        plot!(plt, storage[i][edges[j,:],1], storage[i][edges[j,:],2])
    end

    # gif(anim, "/tmp/anim_fps30.gif", fps = 30)

end

# Plot in the loop
# TODO: Need to install and compile plots library
# plt = scatter(nodes[:,1], nodes[:,2])
#
# for i = 1:size(edges, 1)
#     plot!(plt, nodes[edges[i,:],1], nodes[edges[i,:],2])
# end
#
# gif(anim, "/tmp/anim_fps30.gif", fps = 30)


#display(plt)
