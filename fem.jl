# Introduction to Finite Element Analysis Using MATLAB and Abaqus

using ForwardDiff, ReverseDiff
# using Zygote

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
iNodes = 1:NNodes
iNodeFixed = [1, 9]
iNodeFree = setdiff(iNodes, iNodeFixed)

NFixed = length(iNodeFixed)
NFree = length(iNodeFree)

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

loads = zeros(NNodes, 2)
loads[2, :] = [15. 0.]
loads[3, :] = [0. -5.]
loads[4, :] = [0. -7.]
loads[7, :] = [0. -10.]

# Indexing loads by DOF as opposed to node
# TODO: Why [:]
loadsDOF = reshape(loads',:,1)[:]

NEdges = size(edges, 1)

Es = 30.e6 * ones(NEdges, 1)
As = 0.02 * ones(NEdges, 1)

function solveFEM(As)

    NNodes = size(nodes, 1)
    NDOFNode = 2
    NDOFSystem = NDOFNode * NNodes

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
    KSystem = zeros(typeof(As[1]), NDOFSystem, NDOFSystem)

    for i = 1:NEdges
        Ks = reshape(KEdge_Global[i], 16, 1) # Unravelling a 4x4
        LIs = Array{CartesianIndex{2},1}(undef,16)
        base = [edges[i,1]*2-1 edges[i,1]*2 edges[i,2]*2-1 edges[i,2]*2]
        for j1 = 1:4, j2 = 1:4
            ix = (j1-1)*4+j2
            LIs[ix] = CartesianIndex(base[j2], base[j1])
        end
        KSystem[LIs] += Ks
    end

    KSystem_FixedFixed = KSystem[iDOFFixed, iDOFFixed]
    KSystem_FixedFree = KSystem[iDOFFixed, iDOFFree]
    KSystem_FreeFixed = KSystem[iDOFFree, iDOFFixed]
    KSystem_FreeFree = KSystem[iDOFFree, iDOFFree]

    FFree = loadsDOF[iDOFFree]
    xFixed = zeros(size(iDOFFixed))
    xFree = KSystem_FreeFree \ (FFree - KSystem_FreeFixed * xFixed)
    FFixed = KSystem_FixedFixed * xFixed + KSystem_FixedFree * xFree

    return xFree[1]

end

solution = solveFEM(As)

out = ForwardDiff.gradient(solveFEM, As)
println(out)
out = ReverseDiff.gradient(solveFEM, As)
println(out)
# out = Zygote.gradient(solveFEM, As)
# println(out)
