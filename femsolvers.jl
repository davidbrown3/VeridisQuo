function solveFEM(Sim::LoadSimulation)

    # Parsing inputs
    Nodes = Sim.Structure.Nodes
    Edges = Sim.Structure.Edges
    Loads = Sim.Loads

    # Parsing Nodes
    iDOFFree, iDOFFixed = Int64[], Int64[]
    for (i,Node) in enumerate(Nodes)
        if Node.bX; push!(iDOFFree, Node.iX); else; push!(iDOFFixed, Node.iX); end
        if Node.bY; push!(iDOFFree, Node.iY); else; push!(iDOFFixed, Node.iY); end
    end

    xNodes = map(node->node.X, Nodes)
    yNodes = map(node->node.Y, Nodes)
    NDOFSystem = max(maximum(iDOFFree), maximum(iDOFFixed))

    # Parsing loads
    LoadsDOF = reshape(Loads',:,1)[:] # Indexing loads by DOF as opposed to node

    # For compatability with autodiff
    ADType = typeof(xNodes[1])

    # Calculate length of each edge
    Ls = map(e->(sqrt((xNodes[e.nodeA]-xNodes[e.nodeB])^2 + (yNodes[e.nodeA]-yNodes[e.nodeB])^2)),
                        Edges)

    # Calculate local→global angular deflection for each edge
    θs = map(e->atan(yNodes[e.nodeB]-yNodes[e.nodeA], xNodes[e.nodeB]-xNodes[e.nodeA]),
                        Edges)

    # Calculating rotation matrix
    # [4x4xNEdges]
    R_Local = map(calc_R, θs)

    # Build local stiffness matrix
    # [4x4xNEdges]
    KEdge_Local = map(calc_KLocal, Edges, Ls)

    # Transforming stiffness matrix to global coordinates
    # [4x4xNEdges]
    KEdge_Global = map((R, K)-> R * K * R', R_Local, KEdge_Local)

    # Assembling global stiffness matrix
    # [NDOFxNDOF]
    KSystem = zeros(ADType, NDOFSystem, NDOFSystem)
    for (Edge,KEdge) in zip(Edges,KEdge_Global)
        KSystem[Edge.LI] += KEdge
    end

    # Parsing global stiffness matrix
    KSystem_FixedFixed = KSystem[iDOFFixed, iDOFFixed]
    KSystem_FixedFree = KSystem[iDOFFixed, iDOFFree]
    KSystem_FreeFixed = KSystem[iDOFFree, iDOFFixed]
    KSystem_FreeFree = KSystem[iDOFFree, iDOFFree]

    # Prescribing known forces / displacements
    FFree = LoadsDOF[iDOFFree]
    xFixed = zeros(size(iDOFFixed))

    # Calculating unknown forces / displacements
    xFree = KSystem_FreeFree \ (FFree - KSystem_FreeFixed * xFixed)
    FFixed = KSystem_FixedFixed * xFixed + KSystem_FixedFree * xFree

    # Parsing displacements
    x = zeros(ADType, NDOFSystem)
    x[iDOFFixed] = xFixed
    x[iDOFFree] = xFree

    # Calculating force through edges
    FEdge = zeros(ADType, 4, length(Edges))
    for (i,edge) in enumerate(Edges)
        FEdge[:,i] = KEdge_Local[i] * x[[
                                        Nodes[edge.nodeA].iX;
                                        Nodes[edge.nodeA].iY;
                                        Nodes[edge.nodeB].iX;
                                        Nodes[edge.nodeB].iY
                                        ]]
    end

    # Calculating node euclidian displacement
    xDisplacement = map(node->sqrt(x[node.iX]^2 + x[node.iY]^2), Nodes)

    return FEdge, xDisplacement

end
