from tkinter import N # TODO: what is this for?
import numpy as np
import scipy
np.set_printoptions(threshold=10000)
from scipy.sparse import csr_matrix
from . import Common
from .LogSinkhorn import LogSinkhorn as LogSinkhorn
from .CPPSinkhorn import CPPSinkhorn as CPPSinkhorn
# from .MultiScaleOT import MultiScaleOT as MultiScaleOT
from . import MultiScaleOT
#import lib.LogSinkhorn.LogSinkhorn as LogSinkhorn
#import lib.CPPSinkhorn.CPPSinkhorn as CPPSinkhorn
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
# aux functions for setting up the partitions

def GetPartitionIndices1D(N,cellSize,offset=0,full=False):
    """Provides indices for partitioning a 1D array of length N into equal blocks of size cellSize
    (with initial offset offset). Returns list of values [[a_1,b_1],...,[a_n,b_n]] where a_i to (b_i-1)
    are the indices of the i-th cell.
    If full=True is set, then each list is created explicitly, i.e. in a fina lstep the pair [a_i,b_i]
    is replaced by the list between the two values."""
    result=[]
    if offset>0:
        result.append([0,offset])
    for i in range((N-offset-1)//cellSize+1):
        result.append([offset+i*cellSize,min(N,offset+(i+1)*cellSize)])
        
    if full:
        result=[[x for x in range(*c)] for c in result]
    return result

def GetProductList(i1,i2,N2):
    """For two 1D slices i1=[a_1,b_1] and i2=[a_2,b_2] this returns an explicit list of all all flattened 2D
    indices in the product of the two 1D cells.
    N2 is the length of the second axis of the 2D array and is needed to compute the strides of the flattened indices."""
    return [x*N2+y for x in range(*i1) for y in range(*i2)]
    

def GetPartitionIndices2D(shape,cellSize,offset=0):
    """Provides lists of indices for partitioning a 2D array of shape shape into equal cells of size
    cellSize x cellSize with an offset offset at the boundaries.
    Returns a list of index lists, where each index list provides indices for one cell.
    The indices are given as 1D indices for the flattened version of the array to be partitioned."""
    l1=GetPartitionIndices1D(shape[0],cellSize,offset)
    l2=GetPartitionIndices1D(shape[1],cellSize,offset)
    
    return [GetProductList(i1,i2,shape[1]) for i1 in l1 for i2 in l2]


def GetPartitionData(atomicCells,metaCells):
    """Provides various index lists used for domain decomposition.
    
    atomicCells: lists of indices corresponding to the atomic cells of the domain partition
    metaCells: list of which atomicCells should be joined to form one cell
    
    Returns:
    cells: lists of indices of the full cells, obtained by merging the corresponding atomicCells lists
    children: list which atomicCells are joined to form a specific full cell
        (this is essentially a copy of the metaCells input argument)
    childrenIndices: which indices in what full cell correspond to indices of the original atomic cells?
        for each atomic cell it contains the parent full cell p_i and start and end index a_i,b_i of which part of the full cell
        comes from the given atomic cell.
    """
    # indices of cells in final partition
    resultCells=[]
    # indices of atomic cells in each final partition cell
    resultChildren=[]
    # indices of elements belonging to each atomic cell
    resultChildrenIndices=[None for i in range(len(atomicCells))]
    
    for iMeta,metaCell in enumerate(metaCells):
        resultChildren.append(metaCell)
        
        indexOffset=0
        currentCell=[]
        for i in metaCell:
            currentCell+=atomicCells[i]
            resultChildrenIndices[i]=[iMeta,indexOffset,indexOffset+len(atomicCells[i])]
            indexOffset+=len(atomicCells[i])
        resultCells.append(currentCell)

        
    return (resultCells,resultChildren,resultChildrenIndices)

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
# one half-step of the domain decomposition algorithm
# old serial implementation, deprecated

#def DomDecIteration(muXList,posXList,alphaList,muY,posY,muYAtomicList,partitionData,eps,SolveOnCell,
#            verbose=False,verboseInterval=10):
#    """One half-iteration of the domain decomposition algorithm.
#    Input:
#    muXList: X-marginals on each X-cell
#    posXList: position of X-masses for each X-cell
#    alphaList: current values of duals on each X-cell
#    muY: full Y marginal
#    posY: full Y positions
#    muYAtomicList: for each atomic X-cell these are the corresponding Y-marginals
#    partitionData: info about cells as returned by GetPartitionData.
#    eps: regularization strength
#    SolveOnCell: function to use for updating the coupling on each cell
#    
#    Returns: nothing. Will directly overwrite progress into
#    alphaList and muYAtomicList"""
#    
#    nCells=len(muXList)
#    for i in range(nCells):
#        if verbose:
#            if i%verboseInterval==0:
#                print("\t{:d}".format(i))

#        # compute Y marginal of cell by summing atomic Y marginals
#        muYCell=np.zeros_like(muY)
#        for j in partitionData[1][i]:
#            muYCell+=muYAtomicList[j]

#        # solve on cell
#        result=SolveOnCell(muXList[i],muYCell,posXList[i],posY,muXList[i],muY,alphaList[i],eps)

#        # update partial dual variable
#        alphaList[i]=result[1]

#        # update atomic Y marginals
#        pi=result[3]

#        for j in partitionData[1][i]:
#            muYAtomicList[j]=Common.GetPartialYMarginal(pi,range(*partitionData[2][j][1:3]))
#            


#def DomDecIteration_SparseY(\
#        muXList,posXList,alphaList,muY,posY,muYAtomicList,partitionData,eps,SolveOnCell,\
#        verbose=False,verboseInterval=10,sparseThreshold=-1.):
#    """One half-iteration of the domain decomposition algorithm.
#    Input:
#    muXList: X-marginals on each X-cell
#    posXList: position of X-masses for each X-cell
#    alphaList: current values of duals on each X-cell
#    muY: full Y marginal
#    posY: full Y positions
#    muYAtomicList: for each atomic X-cell these are the corresponding Y-marginals
#    partitionData: info about cells as returned by GetPartitionData.
#    eps: regularization strength
#    SolveOnCell: function to use for updating the coupling on each cell
#    
#    sparseThreshold: threshold for truncation to sparse vectors of new y marginals
#    
#    Returns: nothing. Will directly overwrite progress into
#    alphaList and muYAtomicList"""
#    
#    nCells=len(muXList)
#    for i in range(nCells):
#        if verbose:
#            if i%verboseInterval==0:
#                print("\t{:d}".format(i))

#        # compute Y marginal of cell by summing atomic Y marginals
#        j=partitionData[1][i][0]
#        muYCell=muYAtomicList[j].copy()
#        for j in partitionData[1][i][1:]:
#            muYCell+=muYAtomicList[j]

#        # solve on cell
#        result=SolveOnCell(muXList[i],muYCell,posXList[i],posY,muXList[i],muY,alphaList[i],eps)
#        #(msg,alpha,beta,pi)

#            # update partial dual variable
#        alphaList[i]=result[1]

#        # extract new atomic muY
#        pi=result[3]
#        for j in partitionData[1][i]:
#            muYAtomicList[j]=Common.getSparseVector(Common.GetPartialYMarginal(pi,range(*partitionData[2][j][1:3])),\
#                    sparseThreshold,\
#                    subIndices=muYCell.indices,subShape=muY.shape[0])

def Iterate(\
        muY,posY,eps,\
        partitionDataCompCells,partitionDataCompCellIndices,\
        muYAtomicDataList,muYAtomicIndicesList,\
        muXList,posXList,alphaList,betaDataList,betaIndexList,\
        SinkhornSubSolver="LogSinkhorn", SinkhornError=1E-4,\
        SinkhornErrorRel=False, SinkhornMaxIter = None,\
        SinkhornInnerIter = 100): # Introducing bounding box as an additional argument
        #introducing the option to remove epsilon scaling, leave const_iterations at 0 to keep the scalling

    nCells=len(muXList)
    keops = 0

    if SinkhornSubSolver=="LogSinkhorn":
        SolveOnCell=SolveOnCell_LogSinkhorn
    elif SinkhornSubSolver=="SparseSinkhorn":
        SolveOnCell=SolveOnCell_SparseSinkhorn
    else:
        SolveOnCell=SinkhornSubSolver    
        
    for i in range(nCells):
        resultAlpha,resultBeta,resultMuYAtomicDataList,muYCellIndices=DomDecIteration_SparseY(SolveOnCell,SinkhornError,SinkhornErrorRel,muY,posY,eps,\
                muXList[i],posXList[i],alphaList[i],\
                [muYAtomicDataList[j] for j in partitionDataCompCells[i]],\
                [muYAtomicIndicesList[j] for j in partitionDataCompCells[i]],\
                partitionDataCompCellIndices[i]\
                )
        alphaList[i]=resultAlpha
        betaDataList[i]=resultBeta
        betaIndexList[i]=muYCellIndices.copy()
        for jsub,j in enumerate(partitionDataCompCells[i]):
            muYAtomicDataList[j]=resultMuYAtomicDataList[jsub]
            muYAtomicIndicesList[j]=muYCellIndices.copy()
    

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
# solve sub problems

def getPi(c,alpha,beta,rhoX,rhoY,eps):
    #pi=-c+np.einsum(alpha,[0],np.ones_like(beta),[1],[0,1])+np.einsum(np.ones_like(alpha),[0],beta,[1],[0,1])
    pi=-c+alpha.reshape((-1,1))+beta.reshape((1,-1))
    pi=np.exp(pi/eps)
    pi=np.einsum(rhoX,[0],rhoY,[1],pi,[0,1],[0,1])
    return pi

#-----------------------------------------------------------------------------------------------------------------------------------------

def SolveOnCell_LogSinkhorn(muX,subMuY,subY,posX,posY,rhoX,rhoY,alphaInit,eps,SinkhornError=1E-4,SinkhornErrorRel=False,YThresh=1E-14):
    
    subPosY=posY[subY].copy()
    subRhoY=rhoY[subY].copy()


    alpha=alphaInit.copy()
    beta=np.zeros_like(subMuY)
    #c=Common.getEuclideanCostFunction(posX,subPosY,p=2.)
    #cT=c.transpose().copy()
    xres=posX.shape[0]
    yres=subY.shape[0]
    c,cT=LogSinkhorn.getEuclideanCost(posX,subPosY)
    c=c.reshape((xres,yres))
    cT=cT.reshape((yres,xres))
    
    if SinkhornErrorRel:
        effectiveError=SinkhornError*np.sum(muX)
    else:
        effectiveError=SinkhornError
    
    # dummy return for time measurements
    #return None
    
    msg=LogSinkhorn.iterateUntilError(alpha,beta,c,cT,muX,subMuY,rhoX,subRhoY,eps,10000,20,effectiveError)
    
    if msg==1:
        print("warning: {:d} : Sinkhorn did not converge to accuracy".format(msg))
    elif msg!=0:
        print("warning: {:d} : unspecified error".format(msg))

    pi=getPi(c,alpha,beta,rhoX,subRhoY,eps)

    return (msg,alpha,beta,pi)

def SolveOnCell_SparseSinkhorn(muX,subMuY,subY,posX,posY,rhoX,rhoY,alphaInit,eps,SinkhornError=1E-4,SinkhornErrorRel=False,YThresh=1E-14,\
        autoEpsFix=True,verbose=True):
    """Runs the Sinkhorn algorithm on the entropic Wasserstein-2 transport problem
    between (muX,posX) and (muY,posY) with regularization eps and initial dual variable
    alphaInit on the X-side.
    
    Entries of the Y-marginal with mass less than YThresh are ignored.
    Sinkhorn algorithm terminates when L1 error is below SinkhornError.

    If autoEpsFix==True the algorithm tries to fix epsilon when something goes wrong numerically.
    I.e. if the algorithm fails during the first run, eps is increased by a factor 2 until a result is obtained.
    Then eps is gradually decreased again until the original value.
    
    If test==1 some debug tests in the c++ code are done."""
    
    subPosY=posY[subY].copy()
    subRhoY=rhoY[subY].copy()

    if SinkhornErrorRel:
        effectiveError=SinkhornError*np.sum(muX)
    else:
        effectiveError=SinkhornError

    subMuYEff=subMuY/np.sum(subMuY)*np.sum(muX)
    
    result=CPPSinkhorn.SolveSinkhorn(muX,subMuYEff,posX,subPosY,rhoX,subRhoY,alphaInit,\
            effectiveError,eps,False,0)

    if autoEpsFix:
        epsExponent=0
        while result[0]!=0:
            # in case the algorithm failed (probably since eps is too small),
            # increase eps until it succeeds
            
            epsExponent+=1
            epsEff=eps*(2**epsExponent)
            
            result=CPPSinkhorn.SolveSinkhorn(muX,subMuY,posX,subPosY,rhoX,subRhoY,alphaInit,\
                    effectiveError,epsEff,False,0)

        if (epsExponent>0) and verbose:
            print("autoFix:\t{:d} ({:f},{:f})".format(epsExponent,posX[0,0],posX[0,1]))
        while epsExponent>0:
            # now start decreasing again
            alphaInitEff=result[1]
            epsExponent-=1
            epsEff=eps*(2**epsExponent)
            result=CPPSinkhorn.SolveSinkhorn(muX,subMuY,posX,subPosY,rhoX,subRhoY,alphaInitEff,\
                    effectiveError,epsEff,False,0)

            
            if result[0]!=0:
                raise ValueError(\
                        "Sinkhorn error on autoEpsFix: {:d}, epsExponent {:d}".format(result[0],epsExponent))


    else:
        raise ValueError("Sinkhorn error: {:d}".format(result[0]))

    # assemble kernel with rescaled Y indices
    resultKernel=scipy.sparse.csr_matrix((result[3],result[4],result[5]),\
            shape=(muX.shape[0],subMuY.shape[0]))
    return (result[0],result[1],result[2],resultKernel)

def DomDecIteration_SparseY(\
        SolveOnCell,SinkhornError,SinkhornErrorRel,muY,posY,eps,\
        muXCell,posXCell,alphaCell,muYAtomicListData,muYAtomicListIndices,partitionDataCompCellIndices\
        ):
    """Iterate a cell in one partition: combine corresponding atomic cells, solve subproblem and compute new atomic partial marginals."""
    
    # un-comment next line to measure pure time it takes for communication etc    
    #return (alphaCell,muYAtomicListData,muYAtomicListIndices)
        
    # compute Y marginal of cell by summing atomic Y marginals

    # old code using scipy.sparse arrays and their adding functionality
    #muYCell=muYAtomicList[0].copy()
    #for muYTerm in muYAtomicList[1:]:
    #    muYCell+=muYTerm
    
    # new code where sparse vectors are represented index and value list of non-zero entries, with custom c++ code for adding
    arrayAdder=LogSinkhorn.TSparseArrayAdder()
    for x,y in zip(muYAtomicListData,muYAtomicListIndices):
        arrayAdder.add(x,y)
    muYCellData,muYCellIndices=arrayAdder.getDataTuple()

    # another dummy return and dummy function call
    #SolveOnCell(muXCell,muYCellData,muYCellIndices,posXCell,posY,muXCell,muY,alphaCell,eps)
    #return (alphaCell,muYAtomicListData,muYAtomicListIndices[0])


    # solve on cell
    msg,resultAlpha,resultBeta,pi=SolveOnCell(muXCell,muYCellData,muYCellIndices,posXCell,posY,muXCell,muY,alphaCell,eps,SinkhornError,SinkhornErrorRel)


    # extract new atomic muY
    resultMuYAtomicDataList=[\
            Common.GetPartialYMarginal(pi,range(*indices))
            for indices in partitionDataCompCellIndices
            ]
            

    return (resultAlpha,resultBeta,resultMuYAtomicDataList,muYCellIndices)


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
# refine the atomic muY measures

## dense version, deprecated
#def GetRefinedAtomicYMarginals(muXL,muYL,muXLOld,muYLOld,parentsY,
#        atomicCells,atomicCellsOld,muYAtomicListOld,metaCellShape):
#    # dense implementation
#    
#    # old atomic cells are 2x2 clustering of new atomic cells
#    newCellChildren=GetPartitionIndices2D(metaCellShape,2,0)
#    # for each new atomic cell compute the old parent atomic cell
#    atomicCellParents=np.zeros((np.prod(metaCellShape),),dtype=np.int)
#    for i,children in enumerate(newCellChildren):
#        atomicCellParents[children]=i


#    # refinement of Y marginals

#    # each new atomic Y-marginal will be a rescaled and refined version of the parent atomic Y-marginal
#    # rescaling factor depends on how much mass the new atomic cell contains
#    # refinement is given by refinement of muYL-old to muYL-new
#    muYAtomicList=[\
#            muYL*(muYAtomicListOld[atomicCellParents[i]]/muYLOld)[parentsY]\
#            *np.sum(muXL[atomicCells[i]])/np.sum(muXLOld[atomicCellsOld[atomicCellParents[i]]])\
#            for i in range(len(atomicCells))]
#    
#    return muYAtomicList

## first sparse version, but does a lot of work multiple times in each new atomic cell, when it should only be done once in each old atomic cell
#def GetRefinedAtomicYMarginals_SparseY(muXL,muYL,muXLOld,muYLOld,parentsYL,
#        atomicCells,atomicCellsOld,muYAtomicListOld,metaCellShape,thresh=1E-15):

#    yresOld=muYLOld.shape[0]

#    # list of children of each coarse node
#    childrenYLOld=[[] for i in range(yresOld)]
#    for i,parent in enumerate(parentsYL):
#        childrenYLOld[parent].append(i)

#    # old atomic cells are 2x2 clustering of new atomic cells
#    newCellChildren=GetPartitionIndices2D(metaCellShape,2,0)
#    # for each new atomic cell compute the old parent atomic cell
#    atomicCellParents=np.zeros((np.prod(metaCellShape),),dtype=np.int)
#    for i,children in enumerate(newCellChildren):
#        atomicCellParents[children]=i


#    def refineMuYAtomic(i):
#        muYAtomicOld=muYAtomicListOld[atomicCellParents[i]]

#        yres=muYL.shape[0]
#        xMassScaleFactor=np.sum(muXL[atomicCells[i]])/np.sum(muXLOld[atomicCellsOld[atomicCellParents[i]]])
#        indicesFine=[]
#        dataFine=[]
#        yMassScaleFactors=muYAtomicOld.data/muYLOld[muYAtomicOld.indices]
#        for d,y in zip(yMassScaleFactors,muYAtomicOld.indices):
#            indicesFine+=childrenYLOld[y]
#            dataFine.append(muYL[childrenYLOld[y]]*d*xMassScaleFactor)
#        indicesFine=np.array(indicesFine,dtype=np.int32)
#        dataFine=np.hstack(dataFine)
#        indptrFine=np.array([0,indicesFine.shape[0]],dtype=np.int32)
#        result=Common.truncateSparseVector(\
#                scipy.sparse.csr_matrix((dataFine,indicesFine,indptrFine),shape=(1,yres)),
#                thresh=1E-15)
#        result.sort_indices()
#        return result
#    
#    return [refineMuYAtomic(i) for i in range(len(atomicCells))]


# substantially improved version, but rescaling with muYL at fine level can still be done per old coarse atomic cell.
#def GetRefinedAtomicYMarginals_SparseY(muYL,muYLOld,parentsYL,\
#        atomicCellMasses,atomicCellMassesOld,\
#        atomicCells,atomicCellsOld,muYAtomicListOld,metaCellShape,thresh=1E-15):

#    yresOld=muYLOld.shape[0]
#    yres=muYL.shape[0]
#    
#    # list of children of each coarse node
#    childrenYLOld=[[] for i in range(yresOld)]
#    for i,parent in enumerate(parentsYL):
#        childrenYLOld[parent].append(i)

#    # old atomic cells are 2x2 clustering of new atomic cells
#    newCellChildren=GetPartitionIndices2D(metaCellShape,2,0)
#    # for each new atomic cell compute the old parent atomic cell
#    atomicCellParents=np.zeros((np.prod(metaCellShape),),dtype=np.int)
#    for i,children in enumerate(newCellChildren):
#        atomicCellParents[children]=i

#    # compute refined muYAtomicOld/muYLOld densities for each old atomic cell
#    # then on each child cell can simply rescale this one, but all tricky sparsity and refinement operations
#    # are only done once
#    def refineMuYAtomicOld(i):
#        muYAtomicOld=muYAtomicListOld[i]
#        
#        indicesFine=[]
#        dataFine=[]
#        yMassScaleFactors=muYAtomicOld.data/muYLOld[muYAtomicOld.indices]
#        for d,y in zip(yMassScaleFactors,muYAtomicOld.indices):
#            indicesFine+=childrenYLOld[y]
#            dataFine.append(np.full((len(childrenYLOld[y]),),d,dtype=np.double))
#        indicesFine=np.array(indicesFine,dtype=np.int32)
#        dataFine=np.hstack(dataFine)
#        indptrFine=np.array([0,indicesFine.shape[0]],dtype=np.int32)
#        result=Common.truncateSparseVector(\
#                scipy.sparse.csr_matrix((dataFine,indicesFine,indptrFine),shape=(1,yres)),
#                thresh=1E-15)
#        result.sort_indices()
#        return result
#        
#        
#    def refineMuYAtomic(i,muYAtomicDummy):
#        xMassScaleFactor=atomicCellMasses[i]/atomicCellMassesOld[atomicCellParents[i]]
#        result=muYAtomicDummy.copy()
#        result.data*=xMassScaleFactor*muYL[result.indices]
#        return result
#    
#    preMuYAtomicList=[refineMuYAtomicOld(i) for i in range(len(atomicCellsOld))]
#    return [refineMuYAtomic(i,preMuYAtomicList[atomicCellParents[i]]) for i in range(len(atomicCells))]


def GetRefinedAtomicYMarginals_SparseY(muYL,muYLOld,parentsYL,\
        atomicCellMasses,atomicCellMassesOld,\
        atomicCells,atomicCellsOld,\
        muYAtomicDataListOld,muYAtomicIndicesListOld,\
        metaCellShape,thresh=1E-15):

    yresOld=muYLOld.shape[0]
    yres=muYL.shape[0]
    
    # list of children of each coarse node
    childrenYLOld=[[] for i in range(yresOld)]
    for i,parent in enumerate(parentsYL):
        childrenYLOld[parent].append(i)

    # old atomic cells are 2x2 clustering of new atomic cells
    newCellChildren=GetPartitionIndices2D(metaCellShape,2,0)
    # for each new atomic cell compute the old parent atomic cell
    atomicCellParents=np.zeros((np.prod(metaCellShape),),dtype=np.int)
    for i,children in enumerate(newCellChildren):
        atomicCellParents[children]=i
        
        
    preMuYAtomicList=[refineMuYAtomicOld(muYL,muYLOld,childrenYLOld,\
            muYAtomicDataListOld[i],muYAtomicIndicesListOld[i]) for i in range(len(atomicCellsOld))]

    resultData=[None for i in range(len(atomicCells))]
    resultIndices=[None for i in range(len(atomicCells))]

    for i,dat in enumerate(preMuYAtomicList):
        # dat=(dataFine,indicesFine)
        for j in newCellChildren[i]:
            resultData[j]=dat[0]*atomicCellMasses[j]/atomicCellMassesOld[i]
            resultIndices[j]=dat[1].copy()

    
    return [resultData,resultIndices]

# compute refined muYAtomicOld/muYLOld densities for each old atomic cell
# then on each child cell can simply rescale this one by x cell weights, but all tricky sparsity and refinement operations are only done once
def refineMuYAtomicOld(muYL,muYLOld,childrenYLOld,\
        muYAtomicDataOld,muYAtomicIndicesOld):


    indicesFine=[]
    dataFine=[]
    yMassScaleFactors=muYAtomicDataOld/muYLOld[muYAtomicIndicesOld]
    for d,y in zip(yMassScaleFactors,muYAtomicIndicesOld):
        indicesFine+=childrenYLOld[y]
        dataFine.append(np.full((len(childrenYLOld[y]),),d,dtype=np.double))
    indicesFine=np.array(indicesFine,dtype=np.int32)
    dataFine=np.hstack(dataFine)
    dataFine*=muYL[indicesFine]
    
    ordering=np.argsort(indicesFine)
    indicesFine=indicesFine[ordering]
    dataFine=dataFine[ordering]
    
    return (dataFine,indicesFine)


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
# upon terminating the sinkhorn iterations, the X-marginals are usually not exactly satisfied
# this leads to differences in mass of the atomic muY measures
# the following functions provide a simply method for balancing the atomic muY measures again

def getPairwiseDelta(d1,d2):
    return min(max(d1,0),max(-d2,0))-min(max(-d1,0),max(d2,0))

# sparse version that assumes that muYAtomicListSub are sparse_csr objects
# with potentially different indices
#def BalanceMeasuresMulti(muYAtomicListSub,atomicCellMassesSub,threshStep=1E-16,threshTerminate=1E-10):
#    # dummy return to test pure communication time
#    #return (0,muYAtomicListSub)
#    nCells=len(muYAtomicListSub)
#    
#    massDeltasSub=np.array([np.sum(a)-b for a,b in zip(muYAtomicListSub,atomicCellMassesSub)])
#    
#    if nCells==1:
#        return (0,muYAtomicListSub)

#    for i in range(nCells):
#        for j in range(i+1,nCells):
#            delta=getPairwiseDelta(massDeltasSub[i],massDeltasSub[j])
#            if delta>0:
#                delta=min(delta,np.sum(muYAtomicListSub[i]))
#            else:
#                delta=max(delta,-np.sum(muYAtomicListSub[j]))
#            if np.abs(delta)>threshStep:
#                LogSinkhorn.balanceMeasures_sparse(\
#                        muYAtomicListSub[i].data,muYAtomicListSub[j].data,\
#                        muYAtomicListSub[i].indices,muYAtomicListSub[j].indices,\
#                        delta)
#                massDeltasSub[i]-=delta
#                massDeltasSub[j]+=delta
#            if np.sum(np.abs(massDeltasSub))<threshTerminate:
#                return (0,muYAtomicListSub)
#    return (1,muYAtomicListSub)

#def BalanceMeasuresMultiAll(muYAtomicList,atomicCellMasses,partitionData,verbose=False):
#    for i in range(len(partitionData[1])):
#        muYAtomicListSub=[muYAtomicList[j] for j in partitionData[1][i]]
#        atomicCellMassesSub=[atomicCellMasses[j] for j in partitionData[1][i]]
#        msg,muYAtomicData,massDeltaData=BalanceMeasuresMulti(muYAtomicListSub,atomicCellMassesSub)

#        for jsub,j in enumerate(partitionData[1][i]):
#            muYAtomicList[j]=muYAtomicData[jsub]
#        if (msg!=0) and (verbose):
#            print("warning: failed to balance measures in cell {:d}".format(i))


# simplified version where muYAtomicListSub are only the data fields of the csr matrices and it is assumed that their indices are the same
def BalanceMeasuresMulti(muYAtomicListSub,atomicCellMassesSub,threshStep=1E-16,threshTerminate=1E-10):
    # dummy return to test pure communication time
    #return (0,muYAtomicListSub)
    nCells=len(muYAtomicListSub)
    
    massDeltasSub=np.array([np.sum(a)-b for a,b in zip(muYAtomicListSub,atomicCellMassesSub)])
    # print(massDeltasSub)
    if nCells==1:
        return (0,muYAtomicListSub)

    for i in range(nCells):
        for j in range(i+1,nCells):
            delta=getPairwiseDelta(massDeltasSub[i],massDeltasSub[j])
            if delta>0:
                delta=min(delta,np.sum(muYAtomicListSub[i]))
            else:
                delta=max(delta,-np.sum(muYAtomicListSub[j]))
            if np.abs(delta)>threshStep:
                LogSinkhorn.balanceMeasures(\
                        muYAtomicListSub[i],muYAtomicListSub[j],\
                        delta)
                massDeltasSub[i]-=delta
                massDeltasSub[j]+=delta
            if np.sum(np.abs(massDeltasSub))<threshTerminate:
                return (0,muYAtomicListSub)
    return (1,muYAtomicListSub)

def GetActualYMarginal(muYAtomicIndicesList, muYAtomicDataList, N):
    """
    Compute current Y marginal based on basic cell marginals.
    """
    piY = np.zeros(N)
    for (indices, data) in zip(muYAtomicIndicesList, muYAtomicDataList):
        piY[indices] += data
    return piY

def BalanceMeasuresMultiAll(muYAtomicDataList,atomicCellMasses,partitionDataCompCells,verbose=False,threshTerminate=1e-6):
    for i in range(len(partitionDataCompCells)):
        muYAtomicListSub=[muYAtomicDataList[j] for j in partitionDataCompCells[i]]
        atomicCellMassesSub=atomicCellMasses[partitionDataCompCells[i]]
        msg,muYAtomicData=BalanceMeasuresMulti(muYAtomicListSub,atomicCellMassesSub,threshTerminate=threshTerminate)

        for jsub,j in enumerate(partitionDataCompCells[i]):
            muYAtomicDataList[j]=muYAtomicData[jsub]
        if (msg!=0) and (verbose):
            print("warning: failed to balance measures in cell {:d}".format(i))


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
# stitching together dual variables, creating one full dual potential
# the following code is also super hacky and essentially only works on 2D grids, partitioned in the way
# it is done above, and probably only if grid and cell sizes are powers of two

# in the domain decomposition algorithm there is no complete dual variable
# all dual alpha variables are only defined on the corresponding cells
# but upon convergence, all of them should only differ by constants on the overlapping regions
# the following implements a simple method to stich these together

# this is very useful for creating a good initial guess for alpha after refining the grid

def getAlphaField(alphaList,cells,n):
    """alphaList: values of alpha on each cell
    cells: list of indices of each cell on complete domain
    n: size of complete domain
    
    This function writes the values of alphaList into the the corresponding cells of the full domain and returns the array.
    The result is a function defined on the full domain, but with no consistency at the cell boundaries."""
    alphaField=np.zeros(n,dtype=np.double)
    for a,c in zip(alphaList,cells):
        alphaField[c]=a
    return alphaField

def getAlphaGraph(alphaDiff,metaCellShape,cellSize,muX=None):
    """Takes the difference between the two alphaFields of the two staggered grids and computes
    everything necessary to remove the offsets of alphaAField and turn it into a consistent global dual variable.
    In detail, it does the following:
    i) on each atomic cell compute the mean of alphaDiff
    this is interpreted as edge weight of a bipartite graph
    where the first vertex set are cells of partition A, the second vertex set are the cells of partition B
    and there is an edge between two cels iff they overlap
    
    ii) Upon perfect convergence, one can show that the edge weights sum to zero along cycles.
    This can be used to define a potential, by "integrating" / summing along paths of this graph.
    So the global offsets between the alphaA values could be computed as difference between the alphaAField and the potential.
    This function just assumes that sums along cycles are zero and uses canonical paths on a 2D grid to define the potential.
    The potential is set to zero at the top left corner. Then it is defined at all other locations by going east first,
    then down."""
    dim=len(metaCellShape)
    
    # partition alphaDiff into cells of size cellSize x cellSize:
    # overall shape will be: metaCellShape[0] x cellSize x metaCellShape[1] x cellSize ...
    graphShape=tuple()
    for i in metaCellShape:
        graphShape+=(i,cellSize)
    alphaGraph=alphaDiff.reshape(graphShape)
    
    # now take mean over each atomic cell block
    sumAxis=tuple([2*i+1 for i in range(dim)])
    
    if muX is None:
        # no reference measure for averging is given
        alphaGraph=np.mean(alphaGraph,axis=sumAxis)
    else:
        muRef=muX.reshape(graphShape)
        alphaGraph=np.sum(alphaGraph*muRef,axis=sumAxis)/np.sum(muRef,axis=sumAxis)


    # V1: old integration along paths: first horizontal, then vertical    
#    # now sum differences along certain paths to get offsets between different cells of partitionA:
#    # THIS ONLY WORKS FOR DIM=2
#    if dim!=2:
#        raise ValueError("alpha stitching was only implemented for dim=2")
#    # horizontal:
#    alphaGraphH=(alphaGraph[::2,1:-1]).reshape((metaCellShape[0]//2,-1,2))
#    alphaGraphH[:,:,::2]*=-1
#    alphaGraphH=np.sum(alphaGraphH,axis=2)
#    alphaGraphH=np.pad(np.cumsum(alphaGraphH,axis=1),((0,0),(1,0)),'constant')
#    alphaGraphH[:,:]=alphaGraphH[0,:]
#    # vertical:
#    alphaGraphV=(alphaGraph[1:-1,::2]).reshape((-1,2,metaCellShape[1]//2))
#    alphaGraphV[:,::2,:]*=-1
#    alphaGraphV=np.sum(alphaGraphV,axis=1)
#    alphaGraphV=np.pad(np.cumsum(alphaGraphV,axis=0),((1,0),(0,0)),'constant')
#    
#    return alphaGraphH+alphaGraphV

#    # V2: new integration: weighted integration over all combinatorial paths that do not make left or upwards steps
#    # value given by weighted sum of value from above and value from left, weights given by coordinates
#    edgeWeightArrayV=alphaGraph[1:-1].reshape((metaCellShape[0]//2-1,2,metaCellShape[1]//2,2)).copy()
#    edgeWeightArrayV[:,0,:,:]*=-1
#    edgeWeightArrayV=np.mean(edgeWeightArrayV,axis=3)
#    edgeWeightArrayV=np.sum(edgeWeightArrayV,axis=1)

#    edgeWeightArrayH=alphaGraph[:,1:-1].reshape((metaCellShape[0]//2,2,metaCellShape[1]//2-1,2)).copy()
#    edgeWeightArrayH[:,:,:,0]*=-1
#    edgeWeightArrayH=np.sum(edgeWeightArrayH,axis=3)
#    edgeWeightArrayH=np.mean(edgeWeightArrayH,axis=1)

#    #edgeWeightArrayH=np.mean(edgeWeightArrayH,axis=(1,3))
#    alphaGraph=np.zeros([x//2 for x in metaCellShape])
#    for x in range(metaCellShape[0]//2):
#        for y in range(metaCellShape[1]//2):
#            if x==0 and y>0:
#                alphaGraph[x,y]=alphaGraph[x,y-1]+edgeWeightArrayH[x,y-1]
#            elif y==0 and x>0:
#                alphaGraph[x,y]=alphaGraph[x-1,y]+edgeWeightArrayV[x-1,y]
#            elif x>0 and y>0:
#                alphaGraph[x,y]=(alphaGraph[x-1,y]+edgeWeightArrayV[x-1,y])*(x/(x+y))+\
#                        (alphaGraph[x,y-1]+edgeWeightArrayH[x,y-1])*(y/(x+y))
#    return alphaGraph

    # V3: "Helmholtz decomposition": let A be discrete gradient operator from grid graph vertices to edges
    # let b be edge lengths obtained by comparing the alpha (i.e. eps log q), represented by edgeWeightArrayV and edgeWeightArrayH above
    # then let x be the approximate alphaGraph potential. obtain it by minimizing ||Ax-b||^2.
    # this distributes the error due to non-integrability of b more evenly and thus yields a better dual score

    # as before: compute edge weights    
    edgeWeightArrayV=alphaGraph[1:-1].reshape((metaCellShape[0]//2-1,2,metaCellShape[1]//2,2)).copy()
    edgeWeightArrayV[:,0,:,:]*=-1
    edgeWeightArrayV=np.mean(edgeWeightArrayV,axis=3)
    edgeWeightArrayV=np.sum(edgeWeightArrayV,axis=1)

    edgeWeightArrayH=alphaGraph[:,1:-1].reshape((metaCellShape[0]//2,2,metaCellShape[1]//2-1,2)).copy()
    edgeWeightArrayH[:,:,:,0]*=-1
    edgeWeightArrayH=np.sum(edgeWeightArrayH,axis=3)
    edgeWeightArrayH=np.mean(edgeWeightArrayH,axis=1)
    
    
    # get discrete gradient operators
    gradH,gradV=Common.getDiscreteGradient(*[x//2 for x in metaCellShape])
    # combine
    A=scipy.sparse.vstack((gradH,gradV))
    # remove first column to make A injective (corresponds to global constant of potential)
    A=A[:,1:]
    
    # setup linear equation for optimality condition
    AT=A.transpose().tocsr()
    ATA=AT.dot(A).tocsr()
    b=np.hstack((edgeWeightArrayH.ravel(),edgeWeightArrayV.ravel()))
    ATb=AT.dot(b)

    # solve and plug into
    alphaGraphPre=scipy.sparse.linalg.spsolve(ATA,ATb)
    alphaGraph=np.zeros([x//2 for x in metaCellShape]).ravel()
    alphaGraph[1:]=alphaGraphPre
    alphaGraph=alphaGraph.reshape([x//2 for x in metaCellShape])
    
    return alphaGraph


def getAlphaFieldEven(alphaAList,alphaBList,cellsA,cellsB,shapeXL,metaCellShape,cellSize,muX=None,requestAlphaGraph=False):
    """Uses getAlphaField and getAlphaGraph to compute one global dual variable alpha from alphaAList and alphaBList."""
    alphaFieldA=getAlphaField(alphaAList,cellsA,np.prod(shapeXL))
    alphaFieldB=getAlphaField(alphaBList,cellsB,np.prod(shapeXL))
    alphaDiff=alphaFieldA-alphaFieldB
    alphaGraph=getAlphaGraph(alphaDiff,metaCellShape,cellSize,muX)
    alphaFieldEven=alphaFieldA.copy()
    for a,c in zip(alphaGraph.ravel(),cellsA):
        alphaFieldEven[c]-=a
    
    if requestAlphaGraph:
        return (alphaFieldEven,alphaGraph)
    
    return alphaFieldEven


def glueBetaList(betaDataList,betaIndexList,res,offsets=None,muYList=None,muY=None):
    result=np.zeros((res,),dtype=np.double)
    
    if muYList is None:
        for i,(dat,ind) in enumerate(zip(betaDataList,betaIndexList)):
            result[ind]=dat
            if offsets is not None:
                result[ind]+=offsets[i]
    else:
        for i,(dat,ind,muYCell) in enumerate(zip(betaDataList,betaIndexList,muYList)):
            result[ind]+=dat*muYCell
            if offsets is not None:
                result[ind]+=offsets[i]*muYCell
        result=result/muY
    return result

def getPrimalInfos(muY,posY,posXList,muXList,alphaList,betaDataList,betaIndexList,eps,getMuYList=False):
    scorePrimalUnreg=0.
    scorePrimal=0.
    errorMargX=0.
    errorMargY=0.
    
    if getMuYList:
        muYList=[]

    margY=np.zeros_like(muY)

    for i in range(len(muXList)):
        posYcell=posY[betaIndexList[i]].copy()
        xresCell=posXList[i].shape[0]
        yresCell=posYcell.shape[0]
        c,cT=LogSinkhorn.getEuclideanCost(posXList[i],posYcell)
        #cEff=c.reshape((xresCell,yresCell))\
        #        -np.einsum(alphaList[i],[0],np.ones((yresCell,),dtype=np.double),[1],[0,1])\
        #        -np.einsum(np.ones((xresCell,),dtype=np.double),[0],betaDataList[i],[1],[0,1])

        cEff=c.reshape((xresCell,yresCell))-alphaList[i].reshape((-1,1))-betaDataList[i].reshape((1,-1))


        piCell=np.einsum(np.exp(-cEff/eps),[0,1],muXList[i],[0],muY[betaIndexList[i]],[1],[0,1])

        scorePrimalUnreg+=np.sum(piCell.ravel()*c)
        scorePrimal+=np.einsum(piCell,[0,1],alphaList[i],[0],[])\
                +np.einsum(piCell,[0,1],betaDataList[i],[1],[])\
                -eps*np.sum(piCell)
                
        
        errorMargX+=np.sum(np.abs(np.sum(piCell,axis=1)-muXList[i]))
        margY[betaIndexList[i]]+=np.sum(piCell,axis=0)
        
        if getMuYList:
            muYList.append(np.sum(piCell,axis=0))


    errorMargY=np.sum(np.abs(muY-margY))
    
    result={"scorePrimal":scorePrimal, "scorePrimalUnreg":scorePrimalUnreg,"errorMargX":errorMargX,"errorMargY":errorMargY}
    
    if getMuYList:
        return (result,muYList)
    
    return result
        
    
def getHierarchicalKernel(MultiScaleSetupX,MultiScaleSetupY,dim,hierarchy_depth,eps,alpha,beta,nIter=0):
    # multi-scale evaluation of dual score, via Sinkhorn solver object
    #MultiScaleSetupX.setupDuals()
    #MultiScaleSetupY.setupDuals()

    # which cost function to use?
    costFunction=MultiScaleOT.THierarchicalCostFunctionProvider_SquaredEuclidean(
            MultiScaleSetupX,MultiScaleSetupY)

    # eps scaling
    epsScalingHandler=MultiScaleOT.TEpsScalingHandler()
    epsScalingHandler.setupGeometricSingleLayer(hierarchy_depth+1,eps,eps,0)


    # various other solver parameters
    cfg=MultiScaleOT.TSinkhornSolverParameters()
    cfg.maxIterations=100000
    cfg.innerIterations=100
    cfg.maxAbsorptionLoops=100
    cfg.absorption_scalingBound=1E3
    cfg.absorption_scalingLowerBound=1E3
    cfg.truncation_thresh=1E-20
    cfg.refineKernel=False


    MultiScaleSetupX.setDual(alpha,hierarchy_depth)
    MultiScaleSetupY.setDual(beta,hierarchy_depth)

    # create the actual solver object
    SinkhornSolver=MultiScaleOT.TSinkhornSolverStandard(epsScalingHandler,
            hierarchy_depth,hierarchy_depth,1E-4,
            MultiScaleSetupX,MultiScaleSetupY,costFunction,cfg)

    SinkhornSolver.changeEps(eps)
    SinkhornSolver.changeLayer(hierarchy_depth)

    SinkhornSolver.generateKernel()
    if nIter==0:
        
        dat=SinkhornSolver.getKernelCSRDataTuple()
        return dat
    else:
        SinkhornSolver.iterate(nIter)
        SinkhornSolver.absorb()
        SinkhornSolver.generateKernel()
        

        dat=SinkhornSolver.getKernelCSRDataTuple()
        
        newAlpha=MultiScaleSetupX.getDual(hierarchy_depth)
        newBeta=MultiScaleSetupY.getDual(hierarchy_depth)
        
        return (dat,newAlpha,newBeta)

from numpy.ma.core import array

# def bounding_Box_2D(data,index,matrix_size):
#   x_size,y_size = matrix_size
#   cartesian_index_x,cartesian_index_y = index // y_size, index % y_size

#   left = np.min(cartesian_index_x)
#   right = np.max(cartesian_index_x)
#   lower = np.min(cartesian_index_y)
#   upper = np.max(cartesian_index_y)
#   box_width = right - left + 1
#   box_hight = upper - lower + 1 

#   box_data = np.zeros((box_width,box_hight))
#   box_data[cartesian_index_x - left,cartesian_index_y - lower] = data

#   box_index = [0] * (box_width * box_hight)
#   box_index = (np.where(box_data>=0)[0] + left)*y_size + np.where(box_data>=0)[1] + lower

#   box_data = box_data.flatten()

#   return box_data,box_index,[box_width,box_hight]

# help funktion for the BatchboundingBox 
# it computes the dimensions of indiividual boundingboxes and converts the Indicies
