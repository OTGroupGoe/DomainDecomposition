import numpy as np
import scipy
import scipy.sparse

import lib.Common as Common
import lib.DomainDecomposition as DomDec
import lib.MPIParallelMap as ParallelMap



def ParallelIterate(comm,\
        muY,posY,eps,\
        partitionDataCompCells,partitionDataCompCellIndices,\
        muYAtomicDataList,muYAtomicIndicesList,\
        muXList,posXList,alphaList,betaDataList,betaIndexList,\
        SinkhornSubSolver="LogSinkhorn", SinkhornError=1E-4, SinkhornErrorRel=False,\
        MPIchunksize=1, MPIprobetime=None):


    nCells=len(muXList)

    if SinkhornSubSolver=="LogSinkhorn":
    	SolveOnCell=DomDec.SolveOnCell_LogSinkhorn
    elif SinkhornSubSolver=="SparseSinkhorn":
    	SolveOnCell=DomDec.SolveOnCell_SparseSinkhorn
    else:
        SolveOnCell=SinkhornSubSolver

    argsGlobal=[SolveOnCell,SinkhornError,SinkhornErrorRel,muY,posY,eps]

    def argList(i):
        return \
            [muXList[i],posXList[i],alphaList[i],\
            [muYAtomicDataList[j] for j in partitionDataCompCells[i]],\
            [muYAtomicIndicesList[j] for j in partitionDataCompCells[i]],\
            partitionDataCompCellIndices[i]\
            ]

    def callReturn(i,dat):
        # dat=(resultAlpha,resultMuYAtomicDataList,resultMuYAtomicIndicesList)
        alphaList[i]=dat[0]
        betaDataList[i]=dat[1]
        betaIndexList[i]=dat[3].copy()
        for jsub,j in enumerate(partitionDataCompCells[i]):
            muYAtomicDataList[j]=dat[2][jsub]
            muYAtomicIndicesList[j]=dat[3].copy()
        

    ParallelMap.ParallelMap(comm,DomDec.DomDecIteration_SparseY,argList,argsGlobal,\
            callableArgList=True, callableArgListLen=nCells, callableReturn=callReturn,\
            chunksize=MPIchunksize, probetime=MPIprobetime)




###############################################################################################################################
# mass balancing between atomic marginals


def ParallelBalanceMeasures(comm,\
        muYAtomicDataList,atomicCellMasses,partitionDataCompCells,\
        verbose=False,\
        MPIchunksize=1, MPIprobetime=None):

    def argList(i):
        return \
            [
                    [muYAtomicDataList[j] for j in partitionDataCompCells[i]],
                    atomicCellMasses[partitionDataCompCells[i]]
            ]

    def callReturn(i,dat):
        # dat=(msg,muYAtomicData)
        for jsub,j in enumerate(partitionDataCompCells[i]):
            muYAtomicDataList[j]=dat[1][jsub]
        if (dat[0]!=0) and (verbose):
            print("warning: failed to balance measures in cell {:d}".format(i))

    ParallelMap.ParallelMap(comm,DomDec.BalanceMeasuresMulti,argList,\
            callableArgList=True, callableArgListLen=len(partitionDataCompCells), callableReturn=callReturn,\
            chunksize=MPIchunksize, probetime=MPIprobetime)





###############################################################################################################################
# measure truncation
def ParallelTruncateMeasures(comm,muYAtomicDataList,muYAtomicIndicesList,thresh,\
        MPIchunksize=1, MPIprobetime=None):
    
    def argList(i):
        return [muYAtomicDataList[i],muYAtomicIndicesList[i],thresh]


    def callReturn(i,dat):
        # dat=new muYAtomic entry
        muYAtomicDataList[i]=dat[0]
        muYAtomicIndicesList[i]=dat[1]

    result=ParallelMap.ParallelMap(comm,Common.truncateSparseVector,argList,\
            callableArgList=True, callableArgListLen=len(muYAtomicDataList), callableReturn=callReturn,\
            chunksize=MPIchunksize, probetime=MPIprobetime)


###############################################################################################################################
# refine muYAtomic

def GetRefinedAtomicYMarginals_SparseY(comm,muYL,muYLOld,parentsYL,\
        atomicCellMasses,atomicCellMassesOld,\
        atomicCells,atomicCellsOld,\
        muYAtomicDataListOld,muYAtomicIndicesListOld,\
        metaCellShape,thresh=1E-15,\
        MPIchunksize=1, MPIprobetime=None):

    yresOld=muYLOld.shape[0]
    yres=muYL.shape[0]
    
    # list of children of each coarse node
    childrenYLOld=[[] for i in range(yresOld)]
    for i,parent in enumerate(parentsYL):
        childrenYLOld[parent].append(i)

    # old atomic cells are 2x2 clustering of new atomic cells
    newCellChildren=DomDec.GetPartitionIndices2D(metaCellShape,2,0)
    # for each new atomic cell compute the old parent atomic cell
    atomicCellParents=np.zeros((np.prod(metaCellShape),),dtype=np.int32)
    for i,children in enumerate(newCellChildren):
        atomicCellParents[children]=i

        
    def argList(i):
        return [muYAtomicDataListOld[i],muYAtomicIndicesListOld[i]]


    resultData=[None for i in range(len(atomicCells))]
    resultIndices=[None for i in range(len(atomicCells))]

    def callReturn(i,dat):
        # dat=(dataFine,indicesFine)
        for j in newCellChildren[i]:
            resultData[j]=dat[0]*atomicCellMasses[j]/atomicCellMassesOld[i]
            resultIndices[j]=dat[1].copy()

    ParallelMap.ParallelMap(comm,DomDec.refineMuYAtomicOld,argList,\
    		[muYL,muYLOld,childrenYLOld],\
    		callableArgList=True, callableArgListLen=len(atomicCellsOld),callableReturn=callReturn,\
    		chunksize=MPIchunksize, probetime=MPIprobetime)

    
    return [resultData,resultIndices]


