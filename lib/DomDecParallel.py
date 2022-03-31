import numpy as np
import scipy
import scipy.sparse

import lib.Common as Common
import lib.DomainDecomposition as DomDec
import lib.LogSinkhorn.LogSinkhorn as LogSinkhorn
import lib.CPPSinkhorn.CPPSinkhorn as CPPSinkhorn

import psutil
import time


import multiprocessing.pool as mppool
import multiprocessing as mp

def getMemory(x):
    time.sleep(0.1)
    process = psutil.Process(os.getpid())
    return (os.getpid(),process.memory_info().rss)  # in bytes


def initWorker(_muXList,_posXList,_muY,_posY,_partitionData,subSolver="LogSinkhorn"):
    global muXList,posXList,muY,posY,partitionData
    muXList=_muXList
    posXList=_posXList
    muY=_muY
    posY=_posY
    partitionData=_partitionData
    
    global SolveOnCell
    if subSolver=="LogSinkhorn":
    	SolveOnCell=SolveOnCell_LogSinkhorn
    elif subSolver=="SparseSinkhorn":
    	SolveOnCell=SolveOnCell_SparseSinkhorn
    

##############################################################################################################################
# solve sub problems

def getPi(c,alpha,beta,rhoX,rhoY,eps):
    pi=-c+np.einsum(alpha,[0],np.ones_like(beta),[1],[0,1])+np.einsum(np.ones_like(alpha),[0],beta,[1],[0,1])
    pi=np.exp(pi/eps)
    pi=np.einsum(rhoX,[0],rhoY,[1],pi,[0,1],[0,1])
    return pi

def SolveOnCell_LogSinkhorn(muX,muY,posX,posY,rhoX,rhoY,alphaInit,eps,YThresh=1E-14,SinkhornError=1E-5):
    
    subY=muY.indices
    # pick partial marginal vector and point positions
    subMuY=muY.data
    subPosY=posY[subY].copy()
    subRhoY=rhoY[subY].copy()


    alpha=alphaInit.copy()
    beta=np.zeros_like(subMuY)
    c=Common.getEuclideanCostFunction(posX,subPosY,p=2.)
    cT=c.transpose().copy()
    
    msg=LogSinkhorn.iterateUntilError(alpha,beta,c,cT,muX,subMuY,rhoX,subRhoY,eps,1000,20,SinkhornError)
    
    if msg==1:
        print("warning: {:d} : Sinkhorn did not converge to accuracy".format(msg))
    elif msg!=0:
        print("warning: {:d} : unspecified error".format(msg))

    pi=getPi(c,alpha,beta,rhoX,subRhoY,eps)

    return (msg,alpha,beta,pi)


def SolveOnCell_SparseSinkhorn(muX,muY,posX,posY,rhoX,rhoY,alphaInit,eps,YThresh=1E-14,SinkhornError=1E-4,\
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
    
    subY=muY.indices
    # pick partial marginal vector and point positions
    subMuY=muY.data
    subPosY=posY[subY].copy()
    subRhoY=rhoY[subY].copy()

    result=CPPSinkhorn.SolveSinkhorn(muX,subMuY,posX,subPosY,rhoX,subRhoY,alphaInit,\
            SinkhornError,eps,0)

    if autoEpsFix:
        epsExponent=0
        while result[0]!=0:
            # in case the algorithm failed (probably since eps is too small),
            # increase eps until it succeeds
            
            epsExponent+=1
            epsEff=eps*(2**epsExponent)
            
            result=CPPSinkhorn.SolveSinkhorn(muX,subMuY,posX,subPosY,rhoX,subRhoY,alphaInit,\
                    SinkhornError,epsEff,0)

        if (epsExponent>0) and verbose:
            print("autoFix:\t{:d}".format(epsExponent))
        while epsExponent>0:
            # now start decreasing again
            alphaInitEff=result[1]
            epsExponent-=1
            epsEff=eps*(2**epsExponent)
            result=CPPSinkhorn.SolveSinkhorn(muX,subMuY,posX,subPosY,rhoX,subRhoY,alphaInitEff,\
                    SinkhornError,epsEff,0)

            
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
        cellNr,cellAlpha,muYAtomicList,eps,\
        sparseThreshold=-1.):
    """Iterate a cell in one partition: combine corresponding atomic cells, solve subproblem and compute new atomic partial marginals."""
        
    # trivial return to measure time required for inter-process communication
    #return (cellAlpha,muYAtomicList)
        
    # compute Y marginal of cell by summing atomic Y marginals
    muYCell=muYAtomicList[0].copy()
    for muYTerm in muYAtomicList[1:]:
        muYCell+=muYTerm

    # solve on cell
    result=SolveOnCell(muXList[cellNr],muYCell,posXList[cellNr],posY,muXList[cellNr],muY,cellAlpha,eps)
    #(msg,alpha,beta,pi)

    # update partial dual variable
    resultAlpha=result[1]

    # extract new atomic muY
    pi=result[3]
    resultMuYAtomicList=[\
            Common.getSparseVector(Common.GetPartialYMarginal(pi,range(*partitionData[2][j][1:3])),\
                sparseThreshold,\
                subIndices=muYCell.indices,subShape=muY.shape[0])
            for j in partitionData[1][cellNr]
            ]

    return (resultAlpha,resultMuYAtomicList)

###############################################################################################################################
# mass balancing between atomic marginals

def getPairwiseDelta(d1,d2):
    return min(max(d1,0),max(-d2,0))-min(max(-d1,0),max(d2,0))

def BalanceMeasuresMulti(muYAtomicListSub,massDeltasSub,threshStep=1E-16,threshTerminate=1E-10):
    nCells=len(muYAtomicListSub)
    if nCells==1:
        return (0,muYAtomicListSub,massDeltasSub)

    for i in range(nCells):
        for j in range(i+1,nCells):
            delta=getPairwiseDelta(massDeltasSub[i],massDeltasSub[j])
            if delta>0:
                delta=min(delta,np.sum(muYAtomicListSub[i]))
            else:
                delta=max(delta,-np.sum(muYAtomicListSub[j]))
            if np.abs(delta)>threshStep:
                LogSinkhorn.balanceMeasures_sparse(\
                        muYAtomicListSub[i].data,muYAtomicListSub[j].data,\
                        muYAtomicListSub[i].indices,muYAtomicListSub[j].indices,\
                        delta)
                massDeltasSub[i]-=delta
                massDeltasSub[j]+=delta
            if np.sum(np.abs(massDeltasSub))<threshTerminate:
                return (0,muYAtomicListSub,massDeltasSub)
    return (1,muYAtomicListSub,massDeltasSub)


###############################################################################################################################

def setSpawnMethod(mthd="spawn"):
    mp.set_start_method(mthd)

class DomainDecompositionParallelIterator:
    def __init__(self,nWorkers,muXList,posXList,muY,posY,partitionData,subSolver="LogSinkhorn"):
        self.nWorkers=nWorkers
        self.muXList=muXList
        self.posXList=posXList
        self.muY=muY
        self.posY=posY
        self.partitionData=partitionData
        self.subSolver=subSolver
        
        self.nCells=len(self.muXList)

        self.pool=mp.Pool(self.nWorkers,initWorker,(self.muXList,self.posXList,self.muY,self.posY,self.partitionData,self.subSolver))

    def iterate(self,alphaList,muYAtomicList,eps):
        argList=[(i,alphaList[i],[muYAtomicList[j] for j in self.partitionData[1][i]],eps) for i in range(self.nCells)]
        result=self.pool.starmap(DomDecIteration_SparseY,argList)

        for i,(alphaData,muYAtomicData) in enumerate(result):
            alphaList[i]=alphaData
            for jsub,j in enumerate(self.partitionData[1][i]):
                muYAtomicList[j]=muYAtomicData[jsub]

    def balanceMeasures(self,muYAtomicList,massDeltas,verbose=False):
        argList=[[[muYAtomicList[j] for j in self.partitionData[1][i]],massDeltas[self.partitionData[1][i]]] for i in range(self.nCells)]
        result=self.pool.starmap(BalanceMeasuresMulti,argList)
        
        for i,(msg,muYAtomicData,massDeltaData) in enumerate(result):
            massDeltas[self.partitionData[1][i]]=massDeltaData
            for jsub,j in enumerate(self.partitionData[1][i]):
                muYAtomicList[j]=muYAtomicData[jsub]
            if (msg!=0) and (verbose):
                print("warning: failed to balance measures in cell {:d}".format(i))

    def truncateMeasures(self,muYAtomicList,thresh):
        argList=[[x,thresh] for x in muYAtomicList]
        result=self.pool.starmap(Common.truncateSparseVector,argList)
        for i in range(len(muYAtomicList)):
            muYAtomicList[i]=result[i]


    def getMemory(self):
        return self.pool.map(getMemory,range(self.nWorkers))

######################################################################################################################################

def initWorkerRefiner(_muYAtomicListOld,_atomicCellParents,_muXL,_muYL,_muXLOld,_muYLOld,_atomicCells,_atomicCellsOld,_childrenYLOld):
    global muYAtomicListOld,atomicCellParents,muXL,muYL,muXLOld,muYLOld,atomicCells,atomicCellsOld,childrenYLOld
    
    muYAtomicListOld=_muYAtomicListOld
    atomicCellParents=_atomicCellParents
    muXL=_muXL
    muYL=_muYL
    muXLOld=_muXLOld
    muYLOld=_muYLOld
    atomicCells=_atomicCells
    atomicCellsOld=_atomicCellsOld
    childrenYLOld=_childrenYLOld


def refineMuYAtomic(i):
    muYAtomicOld=muYAtomicListOld[atomicCellParents[i]]

    yres=muYL.shape[0]
    xMassScaleFactor=np.sum(muXL[atomicCells[i]])/np.sum(muXLOld[atomicCellsOld[atomicCellParents[i]]])
    indicesFine=[]
    dataFine=[]
    yMassScaleFactors=muYAtomicOld.data/muYLOld[muYAtomicOld.indices]
    for d,y in zip(yMassScaleFactors,muYAtomicOld.indices):
        indicesFine+=childrenYLOld[y]
        dataFine.append(muYL[childrenYLOld[y]]*d*xMassScaleFactor)
    indicesFine=np.array(indicesFine,dtype=np.int32)
    dataFine=np.hstack(dataFine)
    indptrFine=np.array([0,indicesFine.shape[0]],dtype=np.int32)
    result=Common.truncateSparseVector(\
            scipy.sparse.csr_matrix((dataFine,indicesFine,indptrFine),shape=(1,yres)),
            thresh=1E-15)
    result.sort_indices()
    return result
    

class MuYAtomicParallelRefiner:
    def __init__(self,nWorkers,muXL,muYL,muXLOld,muYLOld,parentsYL,
            atomicCells,atomicCellsOld,muYAtomicListOld,metaCellShape,thresh=1E-15):
            
        self.nWorkers=nWorkers
        self.muXL=muXL
        self.muYL=muYL
        self.muXLOld=muXLOld
        self.muYLOld=muYLOld
        self.parentsYL=parentsYL
        self.atomicCells=atomicCells
        self.atomicCellsOld=atomicCellsOld
        self.muYAtomicListOld=muYAtomicListOld
        self.metaCellShape=metaCellShape
        self.thresh=thresh


        yresOld=muYLOld.shape[0]

        # list of children of each coarse node
        self.childrenYLOld=[[] for i in range(yresOld)]
        for i,parent in enumerate(parentsYL):
            self.childrenYLOld[parent].append(i)

        # old atomic cells are 2x2 clustering of new atomic cells
        newCellChildren=DomDec.GetPartitionIndices2D(metaCellShape,2,0)
        # for each new atomic cell compute the old parent atomic cell
        self.atomicCellParents=np.zeros((np.prod(metaCellShape),),dtype=np.int)
        for i,children in enumerate(newCellChildren):
            self.atomicCellParents[children]=i
        
        self.pool=mp.Pool(self.nWorkers,initWorkerRefiner,\
                (self.muYAtomicListOld,self.atomicCellParents,self.muXL,self.muYL,self.muXLOld,self.muYLOld,self.atomicCells,self.atomicCellsOld,self.childrenYLOld))

    def getRefinedMuYAtomic(self):
        return self.pool.map(refineMuYAtomic,range(len(self.atomicCells)))


#################################################



