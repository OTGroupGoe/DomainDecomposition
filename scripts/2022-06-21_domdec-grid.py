import torch
import geomloss
import os
import sys
import time

repo = "../"
sys.path.append(repo)

from lib.header_script import *
import lib.Common as Common

import lib.DomainDecomposition as DomDec
#import lib.MultiScaleOT as MultiScaleOT

import psutil
import time

import pickle

from lib.header_params import *
#from lib.Aux import *
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import h5py
import json

config_file = sys.argv[1]

file_params = json.load(open(config_file,"r"))


datadir = r"../data/"

def importData(fn):
    try:
        data=sciio.loadmat(fn)
    except NotImplementedError:
        data = h5py.File(fn, 'r') # matlab v7 files
    posX=np.array(data["posX"],dtype=np.double,order="C")
    muX=np.array(data["muX"],dtype=np.double,order="C").ravel()
    shapeX=np.array(data["shapeX"],dtype=np.int32,order="C").ravel()
    posY=np.array(data["posY"],dtype=np.double,order="C")
    muY=np.array(data["muY"],dtype=np.double,order="C").ravel()
    piData=np.array(data["piData"],dtype=np.double,order="C").ravel()
    piIndices=np.array(data["piIndices"],dtype=np.int32,order="C").ravel()
    piIndptr=np.array(data["piIndptr"],dtype=np.int32,order="C").ravel()
    return posX,muX,shapeX,posY,muY,piData,piIndices,piIndptr
    
def mergeAtomicData(muYAtomicDataList,muYAtomicIndicesList):
    data=np.concatenate(muYAtomicDataList)
    indices=np.concatenate(muYAtomicIndicesList)
    indptr=np.zeros(len(muYAtomicDataList)+1,dtype=np.int32)
    for i in range(len(muYAtomicDataList)):
        indptr[i+1]=indptr[i]+len(muYAtomicDataList[i])
    return data,indices,indptr

def splitAtomicData(data,indices,indptr):
    muYAtomicDataList=[data[indptr[i]:indptr[i+1]].copy() for i in range(len(indptr)-1)]
    muYAtomicIndicesList=[indices[indptr[i]:indptr[i+1]].copy() for i in range(len(indptr)-1)]
    return muYAtomicDataList,muYAtomicIndicesList

# Import data

N = file_params["N"]
fn1 = datadir+r"2D/f-008-{:03d}.mat".format(N)
fn2 = datadir+r"2D/f-009-{:03d}.mat".format(N)

muX,posX,shapeX=Common.importMeasure(fn1)
muY,posY,shapeY=Common.importMeasure(fn2)
posX = posX/N # Normalizing imported, which are [0,1,...,N]
posY = posY/N

# Set up parameters
# There's a lot of parameters corresponding to the parallel routines; we are not
# going to need them for now.
params = getDefaultParams()
params["parallel_iteration"] = False
params

# Get some reasonably sparse initial feasible plan
def NWC(mu,nu):
    M=mu.shape[0]
    N=nu.shape[0]
    gamma=np.zeros((M,N))
    sumr=np.zeros(M)
    sumc=np.zeros(N)
    i=j=0
    while (i<M) and (j<N):
        delta=min(mu[i]-sumr[i],nu[j]-sumc[j])
        gamma[i,j]+=delta
        sumr[i]+=delta
        sumc[j]+=delta
        if mu[i]-sumr[i]<=1E-15:
            i+=1
        elif nu[j]-sumc[j]<=1E-15:
            j+=1
    return gamma

def visualize_deformation_map(muYAtomicIndicesList, muYAtomicDataList, partitionDataACompCells, muY, N):
    # Visualization: take all bottom-left basic cell supports and add them up, same with bottom-right, top-left, top-right. Colour each with one color
    image = np.zeros(N**2)
    for i in [0,1,2,3]:
        for comp in partitionDataACompCells:
            j = comp[i] # basic cell index
            image[muYAtomicIndicesList[j]] += muYAtomicDataList[j]*i
    image = image/muY
    plt.figure(figsize = (6,6))
    plt.imshow(image.reshape(N, N).T,origin = "bottom")

def getPrimalInfos(muY,posY,posXList,muXList,alphaList,betaDataList,betaIndexList,eps,getMuYList=True):
    scorePrimalUnreg=0.
    scorePrimal=0.
    errorMargX=0.
    errorMargY=0.
    
    if getMuYList:
        muYList=[]

    margY=np.zeros_like(muY)
    cellPlans = []
    for i in range(len(muXList)):
        posYcell=posY[betaIndexList[i]].copy()
        xresCell=posXList[i].shape[0]
        yresCell=posYcell.shape[0]
        c,cT=DomDec.LogSinkhorn.getEuclideanCost(posXList[i],posYcell)
        #cEff=c.reshape((xresCell,yresCell))\
        #        -np.einsum(alphaList[i],[0],np.ones((yresCell,),dtype=np.double),[1],[0,1])\
        #        -np.einsum(np.ones((xresCell,),dtype=np.double),[0],betaDataList[i],[1],[0,1])

        cEff=c.reshape((xresCell,yresCell))-alphaList[i].reshape((-1,1))-betaDataList[i].reshape((1,-1))


        piCell=np.einsum(np.exp(-cEff/eps),[0,1],muXList[i],[0],muY[betaIndexList[i]],[1],[0,1])
        cellPlans.append(piCell)

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
    
    return result, cellPlans

# Get map along 1st dimension
print("Computing initial plan")
muX_square = muX.reshape(N, N)
muY_square = muY.reshape(N, N)
muX0 = np.sum(muX_square, axis = 1)
muY0 = np.sum(muY_square, axis = 1)

blocks = np.full((N, N), None)
gamma0 = csr_matrix(NWC(muX0, muY0))

# Get each map along 2nd dimension
for i in range(N):
    mu1 = muX_square[i].reshape(N)
    mu2 = np.array((gamma0[i]/muY0) @ muY_square).reshape(N)
    gammai = csr_matrix(NWC(mu1, mu2))
    for j in gamma0[i].indices:
        #pi0[N*i:N*(i+1), N*j:N*(j+1)] = gammai * (gamma0[i,j]/muX0[i])
        blocks[i, j] = gammai * (gamma0[i,j]/muX0[i])
        
pi0 = scipy.sparse.bmat(blocks, format = "csr")
muX = np.array(np.sum(pi0, axis = 1)).ravel()
muY = np.array(np.sum(pi0, axis = 0)).ravel()
piData,piIndices,piIndptr = pi0.data, pi0.indices, pi0.indptr

# Get the data regarding basic (here "atomic") and composite cells
# as well as their corresponding marginals
dim=len(shapeX)
xres=muX.shape[0]
yres=muY.shape[0]
piZero=scipy.sparse.csr_matrix((piData,piIndices,piIndptr),shape=(xres,yres))

print("Preparing domdec setup")

if dim==1:
    atomicCells=DomDec.GetPartitionIndices1D(shapeX[0],params["domdec_cellsize"],0,full=True)
    metaCellShape=shapeX[0]//params["domdec_cellsize"]
    partitionMetaCellsA=DomDec.GetPartitionIndices1D(metaCellShape,2,0,full=True)
    partitionMetaCellsB=DomDec.GetPartitionIndices1D(metaCellShape,2,1,full=True)
elif dim==2:
    atomicCells=DomDec.GetPartitionIndices2D(shapeX,params["domdec_cellsize"],0)
    metaCellShape=[i//params["domdec_cellsize"] for i in shapeX]
    partitionMetaCellsA=DomDec.GetPartitionIndices2D(metaCellShape,2,0)
    partitionMetaCellsB=DomDec.GetPartitionIndices2D(metaCellShape,2,1)
else:
    raise ValueError("Setting dim={:d} is invalid. Only [1,2] are implemented.".format(dim))



partitionDataA=DomDec.GetPartitionData(atomicCells,partitionMetaCellsA)
partitionDataB=DomDec.GetPartitionData(atomicCells,partitionMetaCellsB)

partitionDataACompCells=partitionDataA[1]
partitionDataACompCellIndices=[np.array([partitionDataA[2][j][1:3] for j in x],dtype=np.int32) for x in partitionDataACompCells]
partitionDataBCompCells=partitionDataB[1]
partitionDataACompCellIndices=[np.array([partitionDataA[2][j][1:3] for j in x],dtype=np.int32) for x in partitionDataACompCells]
partitionDataBCompCells=partitionDataB[1]
partitionDataBCompCellIndices=[np.array([partitionDataB[2][j][1:3] for j in x],dtype=np.int32) for x in partitionDataBCompCells]

muXAList=[muX[cell].copy() for cell in partitionDataA[0]]
muXBList=[muX[cell].copy() for cell in partitionDataB[0]]

posXAList=[posX[cell].copy() for cell in partitionDataA[0]]
posXBList=[posX[cell].copy() for cell in partitionDataB[0]]

atomicCellMasses=np.array([np.sum(muX[cell]) for cell in atomicCells])


# set up new empty beta lists:
betaADataList=[None for i in range(len(muXAList))]
betaAIndexList=[None for i in range(len(muXAList))]
betaBDataList=[None for i in range(len(muXBList))]
betaBIndexList=[None for i in range(len(muXBList))]

# Cast input data to actual cellsizes

# Get basic cell Y-marginals
muYAtomicDataList=[np.array(np.sum(piZero[atomicCells[i]],axis=0)).ravel() for i in range(len(atomicCells))]
muYAtomicIndicesList=[np.arange(muY.shape[0],dtype=np.int32) for i in range(len(atomicCells))]

# truncation
for i in range(len(atomicCells)):
    muYAtomicDataList[i],muYAtomicIndicesList[i]=Common.truncateSparseVector(muYAtomicDataList[i],muYAtomicIndicesList[i],1E-15)

alphaAList=[np.zeros_like(muXAi) for muXAi in muXAList]
alphaBList=[np.zeros_like(muXBi) for muXBi in muXBList]


#Set up second set of Arguments to do 2 independent tests

muYAtomicDataList1 = muYAtomicDataList.copy() 
muYAtomicIndicesList1 = muYAtomicIndicesList.copy() 
posXAList1 = posXAList.copy() 
muXAList1 = muXAList.copy() 
alphaAList1 = alphaAList.copy() 
betaADataList1 = betaADataList.copy() 
betaAIndexList1 = betaAIndexList.copy() 
posXBList1 = posXBList.copy() 
muXBList1 = muXBList.copy() 
alphaBList1 = alphaBList.copy() 
betaBDataList1 = betaBDataList.copy() 
betaBIndexList1 = betaBIndexList.copy() 

partitionDataBCompCellsCut = []
partitionDataBCompCellIndicesCut = []
muXBList1Cut = []
posXBList1Cut = []
alphaBList1Cut = []
betaBDataList1Cut = []
betaBIndexList1Cut = []
for i in range(len(partitionDataBCompCells)):
  if len(partitionDataBCompCells[i]) == 4:
    partitionDataBCompCellsCut.append(partitionDataBCompCells[i])
    partitionDataBCompCellIndicesCut.append(partitionDataBCompCellIndices[i])
    muXBList1Cut.append(muXBList1[i])
    posXBList1Cut.append(posXBList1[i])
    alphaBList1Cut.append(alphaBList1[i])
    betaBDataList1Cut.append(betaBDataList1[i])
    betaBIndexList1Cut.append(betaBIndexList1[i])


eps = 1/N**2
# A iteration
params["sinkhorn_subsolver"] = "SolveOnCellKeopsGrid"
#params["sinkhorn_subsolver"] = "SolveOnCellKeops"
#params["sinkhorn_subsolver"] = "LogSinkhorn"
BoundingBox = True

params = {**params, **file_params}
print("Parameters:")
print(json.dumps("params"))
print("Computing domdec iteration")
t0 = time.time()
DomDec.BatchIterate(muY,posY,eps,\
                    partitionDataACompCells,partitionDataACompCellIndices,\
                    muYAtomicDataList1,muYAtomicIndicesList1,\
                    muXAList1,posXAList1,alphaAList1,betaADataList1,betaAIndexList1,shapeX,\
                    SinkhornSubSolver=params["sinkhorn_subsolver"], SinkhornError=params["sinkhorn_error"],\
                    SinkhornErrorRel=params["sinkhorn_error_rel"],\
                    SinkhornMaxIter = params["sinkhorn_max_iter"],
                    SinkhornInnerIter = params["sinkhorn_inner_iter"], 
                    BatchSize = params["keops_batchsize"]
                    )
print("Time elapsed: {}".format(time.time() - t0))
#visualize_deformation_map(muYAtomicIndicesList1, muYAtomicDataList1, partitionDataACompCells, muY, N)

# balancing

for i in range(len(atomicCells)):
    muYAtomicDataList1[i],muYAtomicIndicesList1[i]=Common.truncateSparseVector(muYAtomicDataList1[i],muYAtomicIndicesList1[i],1E-15)

ScoreGrid1, cellPlans1 = getPrimalInfos(muY,posY,posXAList1,muXAList1,alphaAList1,betaADataList1,betaAIndexList1,eps,getMuYList=False)

print("ScoreGrid1", json.dumps(ScoreGrid1))


