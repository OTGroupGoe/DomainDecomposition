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

import psutil
import time

import pickle

from lib.header_params import *
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import h5py
import json
import numpy as np

config_file = sys.argv[1]

file_params = json.load(open(config_file,"r"))

id = os.path.splitext(os.path.basename(config_file))[0]

datadir = r"../data2/"

resultsdir = r"../results/"

def get_deformation_map(muYAtomicIndicesList, muYAtomicDataList, partitionDataACompCells, muY, N):
    # Visualization: take all bottom-left basic cell supports and add them up, same with bottom-right, top-left, top-right. Colour each with one color
    image = np.zeros(N**2)
    for i in [0,1,2,3]:
        for comp in partitionDataACompCells:
            j = comp[i] # basic cell index
            image[muYAtomicIndicesList[j]] += muYAtomicDataList[j]*i
    image = image/muY
    return image.reshape(N, N)

# def getPrimalInfos(muY,posY,posXList,muXList,alphaList,betaDataList,betaIndexList,eps,getMuYList=True):

#     scorePrimalUnreg=0.
#     scorePrimal=0.
#     errorMargX=0.
#     errorMargY=0.
    
#     if getMuYList:
#         muYList=[]

#     margY=np.zeros_like(muY)
#     cellPlans = []
#     for i in range(len(muXList)):
#         posYcell=posY[betaIndexList[i]].copy()
#         xresCell=posXList[i].shape[0]
#         yresCell=posYcell.shape[0]
#         c,cT=DomDec.LogSinkhorn.getEuclideanCost(posXList[i],posYcell)
#         #cEff=c.reshape((xresCell,yresCell))\
#         #        -np.einsum(alphaList[i],[0],np.ones((yresCell,),dtype=np.double),[1],[0,1])\
#         #        -np.einsum(np.ones((xresCell,),dtype=np.double),[0],betaDataList[i],[1],[0,1])

#         cEff=c.reshape((xresCell,yresCell))-alphaList[i].reshape((-1,1))-betaDataList[i].reshape((1,-1))


#         piCell=np.einsum(np.exp(-cEff/eps),[0,1],muXList[i],[0],muY[betaIndexList[i]],[1],[0,1])
#         cellPlans.append(piCell)

#         scorePrimalUnreg+=np.sum(piCell.ravel()*c)
        
#         scorePrimal+=np.einsum(piCell,[0,1],alphaList[i],[0],[])\
#                 +np.einsum(piCell,[0,1],betaDataList[i],[1],[])\
#                 -eps*np.sum(piCell)
                
        
#         errorMargX+=np.sum(np.abs(np.sum(piCell,axis=1)-muXList[i]))
#         margY[betaIndexList[i]]+=np.sum(piCell,axis=0)
        
#         if getMuYList:
#             muYList.append(np.sum(piCell,axis=0))


#     errorMargY=np.sum(np.abs(muY-margY))
    
#     result={"scorePrimal":scorePrimal, "scorePrimalUnreg":scorePrimalUnreg,"errorMargX":errorMargX,"errorMargY":errorMargY}
    
#     if getMuYList:
#         return (result,muYList)
    
#     return result, cellPlans


# Start benchmarking

# def benchmark2D(file1,dump1,dump2,file_params,testID,error = 0.0001):

timeStamp1 = time.time()

params = getDefaultParams()

params = {**params, **file_params}

N = params["N"]
cellsize = params["domdec_cellsize"]
batchsize = params["keops_batchsize"]
innerIter = params["sinkhorn_inner_iter"]
maxIter = params["sinkhorn_max_iter"]

dim = 2
eps = 2*(1024/N)**2

datafile = "problem0_{}.mat".format(N)

f = h5py.File(datafile,'r+')

#load Data
I = np.array(f["I"], dtype = np.int64)
J = np.array(f["J"], dtype = np.int64)
V = np.array(f["V"])

alpha = np.array(f["alpha"])
beta = np.array(f["beta"])
muX = np.array(f["muX"])
muY = np.array(f["muY"])
posX = np.array(f["posX"]).T
posY = np.array(f["posY"]).T

import scipy.sparse as sp
pi_sp = sp.csr_matrix((V, (I, J)))

dx = posX[1,1] - posX[0,1]
print(dx)
x_pad = np.arange(-cellsize*dx, (N+cellsize)*dx, dx)
X0_pad = np.repeat(x_pad.reshape(-1,1), (N+2*cellsize), axis = 1).reshape(-1,1)
X1_pad = np.repeat(x_pad.reshape(1, -1),(N+2*cellsize), axis = 0).reshape(-1,1)
posX = np.concatenate((X0_pad, X1_pad), axis = 1)

#get the padded plan

linear_indices = np.arange((N+2*cellsize)**2)
cartesian0 = linear_indices % (N+2*cellsize)-cellsize 
cartesian1 = linear_indices // (N+2*cellsize)-cellsize 
original_indices = linear_indices[(0 <= cartesian0) & (cartesian0 < N) & (0 <= cartesian1) & (cartesian1 < N)]

padded_plan = sp.csr_matrix(((N+2*cellsize)**2, N**2))
padded_plan.indices = pi_sp.indices # Column indices of the values are the same (Y remains unchanged)
padded_plan.indptr = np.full((N+2*cellsize)**2+1, -1) # Init indptr to -1, to detect new slots
padded_plan.indptr[0] = 0 # Initialize sparse matrix indptr origin
padded_plan.indptr[original_indices+1] = pi_sp.indptr[1:] # Copy indptr to repesctive places
padded_plan.data = pi_sp.data
for i in range(1,len(padded_plan.indptr)):
  if padded_plan.indptr[i] == -1: # Fill missing values with previous indptr, since no new data was added
    padded_plan.indptr[i] = padded_plan.indptr[i-1]

# For some reason some routines fail, even though the data is the same as in the previous version. So we convert to a csr matrix (which already is) and it works
padded_plan = sp.csr_matrix(padded_plan)

alphapad = np.full(((N+2*cellsize), (N+2*cellsize)),1e-40)
alphapad[cellsize:N+cellsize, cellsize:N+cellsize] = alpha.reshape(N, N)
alpha = alphapad.ravel()

muX = np.array(np.sum(pi_sp,axis = 1)).ravel()
muY = np.array(np.sum(pi_sp,axis = 0)).ravel()

# pad MuX
muXpad = np.full(((N+2*cellsize), (N+2*cellsize)),1e-40)
muXpad[cellsize:N+cellsize, cellsize:N+cellsize] = muX.reshape(N, N)
muX = muXpad.ravel()

shapeY = (N,N)
shapeX = (N+2*cellsize, N+2*cellsize)
dim=len(shapeX)


atomicCells=DomDec.GetPartitionIndices2D(shapeX,cellsize,0)
metaCellShape=[i//cellsize for i in shapeX]
partitionMetaCellsA=DomDec.GetPartitionIndices2D(metaCellShape,2,1)
partitionMetaCellsB=DomDec.GetPartitionIndices2D(metaCellShape,2,0)

#remove non square composit cells
partitionMetaCellsA = [cell for cell in partitionMetaCellsA if len(cell) == 4]

partitionDataA=DomDec.GetPartitionData(atomicCells,partitionMetaCellsA)
partitionDataB=DomDec.GetPartitionData(atomicCells,partitionMetaCellsB)

partitionDataACompCells=partitionDataA[1]
partitionDataACompCellIndices=[np.array([partitionDataA[2][j][1:3] for j in x],dtype=np.int32) for x in partitionDataACompCells]
partitionDataBCompCells=partitionDataB[1]
partitionDataBCompCellIndices=[np.array([partitionDataB[2][j][1:3] for j in x],dtype=np.int32) for x in partitionDataBCompCells]

muXAList=[muX[cell].copy() for cell in partitionDataA[0]]
muXBList=[muX[cell].copy() for cell in partitionDataB[0]]

posXAList=[posX[cell].copy() for cell in partitionDataA[0]]
posXBList=[posX[cell].copy() for cell in partitionDataB[0]]

# New way of computing cell marginals
muYAtomicDataList = []
muYAtomicIndicesList = []
for atomic_indices in atomicCells:
  # Get all indices and values corresponding to the atomic cell
  y_indices = np.concatenate([padded_plan.indices[padded_plan.indptr[j]:padded_plan.indptr[j+1]] for j in atomic_indices])
  y_data = np.concatenate([padded_plan.data[padded_plan.indptr[j]:padded_plan.indptr[j+1]] for j in atomic_indices])
  nui = np.zeros(padded_plan.shape[1])
  # Add them up
  np.add.at(nui, y_indices, y_data) # Update nui with y_data at given y_indices with possibly repeated indices
  nzi = np.where(nui)[0]
  muYAtomicDataList.append(nui[nzi])
  muYAtomicIndicesList.append(nzi)

# truncation
for i in range(len(atomicCells)):
  muYAtomicDataList[i],muYAtomicIndicesList[i]=Common.truncateSparseVector(muYAtomicDataList[i],muYAtomicIndicesList[i],1E-15)

#alphaAList=[np.zeros_like(muXAi) for muXAi in muXAList] # needs padding #alphas missing
alphaAList=[alpha[cell].copy() for cell in partitionDataA[0]]
alphaBList=[np.zeros_like(muXBi) for muXBi in muXBList]

atomicCellMasses=np.array([np.sum(muX[cell]) for cell in atomicCells])

# set up new empty beta lists:
betaADataList=[None for i in range(len(muXAList))]
betaAIndexList=[None for i in range(len(muXAList))]
betaBDataList=[None for i in range(len(muXBList))]
betaBIndexList=[None for i in range(len(muXBList))]

timeStamp2 = time.time()

# A1 iteration
DomDec.BatchIterate(muY,posY,eps,\
                  partitionDataACompCells,partitionDataACompCellIndices,\
                  muYAtomicDataList,muYAtomicIndicesList,\
                  muXAList,posXAList,alphaAList,betaADataList,betaAIndexList,shapeY,\
                  "SolveOnCellKeopsGrid", params["sinkhorn_error"],\
                  False,\
                  SinkhornMaxIter = maxIter
                  , BatchSize = batchsize
                  )

# balancing
DomDec.BalanceMeasuresMultiAll(muYAtomicDataList,atomicCellMasses,partitionDataACompCells,verbose=True)
# truncation
for i in range(len(atomicCells)):
  muYAtomicDataList[i],muYAtomicIndicesList[i]=Common.truncateSparseVector(muYAtomicDataList[i],muYAtomicIndicesList[i],1E-15)

timeStamp3 = time.time()

# B1 iteration
DomDec.BatchIterate(muY,posY,eps,\
                  partitionDataBCompCells,partitionDataBCompCellIndices,\
                  muYAtomicDataList,muYAtomicIndicesList,\
                  muXBList,posXBList,alphaBList,betaBDataList,betaBIndexList,shapeY,\
                  "SolveOnCellKeopsGrid", params["sinkhorn_error"],\
                  False,\
                  SinkhornMaxIter = maxIter, BatchSize = batchsize
                  )

# balancing
DomDec.BalanceMeasuresMultiAll(muYAtomicDataList,atomicCellMasses,partitionDataBCompCells,verbose=True)
# truncation
for i in range(len(atomicCells)):
  muYAtomicDataList[i],muYAtomicIndicesList[i]=Common.truncateSparseVector(muYAtomicDataList[i],muYAtomicIndicesList[i],1E-15)

timeStamp4 = time.time()

# A2 iteration
DomDec.BatchIterate(muY,posY,eps,\
                  partitionDataACompCells,partitionDataACompCellIndices,\
                  muYAtomicDataList,muYAtomicIndicesList,\
                  muXAList,posXAList,alphaAList,betaADataList,betaAIndexList,shapeY,\
                  "SolveOnCellKeopsGrid", params["sinkhorn_error"],\
                  False,\
                  SinkhornMaxIter = maxIter, BatchSize = batchsize
                  )

# balancing
DomDec.BalanceMeasuresMultiAll(muYAtomicDataList,atomicCellMasses,partitionDataACompCells,verbose=True)
# truncation
for i in range(len(atomicCells)):
  muYAtomicDataList[i],muYAtomicIndicesList[i]=Common.truncateSparseVector(muYAtomicDataList[i],muYAtomicIndicesList[i],1E-15)

timeStamp5 = time.time()

# B2 iteration
DomDec.BatchIterate(muY,posY,eps,\
                  partitionDataBCompCells,partitionDataBCompCellIndices,\
                  muYAtomicDataList,muYAtomicIndicesList,\
                  muXBList,posXBList,alphaBList,betaBDataList,betaBIndexList,shapeY,\
                  "SolveOnCellKeopsGrid", params["sinkhorn_error"],\
                  False,\
                  SinkhornMaxIter = maxIter, BatchSize = batchsize
                  )

# visualize_deformation_map(muYAtomicIndicesList, muYAtomicDataList, partitionDataBCompCells, muY, N)

# balancing
DomDec.BalanceMeasuresMultiAll(muYAtomicDataList,atomicCellMasses,partitionDataBCompCells,verbose=True)
# truncation
for i in range(len(atomicCells)):
  muYAtomicDataList[i],muYAtomicIndicesList[i]=Common.truncateSparseVector(muYAtomicDataList[i],muYAtomicIndicesList[i],1E-15)

timeStamp6 = time.time()

# Get deformation map to check result on sight
image = get_deformation_map(muYAtomicIndicesList, muYAtomicDataList, partitionDataACompCells, muY, N)

# largedict = {
#     "id": testID,
#     "muYAtomicDataList": muYAtomicDataList,
#     "muYAtomicIndicesList": muYAtomicIndicesList,
#     "alphaBList": alphaBList,
#     "betaBDataList": betaBDataList,
#     "betaBIndexList": betaBIndexList,
# }

results = {
    "setup": timeStamp2 - timeStamp1,
    "AIteration1": timeStamp3 - timeStamp2,
    "BIteration1": timeStamp4 - timeStamp3,
    "AIteration2": timeStamp5 - timeStamp4,
    "BIteration2": timeStamp6 - timeStamp5,
    "deformation_map": image
    #"primalScore": getPrimalInfos(muY,posY,posXAList,muXAList,alphaAList,betaADataList,betaAIndexList,eps,getMuYList=False)
}

#where to dump?

# with open(dump1, 'wb') as f:
#   pickle.dump(largedict,f)

with open(resultsdir + id + ".pickle", 'wb') as f:
  pickle.dump(results,f)
