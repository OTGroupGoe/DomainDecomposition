#!/usr/bin/env python
# coding: utf-8

from math import e
import torch
import os
import sys
import time
import pickle

repo = "../"
sys.path.append(repo)


from lib.header_script import *
import lib.Common as Common

import lib.DomainDecomposition as DomDec
import lib.DomainDecompositionGPU as DomDecGPU
import lib.DomainDecompositionHybrid as DomDecHybrid
from lib.LogSinkhorn import LogSinkhorn as LogSinkhorn

import psutil
import time

import pickle

from lib.header_params import *
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import h5py
import json
import numpy as np
import scipy.io as io
import scipy.sparse as sp
import os


path_to_experiments = "experiments/hybrid_experiments"
path_to_results = "results/hybrid_experiments"

config_file = sys.argv[1]
file_params = json.load(open(os.path.join(path_to_experiments, config_file),"r"))
if len(sys.argv) == 3:
    Niter = int(sys.argv[2])
elif len(sys.argv) > 3: 
    assert False, "too many arguments"
else:  # Niter not given
    Niter = 4*file_params["N"]//file_params["domdec_cellsize"]

file_params["Niter"] = Niter

# datadir = r"../data2/"

# resultsdir = r"../results/"


def get_deformation_map(muYAtomicIndicesList, muYAtomicDataList, partitionDataACompCells, muY, N, ind=None):
    # Visualization: take all bottom-left basic cell supports and add them up, same with bottom-right, top-left, top-right. Colour each with one color
    image = np.zeros(N**2)
    if ind == None:
        ind = slice(None,)
    for i in [0,1,2,3]:
        for comp in partitionDataACompCells[ind]:
            j = comp[i] # basic cell index
            image[muYAtomicIndicesList[j]] += muYAtomicDataList[j]*(i+1)
    image = image/muY
    return image.reshape(N, N)

def get_mask_map(muYAtomicIndicesList, muYAtomicDataList, N, ind):
    # Visualization: take all bottom-left basic cell supports and add them up, same with bottom-right, top-left, top-right. Colour each with one color
    # ind is here list of basic cells
    image = np.zeros(N**2)
    for j in ind:
        image[muYAtomicIndicesList[j]] += muYAtomicDataList[j]
    return image.reshape(N, N)

def gaussian(X, mean, std):
    std_inv = np.linalg.inv(std)
    std_det = np.linalg.det(std)
    k = X.shape[1]
    assert k == mean.shape[-1]
    X_ = X - mean.reshape(1,-1)
    sqrd_norm = np.sum((X_) * (X_ @ std_inv.T), axis = 1).ravel()
    return np.exp(-0.5*sqrd_norm) / np.sqrt((2*np.pi)**k * std_det)

params = getDefaultParams()

device = "cuda"

params = {**params, **file_params}

N = params["N"]
cellsize = params["domdec_cellsize"]
InnerIter = params["sinkhorn_inner_iter"] = 10
maxIter = params["sinkhorn_max_iter"] = 10000
eps = params["eps"]
problem_type = params["problem_type"]
hybrid_mode = params["hybrid_mode"]
batchsize = params["batchsize"] = np.inf



#mu_shape = "gaussian"

dim = 2
#eps = 2*(1024/N)**2
#eps = 1/N**2

    
#x1 = x2 = torch.arange(0,N)/N
x1 = x2 = torch.linspace(-0.5, 0.5, N, device = device)
X = torch.cartesian_prod(x1, x2)


dx = (x1[1] - x1[0]).item()

# Get marginals
if problem_type in ["lebesgue_continuous", "lebesgue_semidiscrete"]:
    muX = np.ones(N**2)
    muX = muX/np.sum(muX)
    shapeX = (N, N)
    if problem_type == "lebesgue_continuous":
        muY = muX.copy()
        shapeY = (N,N)
    elif problem_type == "lebesgue_semidiscrete":
        muY = np.zeros(N)
        muY[0] = 1.0
        muY[-1] = 1.0
        muY = muY/np.sum(muY)
        shapeY = (N,1)
elif problem_type == "gaussian_sum":
    x_temp = torch.linspace(-0.5, 0.5,N//2)
    X_temp = torch.cartesian_prod(x_temp, x_temp)
    z = gaussian(X_temp.numpy(), np.zeros(2), 0.2*np.eye(2))
    z = z.reshape(N//2, N//2)
    muX = np.block([[z, z], [z, z]]).ravel()
    muX = muX/np.sum(muX)
    muY = muX.copy()

    shapeX = (N, N)
    shapeY = (N, N)
elif problem_type == "cross":
    cross = np.loadtxt("data/cross.txt")
    # Invert
    z = 1.0 - cross   # invert
    z = z + 0.1*cross # add background density
    N_cross = len(z)
    n_repeat = N // N_cross
    muX = np.repeat(np.repeat(z, n_repeat, axis = 0), n_repeat, axis = 1)
    muX = muX.ravel()
    muX = muX/np.sum(muX)
    muY = muX.copy()
    shapeX = (N, N)
    shapeY = (N, N)    

# Get init
print(json.dumps(params,indent = 4))
I = np.arange(N**2)
if problem_type in ["lebesgue_continuous", "gaussian_sum", "cross"]:
    if params["pi_init"] == "rotated":
        J = np.array([(N-1-j)*N+i for i in range(N) for j in range(N)]) 
    elif params["pi_init"] == "identity":
        J = I.copy()
elif problem_type == "lebesgue_semidiscrete":
    if params["pi_init"] == "rotated":
        theta = 1.00000001*np.pi/4
    elif params["pi_init"] == "identity":
        theta = 0
    e_theta = torch.tensor([np.cos(theta), np.sin(theta)], 
                           device = device, dtype = torch.float32)
    J = ((X @ e_theta) > 0).cpu().numpy() # Assigns to one of the Y points
    J = np.array(J, dtype = np.int32)*(N-1)
V = np.copy(muX)
pi_sp = sp.csr_matrix((V, (I, J)))

# Make sure marginals don't have any error
muX = np.array(pi_sp.sum(axis = 1)).ravel()
muY = np.array(pi_sp.sum(axis = 0)).ravel()
# Init alpha, beta

alpha = np.zeros_like(muX)
beta = np.zeros_like(muY)

dim=len(shapeX)

#############################
# End init
#############################
# Assert that shape of muX is already multiple of cellsize (otherwise change above)
print("Padding data")
for s in shapeX:
    assert s%cellsize == 0, "elements of shapeX must be divisible by cellsize"

# Pad plan for GPU
shapeX_pad = tuple(s + 2*cellsize for s in shapeX)

linear_indices = np.arange(np.prod(shapeX_pad))
cartesian0 = linear_indices % shapeX_pad[1]-cellsize 
cartesian1 = linear_indices // shapeX_pad[1]-cellsize 
original_indices = linear_indices[(0 <= cartesian0) & (cartesian0 < N) & (0 <= cartesian1) & (cartesian1 < N)]

padded_plan = sp.csr_matrix((np.prod(shapeX_pad), np.prod(shapeY)))
padded_plan.indices = pi_sp.indices # Column indices of the values are the same (Y remains unchanged)
padded_plan.indptr = np.full(np.prod(shapeX_pad)+1, -1) # Init indptr to -1, to detect new slots
padded_plan.indptr[0] = 0 # Initialize sparse matrix indptr origin
padded_plan.indptr[original_indices+1] = pi_sp.indptr[1:] # Copy indptr to repesctive places
padded_plan.data = pi_sp.data
for i in range(1,len(padded_plan.indptr)):
    if padded_plan.indptr[i] == -1: # Fill missing values with previous indptr, since no new data was added
        padded_plan.indptr[i] = padded_plan.indptr[i-1]

# For some reason some routines fail, even though the data is the same as in the previous version. So we convert to a csr matrix (which already is) and it works
padded_plan = sp.csr_matrix(padded_plan)

# Pad alpha init
# alphapad = np.full(((N+2*cellsize), (N+2*cellsize)),0.0)
# alphapad[cellsize:N+cellsize, cellsize:N+cellsize] = alpha.reshape(N, N)
# alpha = alphapad.ravel()
# New cleaner way
alphapad = DomDecGPU.pad_array(alpha.reshape(shapeX), cellsize).ravel()

# Pad MuX
# muXpad = np.full(((N+2*cellsize), (N+2*cellsize)),1e-40)
# muXpad[cellsize:N+cellsize, cellsize:N+cellsize] = muX.reshape(N, N)
# muX = muXpad.ravel()
muXpad = DomDecGPU.pad_array(muX.reshape(shapeX), cellsize, 
                             pad_value = 1e-40).ravel()

# Create atomic, composite and partitions from the padded data

print("Preparing domdec iterations")
atomicCells=DomDec.GetPartitionIndices2D(shapeX_pad,cellsize,0)
metaCellShape=[i//cellsize for i in shapeX_pad]
partitionMetaCellsA=DomDec.GetPartitionIndices2D(metaCellShape,2,1)
partitionMetaCellsB=DomDec.GetPartitionIndices2D(metaCellShape,2,0)

# Remove non square composite cells
partitionMetaCellsA = [cell for cell in partitionMetaCellsA if len(cell) == 4]

# Compute batch dimension for A and B partitions:
batchdim_A = len(partitionMetaCellsA)
batchdim_B = len(partitionMetaCellsB)

partitionDataA=DomDec.GetPartitionData(atomicCells,partitionMetaCellsA)
partitionDataB=DomDec.GetPartitionData(atomicCells,partitionMetaCellsB)

partitionDataACompCells=partitionDataA[1]
partitionDataACompCellIndices=[np.array([partitionDataA[2][j][1:3] for j in x],dtype=np.int32) for x in partitionDataACompCells]
partitionDataBCompCells=partitionDataB[1]
partitionDataBCompCellIndices=[np.array([partitionDataB[2][j][1:3] for j in x],dtype=np.int32) for x in partitionDataBCompCells]


# Generate problem data from partition data. Reshape it already for GPU use. 

muXA = torch.tensor(np.array([muXpad[cell] for cell in partitionDataA[0]], dtype = np.float32)).cuda()
muXB = torch.tensor(np.array([muXpad[cell] for cell in partitionDataB[0]], dtype = np.float32)).cuda()
# Need to permute dims
muXA = muXA.view(-1, 2, 2, cellsize, cellsize).permute((0,1,3,2,4)).contiguous().view(-1, 2*cellsize, 2*cellsize)
muXB = muXB.view(-1, 2, 2, cellsize, cellsize).permute((0,1,3,2,4)).contiguous().view(-1, 2*cellsize, 2*cellsize)

# Get cell marginals (replaced by "refine" in the multiscale setting)
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

muYAtomicDataList_init = [a.copy() for a in muYAtomicDataList]
muYAtomicIndicesList_init = [a.copy() for a in muYAtomicIndicesList]

# Init alphas
alphaA = torch.tensor(np.array([alphapad[cell] for cell in partitionDataA[0]], dtype = np.float32)).cuda()
alphaB = torch.tensor(np.array([alphapad[cell] for cell in partitionDataB[0]], dtype = np.float32)).cuda()

# Need to permute dims
alphaA = alphaA.view(-1, 2, 2, cellsize, cellsize).permute((0,1,3,2,4)).contiguous().view(-1, 2*cellsize, 2*cellsize)
alphaB = alphaB.view(-1, 2, 2, cellsize, cellsize).permute((0,1,3,2,4)).contiguous().view(-1, 2*cellsize, 2*cellsize)

# TODO: this is same as basic_mass below? Then simplify
atomicCellMasses=np.array([np.sum(muXpad[cell]) for cell in atomicCells])

# set up new empty beta lists:
betaADataList=[None for _ in partitionDataACompCells]
betaAIndexList=[None for _ in partitionDataACompCells]
betaBDataList=[None for _ in partitionDataBCompCells]
betaBIndexList=[None for _ in partitionDataBCompCells]

timeStamp2 = time.time()

balance_verbose = True

# Get position of composite cells
X_width = shapeX_pad[1]
leftA = torch.tensor([cell[0]//X_width for cell in partitionDataA[0]], device = device)
leftB = torch.tensor([cell[0]//X_width for cell in partitionDataB[0]], device = device)
bottomA = torch.tensor([cell[0]%X_width for cell in partitionDataA[0]], device = device)
bottomB = torch.tensor([cell[0]%X_width for cell in partitionDataB[0]], device = device)

x1A, x2A = DomDecGPU.get_grid_cartesian_coordinates(leftA, bottomA, 2*cellsize, 2*cellsize, dx)
x1B, x2B = DomDecGPU.get_grid_cartesian_coordinates(leftB, bottomB, 2*cellsize, 2*cellsize, dx)
posXA = (x1A, x2A)
posXB = (x1B, x2B)


if hybrid_mode == "hybrid":
    print("Preparing hybrid iterations")
    # Prepare parameters for hybrid iteration
    basic_shape = tuple(s//cellsize for s in shapeX)
    (sb1, sb2) = basic_shape
    muX_basic = DomDecGPU.convert_to_basic_2D(muXA, basic_shape, cellsize)
    # Flatten X and Y dimensions to make easier to handle
    muX_basic = muX_basic.reshape(*basic_shape, cellsize**2)
    basic_mass = muX_basic.sum(dim = -1)
    capacities_nodes = basic_mass.ravel().cpu().numpy()
    edges = DomDecHybrid.get_edges_2D(basic_shape)

    # Compute plans between basic cell marginals
    eps_gamma = params["eps_gamma"]
    gamma = DomDecHybrid.get_gamma_2D(muX_basic, basic_mass, eps_gamma, cellsize)

# Init domdec history 
muYAtomicDataList = [a.copy() for a in  muYAtomicDataList_init]
muYAtomicIndicesList = [a.copy() for a in muYAtomicIndicesList_init]

atomic_data_hist = []
atomic_indices_hist = []
alpha_hist = [None]
flow_hist = []

atomic_data_hist.append([a.copy() for a in  muYAtomicDataList_init])
atomic_indices_hist.append([a.copy() for a in muYAtomicIndicesList_init])


# Get Nu_basic
torch_options = dict(dtype = torch.float64, device = "cuda")
torch_options_int = dict(dtype = torch.int32, device = "cuda")
Nu_basic, _, left, bottom, max_width, max_height = DomDecGPU.batch_cell_marginals_2D(
    muYAtomicIndicesList, muYAtomicDataList, shapeY, muY
)

Nu_basic = torch.tensor(Nu_basic, **torch_options)
# Trim extra basic cells (we have to do it before when we improve refinement)
Nu_basic = Nu_basic.view(*metaCellShape,*Nu_basic.shape[1:])[1:-1,1:-1]
left = torch.tensor(left, **torch_options_int).view(metaCellShape)[1:-1,1:-1]
bottom = torch.tensor(bottom, **torch_options_int).view(metaCellShape)[1:-1,1:-1]
    

# Perform domdec iterations
print(f"Starting {hybrid_mode} iterations" )
for i in range(Niter):
    if i%2 == 0:
        print(i)
        # A iteration
        posY = None # TODO: remove, not used
        alphaA, Nu_basic, left, bottom, info = DomDecGPU.BatchIterateBox(
            muY,posY,dx,eps,\
            muXA,posXA,alphaA,Nu_basic, left, bottom, shapeY,"A",
            SinkhornError = params["sinkhorn_error"],
            SinkhornErrorRel = params["sinkhorn_error_rel"],
            SinkhornMaxIter = params["sinkhorn_max_iter"], 
            SinkhornInnerIter = params["sinkhorn_inner_iter"],
            BatchSize = params["batchsize"]
        )
        alpha_hist.append(alphaA)
        if hybrid_mode == "hybrid":
            # Implement flow 
            w, h = Nu_basic.shape[2:]
            muYAtomicIndicesList, muYAtomicDataList = \
                DomDecGPU.unpack_cell_marginals_2D_box(
                    Nu_basic.view(-1, w, h).cpu().numpy(), 
                    left.ravel().cpu().numpy(), 
                    bottom.ravel().cpu().numpy(), shapeY
                )
            w, time_flow = DomDecHybrid.flow_update(muYAtomicDataList, muYAtomicIndicesList, 
                        info["solver"], muX_basic, basic_mass, gamma, basic_shape, 
                        capacities_nodes, edges, cellsize)
            print(time_flow)
            
            print("Norm of flow", np.linalg.norm(w))

            w = DomDecHybrid.flow_update(muYAtomicDataList, muYAtomicIndicesList, 
                        info["solver"], muX_basic, basic_mass, gamma, basic_shape, 
                        capacities_nodes, edges, cellsize)
            flow_hist.append(w)
            # TUrn muY back to nu_basic
            Nu_basic, _, left, bottom, max_width, max_height = DomDecGPU.batch_cell_marginals_2D(
                muYAtomicIndicesList, muYAtomicDataList, shapeY, muY
            )

            Nu_basic = torch.tensor(Nu_basic, **torch_options)
            # Trim extra basic cells (we have to do it before when we improve refinement)
            Nu_basic = Nu_basic.view(*metaCellShape,*Nu_basic.shape[1:])[1:-1,1:-1]
            left = torch.tensor(left, **torch_options_int).view(metaCellShape)[1:-1,1:-1]
            bottom = torch.tensor(bottom, **torch_options_int).view(metaCellShape)[1:-1,1:-1]
    else:
        # B iter
        alphaB, Nu_basic, left, bottom, info = DomDecGPU.BatchIterateBox(
            muY,posY,dx,eps,\
            muXB,posXB,alphaB,Nu_basic, left, bottom, shapeY, "B",
            SinkhornError = params["sinkhorn_error"],
            SinkhornErrorRel = params["sinkhorn_error_rel"],
            SinkhornMaxIter = params["sinkhorn_max_iter"], 
            SinkhornInnerIter = params["sinkhorn_inner_iter"],
            BatchSize = params["batchsize"]
        )
        alpha_hist.append(alphaB)
    
    atomic_data_hist.append([a.copy() for a in  muYAtomicDataList])
    atomic_indices_hist.append([a.copy() for a in muYAtomicIndicesList])
# Dump results

results = {
    "atomic_data_hist": atomic_data_hist,
    "atomic_indices_hist": atomic_indices_hist,
    "flow_hist": flow_hist,
    "alpha_hist": alpha_hist, 
    "params": params
}

print("Dumping data")
results_file = os.path.join(path_to_results, config_file[:-5]+".pickle")
with open(results_file, "wb") as f:
    pickle.dump(results, f)