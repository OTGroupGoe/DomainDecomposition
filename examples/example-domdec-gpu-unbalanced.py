import time
import argparse
import sys
sys.path.append("../")
from lib.header_script import *
import lib.Common as Common
import lib.DomainDecompositionGPU as DomDecGPU
import lib.DomDecUnbalancedGPU as DomDecUnbalanced
import torch

# from LogSinkhornGPU import LogSinkhornCudaImage, AbstractSinkhorn
import LogSinkhornGPU
import time
import json
import numpy as np
import pickle
import os

from lib.header_params import *
from lib.AuxConv import *

###############################################################################
# # GPU multiscale domdec for unbalanced entropic optimal transport [1]
# =============================================================================
#
# The input measures are provided in a .pickle file containing a dictionary
# with keys:
# * mu: (N,) array
#       Weights of the measure. 
# * pos: (N, d) array
#       Positions of the measure's support. They are supposed to be equispaced
#       on a regular grid of shape `shapeX`
# * shape: d-tuple
#       Shape of grid.
# 
# Provided multiscale implementation only support shapes that are power of 2,  
# and equispaced grids.
# 
# [1] Ismael Medina, The Sang Nguyen and Bernhard Schmitzer. 
#     *Domain decomposition for entropic unbalanced optimal transport*. 
#     arXiv:2410.08859, 2024.
###############################################################################


class SinkhornKL(LogSinkhornGPU.UnbalancedSinkhornCudaImageOffset):
    """
    Same as LogSinkhornGPU.UnbalancedPartialSinkhornCudaImageOffset but with 
    KL stopping criterion
    """
    def __init__(self, mu, nu, C, eps, lam, **kwargs):
        super().__init__(mu, nu, C, eps, lam, **kwargs)

    def get_current_error(self):
        """
        KL marginal error
        """
        PXpi = self.get_actual_X_marginal()
        PXpi_opt = torch.exp(-self.alpha / self.lam) * self.mu
        # Return KL
        return LogSinkhornGPU.KL(PXpi, PXpi_opt)
    

def get_global_X_marginal(PXpi_comp, shapeX, part):
    cellsize = PXpi_comp.shape[-1] // 2
    c1, c2 = shapeX[0]//(2*cellsize), shapeX[1]//(2*cellsize)
    if part == "B":
        c1, c2 = c1+1, c2+1
    PXpi = PXpi_comp.view((c1, c2, 2*cellsize, 2*cellsize)) \
                .permute(0,2,1,3).reshape(2*c1*cellsize, 2*c2*cellsize).contiguous()
    if part == "B":
        PXpi = PXpi[cellsize:-cellsize, cellsize:-cellsize]
    return PXpi

# read parameters from command line and cfg file
print("setting script parameters")
params = getDefaultParams()

# Input data files
params["setup_fn1"] = "data/f-000-256.pickle"
params["setup_fn2"] = "data/f-001-256.pickle"

# Sinkhorn parameters
params["sinkhorn_max_iter"] = 10000
params["sinkhorn_inner_iter"] = 10
params["sinkhorn_error"] = 2e-5
params["sinkhorn_error_rel"] = True

# DomDec parameters
params["domdec_cellsize"] = 4 # Domain decomposition cellsize
cellsize = params["domdec_cellsize"]
params["batchsize"] = np.inf # Maximum size of the batches
params["balance"] = True # Whether performing the balancing procedure, described in [1]

# Unbalanced parameters
params["reach"] = 1.0 # Square root of the soft-penalty parameter $\lambda$
params["max_time"] = 300 # Maximum running time, after which `sinkhorn_max_iter` is set to zero
params["safeguard_threshold"] = 0.005 # Allowed relative increase of the primal score

# Warm starting parameters
params["nLayerSinkhornLast"] = 7 # Last multiscale layer solved with global Sinkhorn; after this domdec is used
params["sinkhorn_error_multiplier"] = 0.25 # Objective error for global sinkhorn, relative to `sinkhorn_error`

# Multiscale parameters
params["hierarchy_top"] = 3 # First multiscale layer

# Dump files
params["aux_dump_finest"] = False 
params["aux_evaluate_scores"] = True 
params["dump_results"] = True
params["additional_tag"] = "" 

##########################################################
# Torch parameters
device = "cuda"
# torch_dtype = torch.float32
torch_dtype = torch.float64

torch_options = dict(dtype=torch_dtype, device=device)
torch_options_int = dict(dtype=torch.int32, device=device)

###############################################################
###############################################################

# Allow all parameters to be overriden on the command line
args = argparse.ArgumentParser()
for key in params.keys():
    args.add_argument(f"--{key}", dest = key, 
            default = params[key], type = type(params[key]))
params = vars(args.parse_args())

print("final parameter settings")
for k in sorted(params.keys()):
    print("\t", k, params[k])

# Dump files
tag1 = params["setup_fn1"].split("-")[1]
tag2 = params["setup_fn2"].split("-")[1]
additional_tag = params["additional_tag"]
Nstr = int(params["setup_fn1"].split("-")[2].split(".")[0])
reach = params["reach"]

# Create results folder if it does not exist
if not os.path.isdir("results"):
    os.mkdir("results")

params["setup_resultfile"] = f"results/domdec{additional_tag}-reach-{reach}-{Nstr:04d}-{tag1}-{tag2}.txt"
params["setup_dumpfile_finest"] = f"results/domdec{additional_tag}-reach-{reach}-{Nstr:04d}-{tag1}-{tag2}-dumpfinest.pickle"

print(f"Solving with {params['sinkhorn_subsolver']} solver")
# load input measures from file
# do some preprocessing and setup multiscale representation of them
muX, posX, shapeX = Common.importMeasure(params["setup_fn1"])
muY, posY, shapeY = Common.importMeasure(params["setup_fn2"])
N = shapeX[0]
params["hierarchy_depth"] = int(np.log2(N))

lam = (reach*N)**2
params["lam"] = lam

print(f"N = {N}")
print(torch_dtype)


# Get multiscale torch hierarchy
muX_final = torch.tensor(muX, **torch_options).view(shapeX)
muY_final = torch.tensor(muY, **torch_options).view(shapeX)
muX_layers = DomDecGPU.get_multiscale_layers(muX_final, shapeX)
muY_layers = DomDecGPU.get_multiscale_layers(muY_final, shapeY)

# muX: 1d array of masses
# posX: (*,dim) array of locations of masses
# shapeX: dimensions of original spatial grid on which the masses live
#    useful for visualization and setting up the domdec cells

# setup eps scaling
if params["eps_schedule"] == "default":
    params["eps_list"] = Common.getEpsListDefault(params["hierarchy_depth"], params["hierarchy_top"],
                                                  params["eps_base"], params["eps_layerFactor"], params["eps_layerSteps"], params["eps_stepsFinal"],
                                                  nIterations=params["eps_nIterations"], nIterationsLayerInit=params[
                                                      "eps_nIterationsLayerInit"], nIterationsGlobalInit=params["eps_nIterationsGlobalInit"],
                                                  nIterationsFinal=params["eps_nIterationsFinal"])



nLayerTop = params["hierarchy_top"]
nLayerFinest = params["hierarchy_depth"]

evaluationData = {}
evaluationData["time_iterate"] = 0.
evaluationData["time_sinkhorn"] = 0.
evaluationData["time_refine"] = 0.
evaluationData["time_measureBalancing"] = 0.
evaluationData["time_measureTruncation"] = 0.
evaluationData["time_bounding_box"] = 0.
evaluationData["time_clustering"] = 0.
evaluationData["time_join_clusters"] = 0.
evaluationData["time_PYpi"] = 0.
evaluationData["time_check_scores"] = 0.
evaluationData["timeList_global"] = []
evaluationData["sparsity_muYAtomicEntries"] = []
evaluationData["shape_muYAtomicEntries"] = []
######################################
# First stage
# Warm start with sinkhorn solver
######################################

globalTime1 = time.perf_counter()
info = dict()

# nLayerFinest=nLayerTop
nLayer = nLayerTop
nLayerSinkhornLast = params["nLayerSinkhornLast"]
print("eps\ttime\tscore\tNiters")
while nLayer <= min(nLayerSinkhornLast, nLayerFinest):

    ################################################################################################################################
    timeRefine1 = time.perf_counter()
    print("layer: {:d}".format(nLayer))
    # setup hiarchy layer

    # keep old info for a little bit longer
    if nLayer > nLayerTop:
        # Get previous cell marginals
        # TODO: generalize for 3D
        alpha_old = alpha
        muXLOld = muXL
        muYLOld = muYL

    # End remove
    muXL = muX_layers[nLayer]
    muYL = muY_layers[nLayer]
    shapeXL = muXL.shape
    shapeYL = muYL.shape
    


    dx = 2.0**(nLayerFinest - nLayer)

    x = torch.arange(shapeXL[0], **torch_options) * dx
    x = x.reshape(1, -1)

    if nLayer == nLayerTop:
        alpha = torch.zeros(shapeXL, **torch_options)

    else:

        if params["domdec_refineAlpha"]:
            # Inerpolate previous alpha field

            alpha = torch.nn.functional.interpolate(
                alpha_old[None, None, :, :], scale_factor=2,
                mode="bilinear").squeeze()
        else:
            alpha = torch.zeros(shapeXL, **torch_options)


    timeRefine2 = time.perf_counter()
    evaluationData["time_refine"] += timeRefine2-timeRefine1
    ################################################################################################################################


    # run algorithm at layer

    t0 = time.perf_counter()
    for nEps, (eps, nIterationsMax) in enumerate(params["eps_list"][nLayer]):
        # Only need to do one iteration per epsilon
        # for nIterations in range(nIterationsMax):
        time1 = time.perf_counter()
        solver = SinkhornKL(muXL.view(1, *shapeXL), 
            muYL.view(1, *shapeYL), ((x, x), (x, x)), eps, lam,
            alpha_init = alpha.view(1, *shapeXL), 
            inner_iter = params["sinkhorn_inner_iter"],
            max_iter = params["sinkhorn_max_iter"],
            max_error = params["sinkhorn_error_multiplier"]*params["sinkhorn_error"] ,
            max_error_rel = params["sinkhorn_error_rel"])
        solver.iterate_until_max_error()
        alpha = solver.alpha.view(*shapeXL)
        globalTime2 = time.perf_counter()
        primal_score = solver.primal_score()
        print(
            f"{eps:.2f}\t{globalTime2-globalTime1:.1f}\t{primal_score:.1f}\t{solver.Niter}")

      
    nLayer += 1
    evaluationData["time_iterate"] += time.perf_counter() - t0

evaluationData["time_warm_up"] = time.perf_counter() - globalTime1


###################################
# Transition to DomDec
###################################
balance_transition = True if nLayerSinkhornLast == nLayerTop else False
muY_basic_box, PXpi, basic_mass, alphaFieldEven, current_basic_score = \
    DomDecGPU.sinkhorn_to_domdec(solver, cellsize, balance_transition)
basic_shape = tuple(s//cellsize for s in shapeXL)

####################################
# Second stage
# DomDec iterations
####################################

nLayer = nLayerSinkhornLast + 1
while nLayer <= nLayerFinest:

    ################################################################################################################################
    timeRefine1 = time.perf_counter()
    print("layer: {:d}".format(nLayer))
    # setup hiarchy layer

    cellsize = params["domdec_cellsize"]
    # keep old info for a little bit longer
    # Get previous cell marginals
    # TODO: generalize for 3D
    b1, b2 = basic_shape
    c1, c2 = b1//2, b2//2
    muY_basic_box_old = muY_basic_box
    muXLOld = muXL
    muYLOld = muYL
    basic_mass_old = basic_mass
    basic_shape_old = basic_shape
    basic_shape_pad_old = (b1+2, b2+2)
    shapeXL_old = shapeXL
    # alphaFieldEven computed in previous cell if just after sinkhorn iter
    # otherwise computed below:
    if nLayer > nLayerSinkhornLast + 1:

        alphaFieldEven = DomDecGPU.get_alpha_field_even_gpu(
            alphaA, alphaB, shapeXL, shapeXL_pad,
            cellsize, basic_shape, muXL_np)

    # End remove
    muXL = muX_layers[nLayer]
    muYL = muY_layers[nLayer]
    muXL_np = muXL.cpu().numpy().ravel()
    shapeXL = muXL.shape
    shapeYL = muYL.shape

    # Create padding
    shapeXL_pad = tuple(s + 2*cellsize for s in shapeXL)
    muXLpad = DomDecGPU.pad_tensor(muXL, cellsize, pad_value=0.0)

    # TODO: we now know how to build all this without recurring to partitions

    basic_shape = tuple(i//cellsize for i in shapeXL)
    basic_shape_pad = tuple(i//cellsize for i in shapeXL_pad)
    composite_shape_A = tuple(i//(2*cellsize) for i in shapeXL)
    # TODO: replace also
    b1, b2 = basic_shape
    c1, c2 = composite_shape_A

    muXA = muXL.view(c1, 2*cellsize, c2, 2*cellsize).permute(0, 2, 1, 3) \
        .reshape(-1, 2*cellsize, 2*cellsize)

    muXB = muXLpad.reshape(c1+1, 2*cellsize, c2+1, 2*cellsize).permute(0, 2, 1, 3) \
        .reshape(-1, 2*cellsize, 2*cellsize)

    # Get X grid coordinates
    X_width = shapeXL_pad[1]
    # dx = posXL[1, 1] - posXL[0, 1]
    dx = 2.0**(nLayerFinest - nLayer)
    dxs = torch.tensor([dx, dx])
    dys = torch.tensor([dx, dx])
    dxs_dys = (dxs, dys)

    # TODO: do this directly
    basic_index_pad = torch.arange(
        (b1+2)*(b2+2), **torch_options_int).view(b1+2, b2+2)
    min_index_cell_A = basic_index_pad[1:-1, 1:-1].reshape(c1, 2, c2, 2) \
        .permute(0, 2, 1, 3).reshape(c1*c2, -1).amin(-1)
    min_index_cell_B = basic_index_pad.reshape(c1+1, 2, c2+1, 2) \
        .permute(0, 2, 1, 3).reshape((c1+1)*(c2+1), -1).amin(-1)

    # The following threw a warning
    # leftA = (min_index_cell_A // (b2+2))*cellsize
    # leftB = (min_index_cell_B // (b2+2))*cellsize
    leftA = (torch.div(min_index_cell_A, b2+2, rounding_mode="trunc") -1)*cellsize
    leftB = (torch.div(min_index_cell_B, b2+2, rounding_mode="trunc") -1)*cellsize

    bottomA = (min_index_cell_A % (b2+2) - 1)*cellsize  # Remove left padding
    bottomB = (min_index_cell_B % (b2+2) - 1)*cellsize

    offsetsA = torch.cat((leftA[:,None], bottomA[:,None]), 1)
    offsetsB = torch.cat((leftB[:,None], bottomB[:,None]), 1)

    muXA_box = DomDecGPU.BoundingBox(muXA, offsetsA, shapeXL)
    muXB_box = DomDecGPU.BoundingBox(muXB, offsetsB, shapeXL_pad)

    x1A, x2A = DomDecGPU.get_grid_cartesian_coordinates(muXA_box, dxs)
    x1B, x2B = DomDecGPU.get_grid_cartesian_coordinates(muXB_box, dxs)

    posXA = (x1A, x2A)
    posXB = (x1B, x2B)
    # # Get current dx
    # dx = (x1A[0, 1] - x1A[0, 0]).item()

    basic_mass = muXL.view(b1, cellsize, b2, cellsize).sum((1, 3))

    # Generate partitions
    basic_index = torch.arange(b1*b2, **torch_options_int).reshape(b1, b2)
    partA = basic_index.view(c1, 2, c2, 2).permute(0,2,1,3).reshape(-1, 4)
    basic_index_B = DomDecGPU.pad_tensor(basic_index, 1, pad_value = -1)

    partB = basic_index_B.view(c1+1, 2, c2+1, 2).permute(0,2,1,3).reshape(-1,4)


    # Refinement step

    # Use PXpi to provide basic mass
    # TODO: remove role of info["PXpiB"]
    if nLayer == nLayerSinkhornLast+1:
        PXpi_old = PXpi
    else:
        PXpi_old = get_global_X_marginal(info["PXpiB"], shapeXL_old, "B")
    PXpi_old_density = PXpi_old / muXLOld
    shape_scale = (shapeXL[0]//2, 2, shapeXL[1]//2, 2)
    PXpi = ( muXL.reshape(shape_scale) * PXpi_old_density[:,None,:,None] ).reshape(shapeXL)
    PXpi_pad = DomDecGPU.pad_tensor(PXpi, cellsize, pad_value=1e-40)
    PXpiA = PXpi.view(c1, 2*cellsize, c2, 2*cellsize).permute(0, 2, 1, 3) \
                .reshape(-1, 2*cellsize, 2*cellsize)

    PXpiB = PXpi_pad.reshape(c1+1, 2*cellsize, c2+1, 2*cellsize).permute(0, 2, 1, 3) \
                .reshape(-1, 2*cellsize, 2*cellsize)

    PXpi_basic_old = PXpi_old.view(b1//2, cellsize, b2//2, cellsize).sum((1,3))
    PXpi_basic = PXpi.view(b1, cellsize, b2, cellsize).sum((1,3))

    # Print PXpi
    #print(PXpi_basic_old.ravel(),"\n", muY_basic_box_old.data.sum((1,2)).ravel())

    # Refine current score
    transport_score, margX_score, margY_score = current_basic_score
    transport_score = transport_score.view(b1//2, 1, b2//2, 1).expand((-1, 2, -1, 2)).contiguous() / 4
    margX_score = margX_score.view(b1//2, 1, b2//2, 1).expand((-1, 2, -1, 2)).contiguous() / 4
    current_basic_score = transport_score.ravel(), margX_score.ravel(), margY_score

    # Scaling with PXpi
    muY_basic_box = DomDecGPU.refine_marginals_CUDA(muY_basic_box_old, PXpi_basic_old, PXpi_basic, muYLOld, muYL)
    # Scaling with muXL (previous)
    #muY_basic_box = DomDecGPU.refine_marginals_CUDA(muY_basic_box_old, basic_mass_old, basic_mass, muYLOld, muYL)

    if params["domdec_refineAlpha"]:
        # Interpolate previous alpha field

        alphaA = torch.nn.functional.interpolate(
            alphaFieldEven[None, None, :, :], scale_factor=2,
            mode="bilinear").squeeze()

        # Init alphaB, using padding
        alphaB = DomDecGPU.pad_tensor(alphaA, cellsize, 0.0)

        # Get shape of A-composite cells grid
        alphaA = alphaA.view(c1, 2*cellsize, c2, 2*cellsize) \
            .permute(0, 2, 1, 3).contiguous() \
            .view(-1, 2*cellsize, 2*cellsize)
        # Shape of B-composite cell grid is that of A plus 1 in every dim.
        alphaB = alphaB.view(c1+1, 2*cellsize, c2+1, 2*cellsize) \
            .permute(0, 2, 1, 3).contiguous() \
            .view(-1, 2*cellsize, 2*cellsize)
    else:
        alphaA = torch.zeros(shapeXL, **torch_options)
        alphaA = alphaA.view(-1, 2*cellsize, 2*cellsize)
        alphaB = torch.zeros(shapeXL_pad,  **torch_options)
        alphaB = alphaB.view(-1, 2*cellsize, 2*cellsize)

    # Perform a set of balanced iterations
    eps = params["eps_list"][nLayer][1][0]
    # Get PXPi
    
    timeRefine2 = time.perf_counter()
    evaluationData["time_refine"] += timeRefine2-timeRefine1
    ################################################################################################################################

    # Clustering parameters
    N_clusters = 4
    params["batchsize"] = np.inf

    if params["aux_printLayerConsistency"]:
        # TODO: rephrase muY_basic
        # muXAtomicSums = np.array([np.sum(a) for a in muYAtomicDataList])
        muXAtomicSums = torch.sum(muY_basic_box.data, dim=(2, 3))
        print("layer partition consistency: ", torch.sum(
            torch.abs(muXAtomicSums-basic_mass)))

    # run algorithm at layer
    for nEps, (eps, nIterationsMax) in enumerate(params["eps_list"][nLayer]):
        print("eps: {:f}".format(eps))
        if globalTime2 - globalTime1 > params["max_time"]:
            # Set iterations to minimum so that it exists as it is
            params["sinkhorn_max_iter"] = 0
        for nIterations in range(nIterationsMax):

            #################################
            # dump finest_pre
            if params["aux_dump_finest_pre"]:
                if (nLayer == nLayerFinest) and (nEps == 0) and (nIterations == 0):
                    print("dumping to file: aux_dump_finest_pre...")
                    with open(params["setup_dumpfile_finest_pre"], 'wb') as f:
                        pickle.dump([muXL, muYL, eps, dxs_dys,
                                     muY_basic_box.data.cpu(), 
                                     muY_basic_box.offsets.cpu(),
                                     muXA.cpu(), alphaA.cpu(),
                                     muXB.cpu(), alphaB.cpu()], f, 2)
                    print("dumping done.")
                    # ParallelMap.Close(comm)
                    # quit()
            ################################
            # iteration A
            time1 = time.perf_counter()

            alphaA, muY_basic_box, info, current_basic_score = \
            DomDecUnbalanced.MiniBatchIterateUnbalanced(
                muYL, posY, dxs_dys, eps, lam,
                muXA, posXA, alphaA, muY_basic_box, shapeYL, partA,
                current_basic_score,
                SinkhornError=params["sinkhorn_error"],
                SinkhornErrorRel=params["sinkhorn_error_rel"],
                SinkhornMaxIter=params["sinkhorn_max_iter"],
                SinkhornInnerIter=params["sinkhorn_inner_iter"],
                batchsize=params["batchsize"],
                N_clusters = N_clusters,
                balance = params["balance"],
                safeguard_threshold = params["safeguard_threshold"]
                    )
            

            time2 = time.perf_counter()
            evaluationData["time_iterate"] += time2-time1
            evaluationData["time_sinkhorn"] += info["time_sinkhorn"]
            evaluationData["time_measureBalancing"] += info["time_balance"]
            evaluationData["time_measureTruncation"] += info["time_truncation"]
            evaluationData["time_bounding_box"] += info["time_bounding_box"]
            evaluationData["time_clustering"] += info["time_clustering"]
            evaluationData["time_join_clusters"] += info["time_join_clusters"]
            evaluationData["time_PYpi"] += info["time_PYpi"]
            evaluationData["time_check_scores"] += info["time_check_scores"]
            Niter_per_batch = info["Niter"]
            # primal_score = DomDecUnbalanced.compute_primal_score(
            #     info["solver"], muY_basic_box, muYL
            # # )
            # t, mX, mY = DomDecUnbalanced.compute_primal_score_components(
            #     info["solver"], muY_basic_box, muYL
            # )
            t, mX, mY = current_basic_score
            t, mX, mY = t.sum().item(), mX.sum().item(), mY.sum().item()
            primal_score = t + mX + mY
            print(f"total\t{primal_score:.1f}\t{t:.1f}\t{mX:.1f}\t{mY:.1f}")
            if np.isnan(primal_score):
                assert False
            ################################
            # count total entries in muYAtomicList:
            bbox_size = tuple(muY_basic_box.data.shape[1:])
            nrEntries = int(torch.sum(muY_basic_box.data > 0).item())
            shape_muY_basic = int(np.prod(muY_basic_box.data.shape))
            #print(f"basic bbox: {bbox_size}")
            evaluationData["sparsity_muYAtomicEntries"].append(
                [nLayer, nEps, nIterations, 0, nrEntries])
            evaluationData["shape_muYAtomicEntries"].append(
                [nLayer, nEps, nIterations, 0, shape_muY_basic])
            
            ################################
            # dump after each iteration
            if params["aux_dump_after_each_iter"]:
                if (nLayer == nLayerFinest) and (nEps == 0):
                    print("dumping to file: after iter...")
                    with open(getDumpName("afterIter_nIter{:d}_A".format(nIterations)), 'wb') as f:
                        pickle.dump([muXL, muYL, eps, dxs_dys,
                                     muY_basic_box.data.cpu(), 
                                     muY_basic_box.offsets.cpu(),
                                     muXA.cpu(), alphaA.cpu(),
                                     muXB.cpu(), alphaB.cpu()], f, 2)
                    print("dumping done.")
            ################################

            ################################
            # global time
            globalTime2 = time.perf_counter()
            # print("time:", globalTime2-globalTime1)
            evaluationData["timeList_global"].append(
                [nLayer, nEps, nIterations, 0, globalTime2-globalTime1])
            # printTopic(evaluationData, "time")
            ################################
            print(f"{eps:.2f}\t{globalTime2-globalTime1:.1f}\t{primal_score:.1f}")
            # print(f"{eps:.2f}\t{globalTime2-globalTime1:.1f}\t{primal_score:.1f}\t{Niter_per_batch},{info['bounding_box']}")

            ################################
            # iteration B
            if globalTime2 - globalTime1 > params["max_time"]:
                # Set iterations to minimum so that it exists as it is
                params["sinkhorn_max_iter"] = 0
            time1 = time.perf_counter()
            alphaB, muY_basic_box, info, current_basic_score = \
            DomDecUnbalanced.MiniBatchIterateUnbalanced(
                muYL, posY, dxs_dys, eps, lam,
                muXB, posXB, alphaB, muY_basic_box, shapeYL, partB, current_basic_score,
                SinkhornError=params["sinkhorn_error"],
                SinkhornErrorRel=params["sinkhorn_error_rel"],
                SinkhornMaxIter=params["sinkhorn_max_iter"],
                SinkhornInnerIter=params["sinkhorn_inner_iter"],
                batchsize=params["batchsize"],
                N_clusters = N_clusters,
                balance = params["balance"],
                safeguard_threshold = params["safeguard_threshold"]
            )
            
            time2 = time.perf_counter()

            # primal_score = DomDecUnbalanced.compute_primal_score(
            #     info["solver"], muY_basic_box, muYL
            # )

            # t, mX, mY = DomDecUnbalanced.compute_primal_score_components(
            #     info["solver"], muY_basic_box, muYL
            # )
            
            t, mX, mY = current_basic_score
            t, mX, mY = t.sum().item(), mX.sum().item(), mY.sum().item()
            primal_score = t + mX + mY
            print(f"total\t{primal_score:.1f}\t{t:.1f}\t{mX:.1f}\t{mY:.1f}")
            evaluationData["time_iterate"] += time2-time1
            evaluationData["time_sinkhorn"] += info["time_sinkhorn"]
            evaluationData["time_measureBalancing"] += info["time_balance"]
            evaluationData["time_measureTruncation"] += info["time_truncation"]
            evaluationData["time_bounding_box"] += info["time_bounding_box"]
            evaluationData["time_clustering"] += info["time_clustering"]
            evaluationData["time_join_clusters"] += info["time_join_clusters"]
            evaluationData["time_PYpi"] += info["time_PYpi"]
            evaluationData["time_check_scores"] += info["time_check_scores"]
            Niter_per_batch = info["Niter"]

            
            ################################
            # count total entries in muYAtomicList:
            bbox_size = tuple(muY_basic_box.data.shape[1:])
            nrEntries = int(torch.sum(muY_basic_box.data > 0).item())
            shape_muY_basic = int(np.prod(muY_basic_box.data.shape))
            #print(f"basic bbox: {bbox_size}")
            evaluationData["sparsity_muYAtomicEntries"].append(
                [nLayer, nEps, nIterations, 1, nrEntries])
            evaluationData["shape_muYAtomicEntries"].append(
                [nLayer, nEps, nIterations, 1, shape_muY_basic])


            #################################
            # dump after each iteration
            if params["aux_dump_after_each_iter"]:
                if (nLayer == nLayerFinest) and (nEps == 0):
                    print("dumping to file: after iter...")
                    with open(getDumpName("afterIter_nIter{:d}_B".format(nIterations)), 'wb') as f:
                        pickle.dump([muXL, muYL, eps, dxs_dys,
                                     muY_basic_box.data.cpu(), 
                                     muY_basic_box.offsets.cpu(),
                                     muXA.cpu(), alphaA.cpu(),
                                     muXB.cpu(), alphaB.cpu()], f, 2)
                    print("dumping done.")
            ################################

            ################################
            if params["aux_printLayerConsistency"]:
                muXAtomicSums = torch.sum(muY_basic_box.data, dim=(2, 3))
                print("layer partition consistency: ", torch.sum(
                    torch.abs(muXAtomicSums-basic_mass)))
            ################################

            ################################
            # global time
            globalTime2 = time.perf_counter()
            #print("time:", aglobalTime2-globalTime1)
            evaluationData["timeList_global"].append(
                [nLayer, nEps, nIterations, 1, globalTime2-globalTime1])
            #printTopic(evaluationData, "time")
            ################################
            print(f"{eps:.2f}\t{globalTime2-globalTime1:.1f}\t{primal_score:.1f}")
            # print(f"{eps:.2f}\t{globalTime2-globalTime1:.1f}\t{primal_score:.1f}\t{Niter_per_batch},{info['bounding_box']}")

        #################################
        # dump after each eps on finest layer
        if params["aux_dump_after_each_eps"]:
            if nLayer == nLayerFinest:
                print("dumping to file: after eps...")
                with open(getDumpName("afterEps_nEps{:d}".format(nEps)), 'wb') as f:
                    pickle.dump([muXL, muYL, eps, dxs_dys,
                                 muY_basic_box.data.cpu(), 
                                 muY_basic_box.offsets.cpu(),
                                 muXA.cpu(), alphaA.cpu(),
                                 muXB.cpu(), alphaB.cpu()], f, 2)
                print("dumping done.")
        ################################

    #################################
    # dump after each layer
    if params["aux_dump_after_each_layer"]:
        print("dumping to file: after layer...")
        with open(getDumpName("afterLayer_l{:d}".format(nLayer)), 'wb') as f:
            pickle.dump([muXL, muYL, eps, dxs_dys,
                         muY_basic_box.data.cpu(), 
                         muY_basic_box.offsets.cpu(),
                         muXA.cpu(), alphaA.cpu(),
                         muXB.cpu(), alphaB.cpu()], f, 2)
        print("dumping done.")
    ################################

    # refine

    nLayer += 1


#################################
# dump finest
if params["aux_dump_finest"]:
    print("dumping to file: aux_dump_finest...")
    with open(params["setup_dumpfile_finest"], 'wb') as f:
        pickle.dump([muXL.cpu(), muYL.cpu(), eps, dxs_dys,
                     muY_basic_box.data.cpu(), 
                     muY_basic_box.offsets.cpu(),
                     muXA.cpu(), alphaA.cpu(),
                     muXB.cpu(), alphaB.cpu()], f, 2)
    print("dumping done.")


#####################################
# evaluate primal and dual score
if params["aux_evaluate_scores"]:

    solution_infos = dict()

    # Get smooth alpha
    alpha_global = DomDecGPU.get_alpha_field_even_gpu(
        alphaA, alphaB, shapeXL, shapeXL_pad,
        cellsize, basic_shape, muXL_np)

    # Get beta with sinkhorn iteration
    x = torch.arange(shapeXL[0], **torch_options) * dx
    x = x.reshape(1, -1)
    C = ((x,x), (x,x))
    solver_global = LogSinkhornGPU.UnbalancedSinkhornCudaImageOffset(
        muXL.view(1, *shapeX), muYL.view(1, *shapeY), C, eps, lam,
        alpha_init = alpha_global.view(1, *shapeX))
    solver_global.iterate(0)
    beta_global = solver_global.beta.squeeze()

    # Dual Score
    dual_score = solver_global.dual_score()
    solution_infos["scoreDual"] = dual_score

    solution_infos["scorePrimal"] = primal_score
    PXpi = get_global_X_marginal(info["PXpiB"], shapeXL, "B")
    PXpi_opt = torch.exp(-alpha_global / lam) * muXL
    PYpi = DomDecUnbalanced.get_current_Y_marginal(muY_basic_box, shapeYL)
    PYpi_opt = torch.exp(-beta_global / lam) * muYL
    solution_infos["errorMargX"] = torch.norm(PXpi - PXpi_opt, p = 1).item()
    solution_infos["errorMargY"] = torch.norm(PYpi - PYpi_opt, p = 1).item()
    # solution_infos["errorMargX"] = muX_error
    solution_infos["scoreGap"] = primal_score - dual_score
    solution_infos["scoreGapRel"] = (primal_score - dual_score)/primal_score
    # muY error
    # current_muY = DomDecGPU.get_current_Y_marginal(muY_basic_box, shapeYL) 
    # muY_error = torch.abs(current_muY.ravel() - muYL.ravel()).sum().item()
    # solution_infos["errorMargY"] = muY_error

    print("===================")
    print("solution infos")
    print(json.dumps(solution_infos, indent = 4))
    print("===================")
    #

    for k in solution_infos.keys():
        evaluationData["solution_"+k] = solution_infos[k]


#####################################
print(evaluationData)
# dump evaluationData into json result file:
if params["dump_results"]:
    with open(params["setup_resultfile"], "w") as f:
        json.dump(evaluationData, f)
