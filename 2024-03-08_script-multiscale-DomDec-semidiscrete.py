import time
from lib.header_script import *
import lib.Common as Common

import lib.DomainDecomposition as DomDec
import lib.DomainDecompositionGPU as DomDecGPU
import lib.DomainDecompositionHybrid as DomDecHybrid
import lib.MultiScaleOT as MultiScaleOT

import LogSinkhornGPU

import os
import psutil
import time

import torch
import numpy as np
import pickle

from lib.header_params import *
from lib.AuxConv import *


device = "cuda"
# torch_dtype = torch.float32
torch_dtype = torch.float64
numpy_dtype = np.float32


torch_options = dict(dtype=torch_dtype, device=device)
torch_options_int = dict(dtype=torch.int32, device=device)

###############################################################
###############################################################

def get_dxs(pos, shape, dtype):
    pos_res = pos.reshape((*shape, 2))
    dx1 = pos_res[1,0,0] - pos_res[0,0,0] if shape[0] > 1 else 1.0
    dx2 = pos_res[0,1,1] - pos_res[0,0,1] if shape[1] > 1 else 1.0
    dxs = torch.tensor([dx1, dx2], device = "cpu", dtype = dtype)
    return dxs


# read parameters from command line and cfg file
print(sys.argv)
print("setting script parameters")
params = getDefaultParams()
ScriptTools.getCommandLineParameters(params, paramsListCommandLine)

params["setup_cfgfile"] = "experiments/"+params["setup_tag"]+".txt"
params["setup_resultfile"] = "experiments/"+params["setup_tag"] + \
    "_results_gpu.txt"

params["setup_dumpfile_finest_pre"] = "experiments/" + \
    params["setup_tag"]+"_dump_finest_pre.dat"
params["setup_dumpfile_finest"] = "experiments/" + \
    params["setup_tag"]+"_dump_finest.dat"


def getDumpName(tag):
    return "experiments/"+params["setup_tag"]+"_dump_"+tag+".dat"


params.update(ScriptTools.readParameters(
    params["setup_cfgfile"], paramsListCFGFile))


# overwrite parallelization settings if nWorkers==0:
# if nWorkers==0:
#     print("set parallelization mode to serial")
#     params["parallel_iteration"]=False
#     params["parallel_truncation"]=False
#     params["parallel_balancing"]=False
#     params["parallel_refinement"]=False


print("final parameter settings")
for k in sorted(params.keys()):
    print("\t", k, params[k])

###############################################################
###############################################################

# Manual parameters
params["aux_dump_after_each_layer"] = False
params["aux_dump_finest"] = True  # TODO: change
params["aux_evaluate_scores"] = True
params["sinkhorn_max_iter"] = 2000

params["aux_dump_after_each_eps"] = False
params["aux_dump_after_each_iter"] = False
params["domdec_cellsize"] = 4
params["hierarchy_top"] = int(np.log2(params["domdec_cellsize"])) + 1

# load input measures from file
# do some preprocessing and setup multiscale representation of them
muX, posX, shapeX = Common.importMeasure(params["setup_fn1"])
muY, posY, shapeY = Common.importMeasure(params["setup_fn2"])

N = shapeX[0]
posX *= N
posY *= N

cellsize = params["domdec_cellsize"]
# params["batchsize"] = np.inf
params["clustering"] = True


# Get multiscale torch hierarchy
muX_final = torch.tensor(muX, **torch_options).view(shapeX)
muY_final = torch.tensor(muY, **torch_options).view(shapeY)
muX_layers = DomDecGPU.get_multiscale_layers(muX_final, shapeX)
# All muY layers are the same
# muY_layers = DomDecGPU.get_multiscale_layers(muY_final, shapeY)

# muX: 1d array of masses
# posX: (*,dim) array of locations of masses
# shapeX: dimensions of original spatial grid on which the masses live
#    useful for visualization and setting up the domdec cells

# convert pos arrays to double for c++ compatibility
posXD = posX.astype(np.float64)
posYD = posY.astype(np.float64)

# TODO: remove this if not needed
# generate multi-scale representation of muX
print(muX.dtype)
MultiScaleSetupX = MultiScaleOT.TMultiScaleSetup(
    posXD, muX, params["hierarchy_depth"], childMode=MultiScaleOT.childModeGrid, setup=True, setupDuals=False, setupRadii=False)

# No multiscale setup for muY
# MultiScaleSetupY = MultiScaleOT.TMultiScaleSetup(
#     posYD, muY, params["hierarchy_depth"], childMode=MultiScaleOT.childModeGrid, setup=True, setupDuals=False, setupRadii=False)


# setup eps scaling
if params["eps_schedule"] == "default":
    params["eps_list"] = Common.getEpsListDefault(params["hierarchy_depth"], params["hierarchy_top"],
                                                  params["eps_base"], params["eps_layerFactor"], params["eps_layerSteps"], params["eps_stepsFinal"],
                                                  nIterations=params["eps_nIterations"], nIterationsLayerInit=params[
                                                      "eps_nIterationsLayerInit"], nIterationsGlobalInit=params["eps_nIterationsGlobalInit"],
                                                  nIterationsFinal=params["eps_nIterationsFinal"])

    # for l in params["eps_list"]:
    #     for i, (eps, nsteps) in enumerate(l):
    #         l[i] = [eps*(params["domdec_cellsize"]/4)**2, nsteps]

    # params["eps_list"][-1].append([0.5, 1])
    # params["eps_list"][-1].append([0.25, 1])
    print(params["eps_list"])


evaluationData = {}
evaluationData["time_iterate"] = 0.
evaluationData["time_sinkhorn"] = 0.
evaluationData["time_refine"] = 0.
evaluationData["time_measureBalancing"] = 0.
evaluationData["time_measureTruncation"] = 0.
evaluationData["time_bounding_box"] = 0.
evaluationData["time_clustering"] = 0.
evaluationData["time_join_clusters"] = 0.


evaluationData["timeList_global"] = []

evaluationData["sparsity_muYAtomicEntries"] = []

globalTime1 = time.perf_counter()


nLayerTop = params["hierarchy_top"]
nLayerFinest = params["hierarchy_depth"]
# nLayerFinest=nLayerTop
nLayer = nLayerTop
while nLayer <= nLayerFinest:

    ################################################################################################################################
    timeRefine1 = time.perf_counter()
    print("layer: {:d}".format(nLayer))
    # setup hiarchy layer

    cellsize = params["domdec_cellsize"]
    # keep old info for a little bit longer
    if nLayer > nLayerTop:
        # Get previous cell marginals
        # TODO: generalize for 3D
        b1, b2 = basic_shape
        c1, c2 = b1//2, b2//2
        muY_basic_old = muY_basic
        left_old = left
        bottom_old = bottom
        muXLOld = muXL
        muYLOld = muYL
        basic_mass_old = basic_mass
        basic_shape_old = basic_shape
        basic_shape_pad_old = basic_shape_pad
        if params["domdec_refineAlpha"]:
            # Remove dummy basic cells for refinement
            alphaFieldEven = DomDecGPU.get_alpha_field_even_gpu(
                alphaA, alphaB, shapeXL, shapeXL_pad,
                cellsize, basic_shape, muXL_np)

    # basic data of current layer
    # TODO: how to get new layer size directly from multiscale solver?
    # TODO: Remove this if not used
    muXL_np = MultiScaleSetupX.getMeasure(nLayer)
    # muYL_np = MultiScaleSetupY.getMeasure(nLayer)
    muYL_np = muY
    posXL = MultiScaleSetupX.getPoints(nLayer)
    # posYL = MultiScaleSetupY.getPoints(nLayer)
    posYL = posY
    # This is not used anymore
    # parentsXL = MultiScaleSetupX.getParents(nLayer)
    # parentsYL = MultiScaleSetupY.getParents(nLayer)
    # parentsYL = MultiScaleSetupY.getParents(nLayer)
    # End remove
    muXL = muX_layers[nLayer]
    # muYL = muY_layers[nLayer]
    muYL = muY_final
    shapeXL = muXL.shape
    shapeYL = muYL.shape

    # Create padding
    shapeXL_pad = tuple(s + 2*cellsize for s in shapeXL)
    muXLpad = DomDecGPU.pad_tensor(muXL, cellsize, pad_value=1e-40)

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
    print(posXL.shape, shapeXL)
    dxs = get_dxs(posXL, shapeXL, torch_dtype)
    dys = get_dxs(posYL, shapeYL, torch_dtype)

    dxs_dys = (dxs, dys)
    # We still assume x grid is equispaced
    dx = dxs_dys[0][0].item()
    # dx = posXL[1, 1] - posXL[0, 1]

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

    bottomA = (min_index_cell_A % (b2+2) - 1)*cellsize
    bottomB = (min_index_cell_B % (b2+2) - 1)*cellsize

    x1A, x2A = DomDecGPU.get_grid_cartesian_coordinates(
        leftA, bottomA, 2*cellsize, 2*cellsize, dxs
    )
    x1B, x2B = DomDecGPU.get_grid_cartesian_coordinates(
        leftB, bottomB, 2*cellsize, 2*cellsize, dxs
    )
    posXA = (x1A, x2A)
    posXB = (x1B, x2B)

    # print(posXA)
    # print(posXB)
    # print(posY)

    basic_mass = muXL.view(b1, cellsize, b2, cellsize).sum((1, 3))

    # Generate partitions
    basic_index = torch.arange(b1*b2, **torch_options_int).reshape(b1, b2)
    partA = basic_index.view(c1, 2, c2, 2).permute(0,2,1,3).reshape(-1, 4)
    basic_index_B = DomDecGPU.pad_tensor(basic_index, 1, pad_value = -1)

    partB = basic_index_B.view(c1+1, 2, c2+1, 2).permute(0,2,1,3).reshape(-1,4)


    if nLayer == nLayerTop:
        # Init cell marginals for first layer
        # TODO: first layer
        # muYAtomicDataList = [muYL_np*m for m in atomicCellMasses]  # product plan
        # muYAtomicIndicesList = [
        #     np.arange(muYL_np.shape[0], dtype=np.int32)
        #     for _ in range(len(atomicCells))
        # ]
        muY_basic = basic_mass.view(-1, 1, 1) * muYL.view(1, *shapeYL)
        B = muY_basic.shape[0]
        left = torch.zeros(B, **torch_options_int)
        bottom = torch.zeros(B, **torch_options_int)

        alphaA = torch.zeros(shapeXL, **torch_options)
        alphaA = alphaA.view(-1, 2*cellsize, 2*cellsize)
        alphaB = torch.zeros(shapeXL_pad, **torch_options)
        alphaB = alphaB.view(-1, 2*cellsize, 2*cellsize)

    else:
        # refine atomic Y marginals from previous layer

        # muY_basic, left, bottom = DomDecGPU.refine_marginals_CUDA(
        #     muY_basic_old, left_old, bottom_old,
        #     basic_mass_old, basic_mass, muYLOld, muYL
        # )
        #
        # For semidiscrete lebesgue we just need to copy along basic dimension
        # and renormalize
        # TODO: just generate dummy muYL and make a reduction afterwards to 
        # bring back to right size. We only need muYLdummy
        muYLdummy = (1/8) * torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],           
            [0, 0, 1, 1]            
        ], **torch_options)

        muY_basic, left, bottom = DomDecGPU.refine_marginals_CUDA(
            muY_basic_old, left_old, bottom_old,
            basic_mass_old, basic_mass, muYLOld + 1e-40, muYLdummy # small offset avoids division by zero
        )
        # Reduce over dummy dimensions
        muY_basic = muY_basic.view(-1, 2, 2, 2, 2).sum((-3, -1))
        # Left and bottom need to be renormalized
        left = left // 2
        bottom = bottom // 2
        # print(left_old)
        # print(left)
        # print(bottom_old)
        # print(bottom)
        # print(muY_basic_old)
        # print(muY_basic)

        print(muY_basic_old.shape, muY_basic.shape, left.shape, bottom.shape)

        if params["domdec_refineAlpha"]:
            # Interpolate previous alpha field

            alphaA = torch.nn.functional.interpolate(
                alphaFieldEven[None, None, :, :], scale_factor=2,
                mode="bilinear").squeeze()

            # Init alphaB, using padding
            alphaB = DomDecGPU.pad_tensor(alphaA, cellsize, 0.0)
            # Reshape and permute dims
            # TODO: generalize for 3D
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

    # set up new empty beta lists:
    betaADataList = [None for i in range(alphaA.shape[0])]
    betaAIndexList = [None for i in range(alphaA.shape[0])]
    betaBDataList = [None for i in range(alphaB.shape[0])]
    betaBIndexList = [None for i in range(alphaB.shape[0])]

    timeRefine2 = time.perf_counter()
    evaluationData["time_refine"] += timeRefine2-timeRefine1
    ################################################################################################################################

    # Clustering parameters
    params["number_clusters"] = 1 # for semidiscrete we don't need so much
    N_clusters = params["number_clusters"]
    # params["batchsize"] = max((N//(2*cellsize) + 1)**2 // N_clusters, 2000)
    params["batchsize"] = (N//(2*cellsize) + 1)**2 // N_clusters

    if params["aux_printLayerConsistency"]:
        # TODO: rephrase muY_basic
        # muXAtomicSums = np.array([np.sum(a) for a in muYAtomicDataList])
        muXAtomicSums = torch.sum(muY_basic, dim=(2, 3))
        print("layer partition consistency: ", torch.sum(
            torch.abs(muXAtomicSums-basic_mass)))

    # TODO: don't reshape
    # muY_basic = muY_basic.view(*basic_shape, *muY_basic.shape[1:])
    # left = left.view(basic_shape)
    # bottom = bottom.view(basic_shape)
    # print("\n\n")
    # print("muY_basic:", muY_basic.shape)
    # print("left\n", left)
    # print("bottom\n", bottom)

    # run algorithm at layer
    for nEps, (eps, nIterationsMax) in enumerate(params["eps_list"][nLayer]):
        print("eps: {:f}".format(eps)) # TODO: Probably we need to rescale eps here.
        for nIterations in range(nIterationsMax):

            #################################
            # dump finest_pre
            if params["aux_dump_finest_pre"]:
                if (nLayer == nLayerFinest) and (nEps == 0) and (nIterations == 0):
                    print("dumping to file: aux_dump_finest_pre...")
                    with open(params["setup_dumpfile_finest_pre"], 'wb') as f:
                        pickle.dump([muXL, muYL, eps, dxs_dys,
                                     muY_basic, left, bottom,
                                     muXA.cpu(), alphaA.cpu(),
                                     muXB.cpu(), alphaB.cpu()], f, 2)
                    print("dumping done.")
                    # ParallelMap.Close(comm)
                    # quit()

            ################################
            # iteration A
            time1 = time.perf_counter()
            alphaA, muY_basic, left, bottom, info = DomDecGPU.MiniBatchIterate(
                muYL, posY, dxs_dys, eps,
                muXA, posXA, alphaA, muY_basic, left, bottom, shapeY, partA,
                SinkhornError=params["sinkhorn_error"],
                SinkhornErrorRel=params["sinkhorn_error_rel"],
                SinkhornMaxIter=params["sinkhorn_max_iter"],
                SinkhornInnerIter=params["sinkhorn_inner_iter"],
                batchsize=params["batchsize"],
                clustering = params["clustering"],
                N_clusters = params["number_clusters"]
            )

            time2 = time.perf_counter()
            evaluationData["time_iterate"] += time2-time1
            evaluationData["time_sinkhorn"] += info["time_sinkhorn"]
            evaluationData["time_measureBalancing"] += info["time_balance"]
            evaluationData["time_measureTruncation"] += info["time_truncation"]
            evaluationData["time_bounding_box"] += info["time_bounding_box"]
            evaluationData["time_clustering"] += info["time_clustering"]
            evaluationData["time_join_clusters"] += info["time_join_clusters"]
            print(
                f"Niter = {info['solver'][0].Niter}, bounding box = {info['bounding_box']}")

            ################################
            # count total entries in muYAtomicList:
            bbox_size = tuple(muY_basic.shape[1:])
            nrEntries = int(np.prod(muY_basic.shape))
            print(f"bbox: {bbox_size}")
            evaluationData["sparsity_muYAtomicEntries"].append(
                [nLayer, nEps, nIterations, 0, nrEntries])
            
            ################################
            # dump after each iteration
            if params["aux_dump_after_each_iter"]:
                if (nLayer == nLayerFinest) and (nEps == 0):
                    print("dumping to file: after iter...")
                    with open(getDumpName("afterIter_nIter{:d}_A".format(nIterations)), 'wb') as f:
                        pickle.dump([muXL, muYL, eps, dxs_dys,
                                     muY_basic, left, bottom,
                                     muXA.cpu(), alphaA.cpu(),
                                     muXB.cpu(), alphaB.cpu()], f, 2)
                    print("dumping done.")
            ################################

            ################################
            # global time
            globalTime2 = time.perf_counter()
            print("time:", globalTime2-globalTime1)
            evaluationData["timeList_global"].append(
                [nLayer, nEps, nIterations, 0, globalTime2-globalTime1])
            printTopic(evaluationData, "time")
            ################################

            ################################
            # iteration B
            time1 = time.perf_counter()
            alphaB, muY_basic, left, bottom, info = DomDecGPU.MiniBatchIterate(
                muYL, posY, dxs_dys, eps,
                muXB, posXB, alphaB, muY_basic, left, bottom, shapeY, partB,
                SinkhornError=params["sinkhorn_error"],
                SinkhornErrorRel=params["sinkhorn_error_rel"],
                SinkhornMaxIter=params["sinkhorn_max_iter"],
                SinkhornInnerIter=params["sinkhorn_inner_iter"],
                batchsize=params["batchsize"],
                clustering = params["clustering"],
                N_clusters = params["number_clusters"]
            )
            time2 = time.perf_counter()
            evaluationData["time_iterate"] += time2-time1
            evaluationData["time_sinkhorn"] += info["time_sinkhorn"]
            evaluationData["time_measureBalancing"] += info["time_balance"]
            evaluationData["time_measureTruncation"] += info["time_truncation"]
            evaluationData["time_bounding_box"] += info["time_bounding_box"]
            print(
                f"Niter = {info['solver'][0].Niter}, bounding box = {info['bounding_box']}")
            ################################

            #################################
            # dump after each iteration
            if params["aux_dump_after_each_iter"]:
                if (nLayer == nLayerFinest) and (nEps == 0):
                    print("dumping to file: after iter...")
                    with open(getDumpName("afterIter_nIter{:d}_B".format(nIterations)), 'wb') as f:
                        pickle.dump([muXL, muYL, eps, dxs_dys,
                                     muY_basic, left, bottom,
                                     muXA.cpu(), alphaA.cpu(),
                                     muXB.cpu(), alphaB.cpu()], f, 2)
                    print("dumping done.")
            ################################

            ################################
            if params["aux_printLayerConsistency"]:
                muXAtomicSums = torch.sum(muY_basic, dim=(2, 3))
                print("layer partition consistency: ", torch.sum(
                    torch.abs(muXAtomicSums-basic_mass)))
            ################################

            ################################
            # global time
            globalTime2 = time.perf_counter()
            print("time:", globalTime2-globalTime1)
            evaluationData["timeList_global"].append(
                [nLayer, nEps, nIterations, 1, globalTime2-globalTime1])
            printTopic(evaluationData, "time")
            ################################

        #################################
        # dump after each eps on finest layer
        if params["aux_dump_after_each_eps"]:
            if nLayer == nLayerFinest:
                print("dumping to file: after eps...")
                with open(getDumpName("afterEps_nEps{:d}".format(nEps)), 'wb') as f:
                    pickle.dump([muXL, muYL, eps, dxs_dys,
                                 muY_basic, left, bottom,
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
                         muY_basic, left, bottom,
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
        pickle.dump([muXL, muYL, eps, dxs_dys,
                     muY_basic, left, bottom,
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
    print(alpha_global)

    # Get beta with sinkhorn iteration
    left_global = torch.zeros(1, **torch_options_int)
    bottom_global = torch.zeros(1, **torch_options_int)
    posX_global = DomDecGPU.get_grid_cartesian_coordinates(
        left_global, bottom_global, *shapeX, dxs
    )
    posY_global = DomDecGPU.get_grid_cartesian_coordinates(
        left_global, bottom_global, *shapeY, dys
    )
    # print(posX_global)
    # print(posY_global)
    # print(left.view(*basic_shape))
    C = (posX_global, posY_global)
    solver_global = LogSinkhornGPU.LogSinkhornCudaImageOffset(
        muXL.view(1, *shapeXL), muYL.view(1, *shapeYL), C, eps, 
        alpha_init = alpha_global.view(1, *shapeX))


    # solver_global = LogSinkhorn(
    #     muXL.view(1, *shapeX), muYL.view(1, *shapeY), dx, eps,
    #     alpha_init = alpha_global.view(1, *shapeX))
    error = solver_global.iterate(1)
    print("error", error)
    beta_global = solver_global.beta.squeeze()

    # Dual Score
    dual_score = torch.sum(solver_global.alpha * solver_global.mu) + \
        torch.sum(solver_global.beta * solver_global.nu)
    dual_score = dual_score.item()
    solution_infos["scoreDual"] = dual_score

    # Get primal score
    # Get updated nu comp by doing a dummy domdec iteration
    _, _, _, _, info = DomDecGPU.MiniBatchIterate(
        muYL, posY, dxs_dys, eps,
        muXB, posXB, alphaB, muY_basic, left, bottom, shapeY, partB,
        SinkhornError=params["sinkhorn_error"],
        SinkhornErrorRel=params["sinkhorn_error_rel"],
        SinkhornMaxIter=0,
        SinkhornInnerIter=0,
        batchsize=params["batchsize"],
        clustering = params["clustering"],
        N_clusters = params["number_clusters"]
    )
    # Get primal_score and muX_error
    primal_score = 0.0
    muX_error = 0.0
    for solverB in info["solver"]:
        primal_score += (torch.sum(solverB.alpha * solverB.mu) + torch.sum(solverB.beta * solverB.nu)).item()
        new_alpha = solverB.get_new_alpha()
        current_mu = solverB.mu * torch.exp((solverB.alpha - new_alpha)/eps)
        muX_error += torch.sum(torch.abs(solverB.mu - current_mu)).item()
    solution_infos["scorePrimal"] = primal_score
    solution_infos["errorMargX"] = muX_error
    solution_infos["scoreGap"] = primal_score - dual_score
    solution_infos["scoreGapRel"] = (primal_score - dual_score)/primal_score
    # muY error
    current_muY = DomDecGPU.get_current_Y_marginal(muY_basic, left, bottom, shapeYL) 
    muY_error = torch.abs(current_muY.ravel() - muYL.ravel()).sum().item()
    solution_infos["errorMargY"] = muY_error

    print("===================")
    print("solution infos")
    print(json.dumps(solution_infos, indent = 4))
    print("===================")
    #


    for k in solution_infos.keys():
        evaluationData["solution_"+k] = solution_infos[k]



#####################################
# dump evaluationData into json result file:
# print(evaluationData)
with open(params["setup_resultfile"], "w") as f:
    json.dump(evaluationData, f)
