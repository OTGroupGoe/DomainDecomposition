import time
from lib.header_script import *
import lib.Common as Common
import torch

from pykeops.torch import LazyTensor
# from LogSinkhornGPU import LogSinkhornCudaImage, AbstractSinkhorn
from LogSinkhornGPU import LogSinkhornCudaImage, LogSinkhornKeopsImage, LogSinkhornCudaImageOffset
import lib.MultiScaleOT as MultiScaleOT

import os
import psutil
import time

import numpy as np
import pickle

from lib.header_params import *
from lib.AuxConv import *

def get_dxs(pos, shape, dtype):
    pos_res = pos.reshape((*shape, 2))
    dx1 = pos_res[1,0,0] - pos_res[0,0,0] if shape[0] > 1 else 1.0
    dx2 = pos_res[0,1,1] - pos_res[0,0,1] if shape[1] > 1 else 1.0
    dxs = torch.tensor([dx1, dx2], device = "cpu", dtype = dtype)
    return dxs

def get_xs(pos, shape, dtype):
    pos_res = pos.reshape((*shape, 2))
    x1 = torch.tensor(pos_res[:,0,0], dtype = dtype, device = "cuda").reshape(1, -1)
    x2 = torch.tensor(pos_res[0,:,1], dtype = dtype, device = "cuda").reshape(1, -1)
    return (x1, x2)

def get_multiscale_layers(muX, shapeX):
    # TODO: Generalize for measures with sizes not powers of 2
    assert len(shapeX) == 2, "only implemented for 2d tensors"
    assert shapeX[0] == shapeX[1], "only implemented for square tensors"
    muX_i = muX
    depth_X = int(np.log2(shapeX[0]))
    muX_layers = [muX]
    for i in range(depth_X):
        n = shapeX[0] // 2**(i+1)
        muX_i = muX_i.view(n, 2, n, 2).sum((1, 3))
        muX_layers.append(muX_i)
    muX_layers.reverse()
    return muX_layers


device = "cuda"
# torch_dtype = torch.float32
torch_dtype = torch.float64


torch_options = dict(dtype=torch_dtype, device=device)
torch_options_int = dict(dtype=torch.int32, device=device)

###############################################################
###############################################################

# read parameters from command line and cfg file
print(sys.argv)
print("setting script parameters")
params = getDefaultParams()
ScriptTools.getCommandLineParameters(params, paramsListCommandLine)

params["setup_cfgfile"] = "experiments/"+params["setup_tag"]+".txt"
params["setup_resultfile"] = "experiments/"+params["setup_tag"] + \
    "_results_sinkhorn.txt"

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



###############################################################
###############################################################

# Manual parameters
params["aux_dump_after_each_layer"] = False
params["aux_dump_finest"] = False  # TODO: change
params["aux_evaluate_scores"] = True  # TODO: allow evaluation
params["sinkhorn_max_iter"] = 10000
params["sinkhorn_inner_iter"] = 10
params["sinkhorn_error"] = 1e-4
sinkhorn_subsolver = "cuda"
params["sinkhorn_subsolver"] = sinkhorn_subsolver



params["aux_dump_after_each_eps"] = False
params["aux_dump_after_each_iter"] = False
params["hierarchy_top"] = 3


print("final parameter settings")
for k in sorted(params.keys()):
    print("\t", k, params[k])

print(f"Solving with {sinkhorn_subsolver} solver")
# load input measures from file
# do some preprocessing and setup multiscale representation of them
muX, posX, shapeX = Common.importMeasure(params["setup_fn1"])
muY, posY, shapeY = Common.importMeasure(params["setup_fn2"])
N = shapeX[0]
posX *= N
posY *= N
posY = posY - posY[[0],:]

print(f"N = {N}")
print(torch_dtype)


# Get multiscale torch hierarchy
muX_final = torch.tensor(muX, **torch_options).view(shapeX)
muY_final = torch.tensor(muY, **torch_options).view(shapeY)
muX_layers = get_multiscale_layers(muX_final, shapeX)

MultiScaleSetupX = MultiScaleOT.TMultiScaleSetup(
    posX.astype(np.float64), muX, params["hierarchy_depth"], 
    childMode=MultiScaleOT.childModeGrid, setup=True, setupDuals=False, setupRadii=False)

# muY_layers = get_multiscale_layers(muY_final, shapeY)

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

    # Add a couple more iterations
    # for l in params["eps_list"]:
    #     for i, (eps, nsteps) in enumerate(l):
    #         l[i] = [eps*(params["domdec_cellsize"]/4)**2, nsteps]

    # params["eps_list"][-1].append([0.5, 1])
    # params["eps_list"][-1].append([0.25, 1])
    # print(params["eps_list"])


evaluationData = {}
evaluationData["time_iterate"] = 0.
evaluationData["time_sinkhorn"] = 0.
evaluationData["time_refine"] = 0.


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
        alpha_old = alpha
        muXLOld = muXL
        muYLOld = muYL

    # End remove
    muXL = muX_layers[nLayer]
    muYL = muY_final
    shapeXL = muXL.shape
    shapeYL = muYL.shape
    posXL = MultiScaleSetupX.getPoints(nLayer)
    posYL = posY
    xs = get_xs(posXL, shapeXL, torch_dtype)
    ys = get_xs(posYL, shapeYL, torch_dtype)
    # print(xs)
    # print(ys)

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
        print("eps: {:f}".format(eps),end = "\t")
        # Only need to do one iteration per epsilon
        # for nIterations in range(nIterationsMax):
        for nIterations in range(1):
            # iteration A
            time1 = time.perf_counter()
            if sinkhorn_subsolver == "cuda":
                solver = LogSinkhornCudaImageOffset(muXL.view(1, *shapeXL), 
                    muYL.view(1, *shapeYL), (xs, ys), eps, 
                    alpha_init = alpha.view(1, *shapeXL), 
                    inner_iter = params["sinkhorn_inner_iter"],
                    max_iter = params["sinkhorn_max_iter"],
                    max_error = params["sinkhorn_error"],
                    max_error_rel = params["sinkhorn_error_rel"])
            elif sinkhorn_subsolver == "keops":
                xs = tuple(xi.view(-1) for xi in xs)
                ys = tuple(yi.view(-1) for yi in ys)
                solver = LogSinkhornKeopsImage(muXL.view(1, *shapeXL), 
                    muYL.view(1, *shapeYL), (xs, ys), eps, 
                    alpha_init = alpha.view(1, *shapeXL), 
                    inner_iter = params["sinkhorn_inner_iter"],
                    max_iter = params["sinkhorn_max_iter"],
                    max_error = params["sinkhorn_error"],
                    max_error_rel = params["sinkhorn_error_rel"])
            else:
                raise NotImplementedError("Solver not implemented")
            solver.iterate_until_max_error()
            print("Niters", solver.Niter)
            alpha = solver.alpha.view(*shapeXL)
    nLayer += 1
    evaluationData["time_iterate"] += time.perf_counter() - t0

print(evaluationData)


# #####################################
# # evaluate primal and dual score
if params["aux_evaluate_scores"]:
    solution_infos = dict()
    # Get muX error
    new_alpha = solver.get_new_alpha()
    current_mu = solver.mu * torch.exp((solver.alpha - new_alpha)/eps)
    muX_error = torch.sum(torch.abs(solver.mu - current_mu)).item()

    # Get dual score to compare with domdec
    dual_score = torch.sum(solver.alpha * solver.mu) + \
        torch.sum(solver.beta * solver.nu)
    dual_score = dual_score.item()
    solution_infos["scoreDual"] = dual_score

    solution_infos["errorMargX"] = muX_error
    print("===================")
    print("solution infos")
    print(json.dumps(solution_infos, indent = 4))
    print("===================")


    
#     for k in solution_infos.keys():
#         evaluationData["solution_"+k] = solution_infos[k]

# print("Total time =", evaluationData["time_iterate"])

# # #####################################
# # dump evaluationData into json result file:
# with open(params["setup_resultfile"], "w") as f:
#     json.dump(evaluationData, f)
