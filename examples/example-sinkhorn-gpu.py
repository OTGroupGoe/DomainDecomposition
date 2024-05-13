import time
import argparse
import sys
sys.path.append("../")
from lib.header_script import *
import lib.Common as Common
import torch

# from LogSinkhornGPU import LogSinkhornCudaImage, AbstractSinkhorn
from LogSinkhornGPU import LogSinkhornCudaImage, LogSinkhornKeopsImage
import time
import numpy as np

from lib.header_params import *
from lib.AuxConv import *

###############################################################################
# # GPU multiscale Sinkhorn
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
###############################################################################

def get_multiscale_layers(muX, shapeX):
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

# read parameters from command line and cfg file
print("setting script parameters")
params = getDefaultParams()

# Input data files
params["setup_fn1"] = "data/f-000-256.pickle"
params["setup_fn2"] = "data/f-001-256.pickle"

# Sinkhorn parameters
params["sinkhorn_max_iter"] = 10000
params["sinkhorn_inner_iter"] = 10
params["sinkhorn_error"] = 1e-4
params["sinkhorn_error_rel"] = True
params["sinkhorn_subsolver"] = "cuda" # options are cuda | keops

# Multiscale parameters
params["hierarchy_top"] = 3

# Dump files
params["setup_resultfile"] = "results-sinkhorn-gpu.txt"
params["setup_dumpfile_finest"] = "dump-sinkhorn-gpu.dat"
params["aux_dump_finest"] = False 
params["aux_evaluate_scores"] = True 

# Print parameters
print("final parameter settings")
for k in sorted(params.keys()):
    print("\t", k, params[k])

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

print(f"Solving with {params['sinkhorn_subsolver']} solver")
# load input measures from file
# do some preprocessing and setup multiscale representation of them
muX, posX, shapeX = Common.importMeasure(params["setup_fn1"])
muY, posY, shapeY = Common.importMeasure(params["setup_fn2"])
N = shapeX[0]
params["hierarchy_depth"] = int(np.log2(N))

print(f"N = {N}")
print(torch_dtype)


# Get multiscale torch hierarchy
muX_final = torch.tensor(muX, **torch_options).view(shapeX)
muY_final = torch.tensor(muY, **torch_options).view(shapeX)
muX_layers = get_multiscale_layers(muX_final, shapeX)
muY_layers = get_multiscale_layers(muY_final, shapeY)

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
    muYL = muY_layers[nLayer]
    shapeXL = muXL.shape
    shapeYL = muYL.shape
    


    dx = 2.0**(nLayerFinest - nLayer)

    x = torch.arange(shapeXL[0], **torch_options) * dx

    print("dx =", dx)


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
            if params["sinkhorn_subsolver"] == "cuda":
                solver = LogSinkhornCudaImage(muXL.view(1, *shapeXL), 
                    muYL.view(1, *shapeYL), ((x, x), (x, x)), eps, 
                    alpha_init = alpha.view(1, *shapeXL), 
                    inner_iter = params["sinkhorn_inner_iter"],
                    max_iter = params["sinkhorn_max_iter"],
                    max_error = params["sinkhorn_error"],
                    max_error_rel = params["sinkhorn_error_rel"])
            elif params["sinkhorn_subsolver"] == "keops":
                solver = LogSinkhornKeopsImage(muXL.view(1, *shapeXL), 
                    muYL.view(1, *shapeYL), ((x, x), (x, x)), eps, 
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
    
    for k in solution_infos.keys():
        evaluationData["solution_"+k] = solution_infos[k]

# #####################################
# dump evaluationData into json result file:
with open(params["setup_resultfile"], "w") as f:
    json.dump(evaluationData, f)
