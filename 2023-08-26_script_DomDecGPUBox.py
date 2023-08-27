import time
from lib.header_script import *
import lib.Common as Common

import lib.DomainDecomposition as DomDec
import lib.DomainDecompositionGPU as DomDecGPU
import lib.DomainDecompositionHybrid as DomDecHybrid
import lib.MultiScaleOT as MultiScaleOT

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

###############################################################
###############################################################

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
params["aux_dump_after_each_layer"] = True
params["aux_dump_finest"] = True # TODO: change
params["aux_evaluate_scores"] = False # TODO: allow evaluation
params["sinkhorn_max_iter"] = 2000
params["aux_dump_after_each_eps"] = False
params["aux_dump_after_each_iter"] = False
params["domdec_cellsize"] = 4
params["hierarchy_top"] = int(np.log2(params["domdec_cellsize"])) + 1
params["batchsize"] = np.inf

# load input measures from file
# do some preprocessing and setup multiscale representation of them
muX, posX, shapeX = Common.importMeasure(params["setup_fn1"])
muY, posY, shapeY = Common.importMeasure(params["setup_fn2"])

# muX: 1d array of masses
# posX: (*,dim) array of locations of masses
# shapeX: dimensions of original spatial grid on which the masses live
#    useful for visualization and setting up the domdec cells

# convert pos arrays to double for c++ compatibility
posXD = posX.astype(np.float64)
posYD = posY.astype(np.float64)

# generate multi-scale representation of muX
MultiScaleSetupX = MultiScaleOT.TMultiScaleSetup(
    posXD, muX, params["hierarchy_depth"], childMode=MultiScaleOT.childModeGrid, setup=True, setupDuals=False, setupRadii=False)

MultiScaleSetupY = MultiScaleOT.TMultiScaleSetup(
    posYD, muY, params["hierarchy_depth"], childMode=MultiScaleOT.childModeGrid, setup=True, setupDuals=False, setupRadii=False)


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
evaluationData["timeList_global"] = []

evaluationData["sparsity_muYAtomicEntries"] = []

globalTime1 = time.time()


nLayerTop = params["hierarchy_top"]
nLayerFinest = params["hierarchy_depth"]
# nLayerFinest=nLayerTop
nLayer = nLayerTop
while nLayer <= nLayerFinest:

    ################################################################################################################################
    timeRefine1 = time.time()
    print("layer: {:d}".format(nLayer))
    # setup hiarchy layer

    cellsize = params["domdec_cellsize"]
    # keep old info for a little bit longer
    if nLayer > nLayerTop:
        # Get previous cell marginals
        # TODO: generalize for 3D
        b1, b2 = Nu_basic.shape[:2]
        c1, c2 = b1//2, b2//2
        geom_shape = Nu_basic.shape[2:]


        # print("left old\n", left)
        # print("bottom old\n", bottom)
        # print(Nu_basic.shape, left.shape, bottom.shape)
        Nu_basic_old = Nu_basic.reshape(b1*b2, *geom_shape).cpu().numpy()
        left_old = left.reshape(b1*b2).cpu().numpy()
        bottom_old = bottom.reshape(b1*b2).cpu().numpy()
        muYAtomicIndicesListOld, muYAtomicDataListOld = \
            DomDecGPU.unpack_cell_marginals_2D_box(
                Nu_basic_old, left_old, bottom_old, shapeYL
            )
        atomicCellsOld = atomicCells
        muXLOld = muXL
        muYLOld = muYL
        atomicCellMassesOld = atomicCellMasses
        metaCellShapeOld = metaCellShape
        if params["domdec_refineAlpha"]:
            # Remove dummy basic cells for refinement
            metaCellShapeTrue = tuple(s - 2 for s in metaCellShape)
            alphaFieldEven = DomDecGPU.get_alpha_field_even_gpu(
                alphaA, alphaB, shapeXL, shapeXL_pad,
                cellsize, metaCellShapeTrue, muXL)

    # basic data of current layer
    # TODO: how to get new layer size directly from multiscale solver?
    shapeXL = [2**(nLayer) for i in range(params["setup_dim"])]
    shapeYL = [2**(nLayer) for i in range(params["setup_dim"])]
    muXL = MultiScaleSetupX.getMeasure(nLayer)
    muYL = MultiScaleSetupY.getMeasure(nLayer)
    posXL = MultiScaleSetupX.getPoints(nLayer)
    posYL = MultiScaleSetupY.getPoints(nLayer)
    parentsXL = MultiScaleSetupX.getParents(nLayer)
    parentsYL = MultiScaleSetupY.getParents(nLayer)

    # Create padding
    shapeXL_pad = tuple(s + 2*cellsize for s in shapeXL)
    muXLpad = DomDecGPU.pad_array(
        muXL.reshape(shapeXL), cellsize, pad_value=1e-40
    ).ravel()

    # create partition into atomic cells
    atomicCells = DomDec.GetPartitionIndices2D(shapeXL_pad, cellsize, 0)
    # join atomic cells into two stacked partitions
    metaCellShape = [i//cellsize for i in shapeXL_pad]

    # For the GPU the A cells have offset of cellsize, since they ignore the padding
    partitionMetaCellsA = DomDec.GetPartitionIndices2D(metaCellShape, 2, 1)
    partitionMetaCellsB = DomDec.GetPartitionIndices2D(metaCellShape, 2, 0)

    # Remove non square composite cells
    comp_size = 2**params["setup_dim"]

    partitionMetaCellsA = [
        cell for cell in partitionMetaCellsA if len(cell) == comp_size
    ]

    # Create partition
    partitionDataA = DomDec.GetPartitionData(atomicCells, partitionMetaCellsA)
    partitionDataB = DomDec.GetPartitionData(atomicCells, partitionMetaCellsB)

    partitionDataACompCells = partitionDataA[1]
    partitionDataACompCellIndices = [
        np.array([partitionDataA[2][j][1:3] for j in x], dtype=np.int32)
        for x in partitionDataACompCells
    ]
    partitionDataBCompCells = partitionDataB[1]
    partitionDataBCompCellIndices = [
        np.array([partitionDataB[2][j][1:3] for j in x], dtype=np.int32)
        for x in partitionDataBCompCells
    ]

    # Generate problem data from partition data. Reshape it already for GPU use.
    muXA = torch.tensor(np.array([muXLpad[cell] for cell in partitionDataA[0]],
                                 dtype=numpy_dtype)).cuda()
    muXB = torch.tensor(np.array([muXLpad[cell] for cell in partitionDataB[0]],
                                 dtype=numpy_dtype)).cuda()
    # Need to permute dims
    muXA = muXA.view(-1, 2, 2, cellsize, cellsize).permute((0, 1, 3, 2, 4)) \
               .contiguous().view(-1, 2*cellsize, 2*cellsize)
    muXB = muXB.view(-1, 2, 2, cellsize, cellsize).permute((0, 1, 3, 2, 4)) \
               .contiguous().view(-1, 2*cellsize, 2*cellsize)
    
    # Get X grid coordinates
    X_width = shapeXL_pad[1]
    dx = posXL[1,1] - posXL[0,1]

    leftA = torch.tensor([cell[0]//X_width for cell in partitionDataA[0]], 
                         device = device)
    leftB = torch.tensor([cell[0]//X_width for cell in partitionDataB[0]], 
                         device = device)
    bottomA = torch.tensor([cell[0]%X_width for cell in partitionDataA[0]], 
                           device = device)
    bottomB = torch.tensor([cell[0]%X_width for cell in partitionDataB[0]], 
                           device = device)

    x1A, x2A = DomDecGPU.get_grid_cartesian_coordinates(
                            leftA, bottomA, 2*cellsize, 2*cellsize, dx
                            )
    x1B, x2B = DomDecGPU.get_grid_cartesian_coordinates(
                            leftB, bottomB, 2*cellsize, 2*cellsize, dx
                            )
    posXA = (x1A, x2A)
    posXB = (x1B, x2B)
    # Get current dx
    dx = (x1A[0,1] - x1A[0,0]).item()


    atomicCellMasses = np.array([np.sum(muXLpad[cell]) for cell in atomicCells])

    if nLayer == nLayerTop:
        # Init cell marginals for first layer
        muYAtomicDataList = [muYL*m for m in atomicCellMasses]  # product plan
        muYAtomicIndicesList = [
            np.arange(muYL.shape[0], dtype=np.int32)
            for _ in range(len(atomicCells))
        ]
        alphaA = torch.zeros(shapeXL, device=device, dtype=torch_dtype)
        alphaA = alphaA.view(-1, 2*cellsize, 2*cellsize)
        alphaB = torch.zeros(shapeXL_pad, device=device, dtype=torch_dtype)
        alphaB = alphaB.view(-1, 2*cellsize, 2*cellsize)

    else:
        # refine atomic Y marginals from previous layer

        # if params["parallel_refinement"]:

        #     muYAtomicDataList, muYAtomicIndicesList = DomDecParallelMPI.GetRefinedAtomicYMarginals_SparseY(
        #         comm,
        #         muYL, muYLOld, parentsYL,
        #         atomicCellMasses, atomicCellMassesOld,
        #         atomicCells, atomicCellsOld,
        #         muYAtomicDataListOld, muYAtomicIndicesListOld,
        #         metaCellShape,
        #         MPIchunksize=params["MPI_chunksize"], MPIprobetime=params["MPI_probetime"])
        # else:
        muYAtomicDataList, muYAtomicIndicesList = \
            DomDecGPU.get_refined_marginals_gpu(
                muYL, muYLOld, parentsYL,
                atomicCellMasses, atomicCellMassesOld,
                atomicCells, atomicCellsOld,
                muYAtomicDataListOld, muYAtomicIndicesListOld,
                metaCellShape, metaCellShapeOld
            )

        if params["domdec_refineAlpha"]:
            # extract alpha values on each cell from refinement of evened alpha field
            # alphaAList=[alphaFieldEven[parentsXL[indices]] for indices in partitionDataA[0]]
            # alphaBList=[alphaFieldEven[parentsXL[indices]] for indices in partitionDataB[0]]

            # this is pretty hacky right now. refine initial dual variables
            # posXLOld=MultiScaleSetupX.getPoints(nLayer-1)
            # shapeXLOld=[2**(nLayer-1) for i in range(params["setup_dim"])]
            # shapeXLOldGrid=posXLOld.reshape((tuple(shapeXLOld)+(2,)))
            # interpX=shapeXLOldGrid[:,0,0]
            # interpY=shapeXLOldGrid[0,:,1]
            # interpAlpha=alphaFieldEven.reshape(shapeXLOld)
            # interp=scipy.interpolate.RectBivariateSpline(\
            #        interpX, interpY, interpAlpha, kx=1, ky=1)
            # alphaFieldEvenNew=interp.ev(posXL[:,0],posXL[:,1])

            alphaFieldEvenNew = MultiScaleSetupX.refineSignal(
                alphaFieldEven, nLayer-1, 1)
            # Copy directly to init alphaA
            alphaA = torch.tensor(alphaFieldEvenNew, device=device, dtype = torch_dtype)
            alphaA = alphaA.view(shapeXL)
            # Init alphaB, using padding
            alphaB = DomDecGPU.pad_tensor(alphaA, cellsize, 0.0)
            # Reshape and permute dims
            # TODO: generalize for 3D
            # Get shape of A-composite cells grid
            s1, s2 = tuple(s // (2*cellsize) for s in shapeXL)
            alphaA = alphaA.view(s1, 2*cellsize, s2, 2*cellsize) \
                .permute(0,2,1,3).contiguous().view(-1, 2*cellsize, 2*cellsize)
            # Shape of B-composite cell grid is that of A plus 1 in every dim.
            alphaB = alphaB.view(s1+1, 2*cellsize, s2+1, 2*cellsize) \
                .permute(0,2,1,3).contiguous().view(-1, 2*cellsize, 2*cellsize)
            

            # alphaAList = [alphaFieldEvenNew[indices]
            #               for indices in partitionDataA[0]]
            # alphaBList = [alphaFieldEvenNew[indices]
            #               for indices in partitionDataB[0]]
        else:
            alphaA = torch.zeros(shapeXL,Nu_basic, left, bottom,
                    muYAtomicDataList, muYAtomicIndicesList,
                    muXA.cpu(), alphaA.cpu(),
                    muXB.cpu(), alphaB.cpu(),
                    device=device, dtype=torch_dtype)
            alphaA = alphaA.view(-1, 2*cellsize, 2*cellsize)
            alphaB = torch.zeros(shapeXL_pad, device=device, dtype=torch_dtype)
            alphaB = alphaB.view(-1, 2*cellsize, 2*cellsize)

    # set up new empty beta lists:
    betaADataList = [None for i in partitionDataACompCells]
    betaAIndexList = [None for i in partitionDataACompCells]
    betaBDataList = [None for i in partitionDataBCompCells]
    betaBIndexList = [None for i in partitionDataBCompCells]

    timeRefine2 = time.time()
    evaluationData["time_refine"] += timeRefine2-timeRefine1
    ################################################################################################################################

    if params["aux_printLayerConsistency"]:
        muXAtomicSums = np.array([np.sum(a) for a in muYAtomicDataList])
        print("layer partition consistency: ", np.sum(
            np.abs(muXAtomicSums-atomicCellMasses)))
        
    # Turn sparse lists to nu_basic
    # TODO: at some point, do refinement also in GPU
    torch_options = dict(dtype = torch.float64, device = "cuda")
    torch_options_int = dict(dtype = torch.int32, device = "cuda")
    Nu_basic, _, left, bottom, max_width, max_height = DomDecGPU.batch_cell_marginals_2D(
        muYAtomicIndicesList, muYAtomicDataList, shapeYL, muYL
    )
    # print(Nu_basic.shape, shapeY)
    Nu_basic = torch.tensor(Nu_basic, **torch_options)

    # Trim extra basic cells (we have to do it before when we improve refinement)
    Nu_basic = Nu_basic.view(*metaCellShape,*Nu_basic.shape[1:])[1:-1,1:-1]
    left = torch.tensor(left, **torch_options_int).view(metaCellShape)[1:-1,1:-1]
    bottom = torch.tensor(bottom, **torch_options_int).view(metaCellShape)[1:-1,1:-1]
    # print("\n\n")
    # print("Nu_basic:", Nu_basic.shape)
    # print("left\n", left)
    # print("bottom\n", bottom)

    # run algorithm at layer
    for nEps, (eps, nIterationsMax) in enumerate(params["eps_list"][nLayer]):
        print("eps: {:f}".format(eps))
        for nIterations in range(nIterationsMax):
    
            #################################
            # dump finest_pre
            if params["aux_dump_finest_pre"]:
                if (nLayer==nLayerFinest) and (nEps==0) and (nIterations==0):
                    print("dumping to file: aux_dump_finest_pre...")
                    with open(params["setup_dumpfile_finest_pre"], 'wb') as f:
                        pickle.dump([muYL, posYL, eps,
                                     Nu_basic, left, bottom,
                                     muYAtomicDataList, muYAtomicIndicesList,
                                     muXA.cpu(), alphaA.cpu(),
                                     muXB.cpu(), alphaB.cpu()], f, 2)
                    print("dumping done.")
                    # ParallelMap.Close(comm)
                    # quit()

            ################################
            # iteration A
            time1 = time.time()
            alphaA, Nu_basic, left, bottom, info = DomDecGPU.BatchIterateBox(
                muY,posY,dx,eps,\
                muXA,posXA,alphaA,Nu_basic, left, bottom, shapeY,"A",
                SinkhornError = params["sinkhorn_error"],
                SinkhornErrorRel = params["sinkhorn_error_rel"],
                SinkhornMaxIter = params["sinkhorn_max_iter"], 
                SinkhornInnerIter = params["sinkhorn_inner_iter"],
                BatchSize = params["batchsize"]
            )
            
            time2 = time.time()
            evaluationData["time_iterate"] += time2-time1
            evaluationData["time_sinkhorn"] += info["time_sinkhorn"]
            evaluationData["time_measureBalancing"] += info["time_balance"]
            evaluationData["time_measureTruncation"] += info["time_truncation"]
            evaluationData["time_bounding_box"]  += info["time_bounding_box"]
            print(f"Niter = {info['solver'].Niter}, bounding box = {info['bounding_box']}")
            ################################

            ################################
            # count total entries in muYAtomicList:
            nrEntries = 0
            for a in muYAtomicDataList:
                nrEntries += a.shape[0]
            print("muYAtomicEntries: {:d}".format(nrEntries))
            evaluationData["sparsity_muYAtomicEntries"].append(
                [nLayer, nEps, nIterations, 0, nrEntries])
            ################################

            ################################
            # balancing A
            time1 = time.time()
            # TODO: measure balancing time
            # if params["parallel_balancing"]:
            #     DomDecParallelMPI.ParallelBalanceMeasures(comm, muYAtomicDataList, atomicCellMasses, partitionDataACompCells,
            #                                               MPIchunksize=params["MPI_chunksize"], MPIprobetime=params["MPI_probetime"])
            # else:
            #     DomDec.BalanceMeasuresMultiAll(
            #         muYAtomicDataList, atomicCellMasses, partitionDataACompCells, verbose=False)

            time2 = time.time()
            evaluationData["time_measureBalancing"] += time2-time1
            ################################

            ################################
            # truncation A
            # TODO: measure truncation time
            time1 = time.time()
            # if params["parallel_truncation"]:
            #     DomDecParallelMPI.ParallelTruncateMeasures(comm, muYAtomicDataList, muYAtomicIndicesList, 1E-15,
            #                                                MPIchunksize=params["MPI_chunksize"], MPIprobetime=params["MPI_probetime"])
            # else:
            #     for i in range(len(atomicCells)):
            #         muYAtomicDataList[i], muYAtomicIndicesList[i] = Common.truncateSparseVector(
            #             muYAtomicDataList[i], muYAtomicIndicesList[i], 1E-15)
            time2 = time.time()
            evaluationData["time_measureTruncation"] += time2-time1
            ################################

            #################################
            # dump after each iteration
            if params["aux_dump_after_each_iter"]:
                if (nLayer == nLayerFinest) and (nEps == 0):
                    print("dumping to file: after iter...")
                    with open(getDumpName("afterIter_nIter{:d}_A".format(nIterations)), 'wb') as f:
                        pickle.dump([muYL, posYL, eps,
                                     Nu_basic, left, bottom,
                                     muYAtomicDataList, muYAtomicIndicesList,
                                     muXA.cpu(), alphaA.cpu(),
                                     muXB.cpu(), alphaB.cpu()], f, 2)
                    print("dumping done.")
            ################################

            ################################
            # global time
            globalTime2 = time.time()
            print("time:", globalTime2-globalTime1)
            evaluationData["timeList_global"].append(
                [nLayer, nEps, nIterations, 0, globalTime2-globalTime1])
            printTopic(evaluationData, "time")
            ################################

            ################################
            # count total entries in muYAtomicList:
            nrEntries = 0
            for a in muYAtomicDataList:
                nrEntries += a.shape[0]
            print("muYAtomicEntries: {:d}".format(nrEntries))
            evaluationData["sparsity_muYAtomicEntries"].append(
                [nLayer, nEps, nIterations, 1, nrEntries])
            ################################

            ################################
            # iteration B
            time1 = time.time()
            alphaB, Nu_basic, left, bottom, info = DomDecGPU.BatchIterateBox(
                muY,posY,dx,eps,\
                muXB,posXB,alphaB,Nu_basic, left, bottom, shapeY, "B",
                SinkhornError = params["sinkhorn_error"],
                SinkhornErrorRel = params["sinkhorn_error_rel"],
                SinkhornMaxIter = params["sinkhorn_max_iter"], 
                SinkhornInnerIter = params["sinkhorn_inner_iter"],
                BatchSize = params["batchsize"]
            )
            time2 = time.time()
            evaluationData["time_iterate"] += time2-time1
            evaluationData["time_sinkhorn"] += info["time_sinkhorn"]
            evaluationData["time_measureBalancing"] += info["time_balance"]
            evaluationData["time_measureTruncation"] += info["time_truncation"]
            evaluationData["time_bounding_box"]  += info["time_bounding_box"]
            print(f"Niter = {info['solver'].Niter}, bounding box = {info['bounding_box']}")
            ################################

            ################################
            # balancing B
            time1 = time.time()

            # if params["parallel_balancing"]:
            #     DomDecParallelMPI.ParallelBalanceMeasures(comm, muYAtomicDataList, atomicCellMasses, partitionDataBCompCells,
            #                                               MPIchunksize=params["MPI_chunksize"], MPIprobetime=params["MPI_probetime"])
            # else:
            #     DomDec.BalanceMeasuresMultiAll(
            #         muYAtomicDataList, atomicCellMasses, partitionDataBCompCells, verbose=False)

            time2 = time.time()
            evaluationData["time_measureBalancing"] += time2-time1
            ################################

            ################################
            # truncation B
            time1 = time.time()
            # if params["parallel_truncation"]:
            #     DomDecParallelMPI.ParallelTruncateMeasures(comm, muYAtomicDataList, muYAtomicIndicesList, 1E-15,
            #                                                MPIchunksize=params["MPI_chunksize"], MPIprobetime=params["MPI_probetime"])
            # else:
            #     for i in range(len(atomicCells)):
            #         muYAtomicDataList[i], muYAtomicIndicesList[i] = Common.truncateSparseVector(
            #             muYAtomicDataList[i], muYAtomicIndicesList[i], 1E-15)
            time2 = time.time()
            evaluationData["time_measureTruncation"] += time2-time1
            ################################

            #################################
            # dump after each iteration
            if params["aux_dump_after_each_iter"]:
                if (nLayer == nLayerFinest) and (nEps == 0):
                    print("dumping to file: after iter...")
                    with open(getDumpName("afterIter_nIter{:d}_B".format(nIterations)), 'wb') as f:
                        pickle.dump([muYL, posYL, eps,
                                     Nu_basic, left, bottom,
                                     muYAtomicDataList, muYAtomicIndicesList,
                                     muXA.cpu(), alphaA.cpu(),
                                     muXB.cpu(), alphaB.cpu()], f, 2)
                    print("dumping done.")
            ################################

            ################################
            if params["aux_printLayerConsistency"]:
                muXAtomicSums = np.array([np.sum(a)
                                         for a in muYAtomicDataList])
                print("layer partition consistency: ", np.sum(
                    np.abs(muXAtomicSums-atomicCellMasses)))
            ################################

            ################################
            # global time
            globalTime2 = time.time()
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
                    pickle.dump([muYL, posYL, eps,
                                     Nu_basic, left, bottom,
                                     muYAtomicDataList, muYAtomicIndicesList,
                                     muXA.cpu(), alphaA.cpu(),
                                     muXB.cpu(), alphaB.cpu()], f, 2)
                print("dumping done.")
        ################################

    #################################
    # dump after each layer
    if params["aux_dump_after_each_layer"]:
        print("dumping to file: after layer...")
        with open(getDumpName("afterLayer_l{:d}".format(nLayer)), 'wb') as f:
            pickle.dump([muYL, posYL, eps,
                        Nu_basic, left, bottom,
                        muYAtomicDataList, muYAtomicIndicesList,
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
        pickle.dump([muYL, posYL, eps,
                    Nu_basic, left, bottom,
                    muYAtomicDataList, muYAtomicIndicesList,
                    muXA.cpu(), alphaA.cpu(),
                    muXB.cpu(), alphaB.cpu()], f, 2)
    print("dumping done.")


#####################################
# evaluate primal and dual score
if params["aux_evaluate_scores"]:
    solutionInfos, muYAList = DomDec.getPrimalInfos(muYL, posYL, posXAList, muXAList, alphaAList, betaADataList, betaAIndexList, eps,
                                                    getMuYList=True)

    alphaFieldEven, alphaGraph = DomDec.getAlphaFieldEven(alphaAList, alphaBList,
                                                          partitionDataA[0], partitionDataB[0], shapeXL, metaCellShape, cellsize,
                                                          muX=muXL, requestAlphaGraph=True)

    betaFieldEven = DomDec.glueBetaList(
        betaADataList, betaAIndexList, muXL.shape[0], offsets=alphaGraph.ravel(), muYList=muYAList, muY=muYL)

    MultiScaleSetupX.setupDuals()
    MultiScaleSetupX.setupRadii()
    MultiScaleSetupY.setupDuals()
    MultiScaleSetupY.setupRadii()

    # hierarchical dual score:

    # fix alpha and beta by doing some iterations (expensive)
#        datDual,alphaDual,betaDual=DomDec.getHierarchicalKernel(MultiScaleSetupX,MultiScaleSetupY,\
#                params["setup_dim"],params["hierarchy_depth"],eps,alphaFieldEven,betaFieldEven,nIter=10)
#        print("entries in dual kernel: ",datDual[0].shape[0])
#        solutionInfos["scoreDual"]=np.sum(muX*alphaDual)+np.sum(muYL*betaDual)-eps*np.sum(datDual[0])

    # do one beta-reduce-only-iteration manually
    datDual = DomDec.getHierarchicalKernel(MultiScaleSetupX, MultiScaleSetupY,
                                           params["setup_dim"], params["hierarchy_depth"], eps, alphaFieldEven, betaFieldEven, nIter=0)
    print("entries in dual kernel: ", datDual[0].shape[0])

    piDual = scipy.sparse.csr_matrix(datDual, shape=(
        alphaFieldEven.shape[0], betaFieldEven.shape[0]))
    muYEff = np.array(piDual.sum(axis=0)).ravel()
    vRel = np.minimum(1, muYL/muYEff)

    solutionInfos["scoreDual"] = np.sum(
        muX*alphaFieldEven)+np.sum(muYL*(betaFieldEven+eps*np.log(vRel)))-eps*np.sum(muYEff*vRel)

    solutionInfos["scoreGap"] = solutionInfos["scorePrimal"] - \
        solutionInfos["scoreDual"]

    print(solutionInfos)

    for k in solutionInfos.keys():
        evaluationData["solution_"+k] = solutionInfos[k]


#####################################
# dump evaluationData into json result file:
print(evaluationData)
with open(params["setup_resultfile"], "w") as f:
    json.dump(evaluationData, f)
