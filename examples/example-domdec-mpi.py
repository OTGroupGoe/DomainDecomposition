from mpi4py import MPI
import numpy as np
import time
import sys
sys.path.append("../")
import lib.DomDecParallelMPI as DomDecParallelMPI
import lib.MPIParallelMap as ParallelMap
import argparse

###############################################################################
# # MPI multiscale domain decomposition for entropic optimal transport
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

comm = MPI.COMM_WORLD

rank=comm.Get_rank()
nWorkers=comm.Get_size()-1

if rank>0:
    # if process is worker process, run worker routine (which listens to messages from main process) and eventually terminate
    ParallelMap.Worker(comm)
    quit()

# otherwise, run main process script
try:
    # this try encapsulates the whole main script. if any exception is raised in the main script, the finally clause at the end frees the worker processes    

    from lib.header_script import *
    import lib.Common as Common

    import lib.DomainDecomposition as DomDec
    import lib.MultiScaleOT as MultiScaleOT

    import os
    import psutil
    import time

    import pickle
    
    from lib.header_params import *
    from lib.AuxConv import *


    ###############################################################
    ###############################################################

    # read parameters from command line and cfg file
    # print(sys.argv)
    print(f"Solving with {nWorkers} workers")
    print("setting script parameters")
    params=getDefaultParams()

    # Input data files
    params["setup_fn1"] = "data/f-000-256.pickle"
    params["setup_fn2"] = "data/f-001-256.pickle"

    # Domdec parameters
    params["domdec_cellsize"] = 4
    cellsize = params["domdec_cellsize"]
    params["parallel_balancing"] = True
    params["parallel_refinement"] = True

    # Subproblem Sinkhorn parameters
    params["sinkhorn_max_iter"] = 10000
    params["sinkhorn_inner_iter"] = 10
    params["sinkhorn_error"] = 1e-4
    params["sinkhorn_error_rel"] = True

    # Multiscale parameters
    params["hierarchy_top"] = int(np.log2(params["domdec_cellsize"])) + 1

    # Dump files
    params["setup_resultfile"] = "results-domdec-mpi.txt"
    params["setup_dumpfile_finest"] = "dump-domdec-mpi.dat"
    params["aux_dump_finest"] = False 
    params["aux_evaluate_scores"] = True 

    # Print parameters
    print("final parameter settings")
    for k in sorted(params.keys()):
        print("\t", k, params[k])

    # overwrite parallelization settings if nWorkers==0:
    if nWorkers==0:
        print("set parallelization mode to serial")
        params["parallel_iteration"]=False
        params["parallel_truncation"]=False
        params["parallel_balancing"]=False
        params["parallel_refinement"]=False

    # Allow all parameters to be overriden on the command line
    args = argparse.ArgumentParser()
    for key in params.keys():
        args.add_argument(f"--{key}", dest = key, 
                default = params[key], type = type(params[key]))
    params = vars(args.parse_args())

    ###############################################################
    ###############################################################

    print("final parameter settings")
    for k in sorted(params.keys()):
        print("\t",k,params[k])

    # load input measures from file
    # do some preprocessing and setup multiscale representation of them
    muX,posX,shapeX=Common.importMeasure(params["setup_fn1"])
    muY,posY,shapeY=Common.importMeasure(params["setup_fn2"])
    muX = muX.ravel()
    muY = muY.ravel()
    N = shapeX[0]
    params["hierarchy_depth"] = int(np.log2(N))

    # muX: 1d array of masses
    # posX: (*,dim) array of locations of masses
    # shapeX: dimensions of original spatial grid on which the masses live
    #    useful for visualization and setting up the domdec cells

    # convert pos arrays to double for c++ compatibility
    posXD=posX.astype(np.double)
    posYD=posY.astype(np.double)

    # generate multi-scale representation of muX
    MultiScaleSetupX=MultiScaleOT.TMultiScaleSetup(posXD,muX,params["hierarchy_depth"],childMode=MultiScaleOT.childModeGrid,setup=True,setupDuals=False,setupRadii=False)

    MultiScaleSetupY=MultiScaleOT.TMultiScaleSetup(posYD,muY,params["hierarchy_depth"],childMode=MultiScaleOT.childModeGrid,setup=True,setupDuals=False,setupRadii=False)


    # setup eps scaling
    if params["eps_schedule"]=="default":
        params["eps_list"]=Common.getEpsListDefault(params["hierarchy_depth"],params["hierarchy_top"],\
                params["eps_base"],params["eps_layerFactor"],params["eps_layerSteps"],params["eps_stepsFinal"],\
                nIterations=params["eps_nIterations"],nIterationsLayerInit=params["eps_nIterationsLayerInit"],nIterationsGlobalInit=params["eps_nIterationsGlobalInit"],\
                nIterationsFinal=params["eps_nIterationsFinal"])


    evaluationData={}
    evaluationData["time_iterate"]=0.
    evaluationData["time_refine"]=0.
    evaluationData["time_measureBalancing"]=0.
    evaluationData["time_measureTruncation"]=0.
    evaluationData["timeList_global"]=[]
    
    evaluationData["sparsity_muYAtomicEntries"]=[]
    
    globalTime1=time.time()


    nLayerTop=params["hierarchy_top"]
    nLayerFinest=params["hierarchy_depth"]
    #nLayerFinest=nLayerTop
    nLayer=nLayerTop
    while nLayer<=nLayerFinest: 


        ################################################################################################################################
        timeRefine1=time.time()
        print("layer: {:d}".format(nLayer))
        ## setup hiarchy layer
        
        # keep old info for a little bit longer
        if nLayer>nLayerTop:
            atomicCellsOld=atomicCells
            muYAtomicDataListOld=muYAtomicDataList
            muYAtomicIndicesListOld=muYAtomicIndicesList
            muXLOld=muXL
            muYLOld=muYL
            atomicCellMassesOld=atomicCellMasses
            if params["domdec_refineAlpha"]:
                alphaFieldEven=DomDec.getAlphaFieldEven(alphaAList,alphaBList,\
                        partitionDataA[0],partitionDataB[0],shapeXL,metaCellShape,params["domdec_cellsize"],muXL)
            
        # basic data of current layer
        shapeXL=[2**(nLayer) for i in range(params["setup_dim"])]
        shapeYL=[2**(nLayer) for i in range(params["setup_dim"])]
        muXL=MultiScaleSetupX.getMeasure(nLayer)
        muYL=MultiScaleSetupY.getMeasure(nLayer)
        posXL=MultiScaleSetupX.getPoints(nLayer)
        posYL=MultiScaleSetupY.getPoints(nLayer)
        parentsXL=MultiScaleSetupX.getParents(nLayer)
        parentsYL=MultiScaleSetupY.getParents(nLayer)

        # create partition into atomic cells
        atomicCells=DomDec.GetPartitionIndices2D(shapeXL,params["domdec_cellsize"],0)
        # join atomic cells into two stacked partitions
        metaCellShape=[i//params["domdec_cellsize"] for i in shapeXL]

        partitionMetaCellsA=DomDec.GetPartitionIndices2D(metaCellShape,2,0)
        partitionMetaCellsB=DomDec.GetPartitionIndices2D(metaCellShape,2,1)

        partitionDataA=DomDec.GetPartitionData(atomicCells,partitionMetaCellsA)
        partitionDataB=DomDec.GetPartitionData(atomicCells,partitionMetaCellsB)

        partitionDataACompCells=partitionDataA[1]
        partitionDataACompCellIndices=[np.array([partitionDataA[2][j][1:3] for j in x],dtype=np.int32) for x in partitionDataACompCells]
        partitionDataBCompCells=partitionDataB[1]
        partitionDataBCompCellIndices=[np.array([partitionDataB[2][j][1:3] for j in x],dtype=np.int32) for x in partitionDataBCompCells]


        muXAList=[muXL[cell].copy() for cell in partitionDataA[0]]
        muXBList=[muXL[cell].copy() for cell in partitionDataB[0]]

        posXAList=[posXL[cell].copy() for cell in partitionDataA[0]]
        posXBList=[posXL[cell].copy() for cell in partitionDataB[0]]

        atomicCellMasses=np.array([np.sum(muXL[cell]) for cell in atomicCells])

        if nLayer==nLayerTop:
            muYAtomicDataList=[muYL*m for m in atomicCellMasses]
            muYAtomicIndicesList=[np.arange(muYL.shape[0],dtype=np.int32) for i in range(len(atomicCells))]
            alphaAList=[np.zeros_like(muXAi) for muXAi in muXAList]
            alphaBList=[np.zeros_like(muXBi) for muXBi in muXBList]
            
        else:
            # refine atomic Y marginals from previous layer
            
            if params["parallel_refinement"]:
        
                muYAtomicDataList,muYAtomicIndicesList=DomDecParallelMPI.GetRefinedAtomicYMarginals_SparseY(
                        comm,\
                        muYL,muYLOld,parentsYL,\
                        atomicCellMasses,atomicCellMassesOld,\
                        atomicCells,atomicCellsOld,\
                        muYAtomicDataListOld,muYAtomicIndicesListOld,\
                        metaCellShape,\
                        MPIchunksize=params["MPI_chunksize"],MPIprobetime=params["MPI_probetime"])
            else:
                muYAtomicDataList,muYAtomicIndicesList=DomDec.GetRefinedAtomicYMarginals_SparseY(muYL,muYLOld,parentsYL,\
                        atomicCellMasses,atomicCellMassesOld,\
                        atomicCells,atomicCellsOld,\
                        muYAtomicDataListOld,muYAtomicIndicesListOld,\
                        metaCellShape)



            if params["domdec_refineAlpha"]:
                # extract alpha values on each cell from refinement of evened alpha field
                #alphaAList=[alphaFieldEven[parentsXL[indices]] for indices in partitionDataA[0]]
                #alphaBList=[alphaFieldEven[parentsXL[indices]] for indices in partitionDataB[0]]
                
                # this is pretty hacky right now. refine initial dual variables
                #posXLOld=MultiScaleSetupX.getPoints(nLayer-1)
                #shapeXLOld=[2**(nLayer-1) for i in range(params["setup_dim"])]
                #shapeXLOldGrid=posXLOld.reshape((tuple(shapeXLOld)+(2,)))
                #interpX=shapeXLOldGrid[:,0,0]
                #interpY=shapeXLOldGrid[0,:,1]
                #interpAlpha=alphaFieldEven.reshape(shapeXLOld)
                #interp=scipy.interpolate.RectBivariateSpline(\
                #        interpX, interpY, interpAlpha, kx=1, ky=1)
                #alphaFieldEvenNew=interp.ev(posXL[:,0],posXL[:,1])
                
                alphaFieldEvenNew=MultiScaleSetupX.refineSignal(alphaFieldEven,nLayer-1,1)
                
                alphaAList=[alphaFieldEvenNew[indices] for indices in partitionDataA[0]]
                alphaBList=[alphaFieldEvenNew[indices] for indices in partitionDataB[0]]
            else:
                alphaAList=[np.zeros_like(muXAi) for muXAi in muXAList]
                alphaBList=[np.zeros_like(muXBi) for muXBi in muXBList]

        # set up new empty beta lists:
        betaADataList=[None for i in range(len(muXAList))]
        betaAIndexList=[None for i in range(len(muXAList))]
        betaBDataList=[None for i in range(len(muXBList))]
        betaBIndexList=[None for i in range(len(muXBList))]



        timeRefine2=time.time()
        evaluationData["time_refine"]+=timeRefine2-timeRefine1
        ################################################################################################################################

        
        if params["aux_printLayerConsistency"]:
            muYAtomicSums=np.array([np.sum(a) for a in muYAtomicDataList])
            print("layer partition consistency: ",np.sum(np.abs(muXAtomicSums-atomicCellMasses)))
        
        ## run algorithm at layer
        
        
        for nEps,(eps,nIterationsMax) in enumerate(params["eps_list"][nLayer]):
            print("eps: {:f}".format(eps))
            for nIterations in range(nIterationsMax):                                        

                ################################
                # iteration A
                time1=time.time()

                if params["parallel_iteration"]:
                    DomDecParallelMPI.ParallelIterate(comm,muYL,posYL,eps,\
                            partitionDataACompCells,partitionDataACompCellIndices,\
                            muYAtomicDataList,muYAtomicIndicesList,\
                            muXAList,posXAList,alphaAList,betaADataList,betaAIndexList,\
                            SinkhornSubSolver=params["sinkhorn_subsolver"], SinkhornError=params["sinkhorn_error"], SinkhornErrorRel=params["sinkhorn_error_rel"],\
                            MPIchunksize=params["MPI_chunksize"],MPIprobetime=params["MPI_probetime"])
                else:
                    DomDec.Iterate(muYL,posYL,eps,\
                            partitionDataACompCells,partitionDataACompCellIndices,\
                            muYAtomicDataList,muYAtomicIndicesList,\
                            muXAList,posXAList,alphaAList,betaADataList,betaAIndexList,\
                            SinkhornSubSolver=params["sinkhorn_subsolver"], SinkhornError=params["sinkhorn_error"], SinkhornErrorRel=params["sinkhorn_error_rel"])

                time2=time.time()
                evaluationData["time_iterate"]+=time2-time1
                ################################

                
                ################################
                # count total entries in muYAtomicList:
                nrEntries=0
                for a in muYAtomicDataList:
                    nrEntries+=a.shape[0]
                print("muYAtomicEntries: {:d}".format(nrEntries))
                evaluationData["sparsity_muYAtomicEntries"].append([nLayer,nEps,nIterations,0,nrEntries])
                ################################
                

                ################################
                # balancing A
                time1=time.time()
                if params["parallel_balancing"]:
                    DomDecParallelMPI.ParallelBalanceMeasures(comm,muYAtomicDataList,atomicCellMasses,partitionDataACompCells,\
                            MPIchunksize=params["MPI_chunksize"],MPIprobetime=params["MPI_probetime"])
                else:
                    DomDec.BalanceMeasuresMultiAll(muYAtomicDataList,atomicCellMasses,partitionDataACompCells,verbose=False)

                time2=time.time()
                evaluationData["time_measureBalancing"]+=time2-time1
                ################################

                ################################
                # truncation A
                time1=time.time()
                if params["parallel_truncation"]:
                    DomDecParallelMPI.ParallelTruncateMeasures(comm,muYAtomicDataList,muYAtomicIndicesList,1E-15,\
    	                    MPIchunksize=params["MPI_chunksize"],MPIprobetime=params["MPI_probetime"])
                else:
                    for i in range(len(atomicCells)):
                        muYAtomicDataList[i],muYAtomicIndicesList[i]=Common.truncateSparseVector(muYAtomicDataList[i],muYAtomicIndicesList[i],1E-15)
                time2=time.time()
                evaluationData["time_measureTruncation"]+=time2-time1
                ################################

                ################################
                # global time
                globalTime2=time.time()
                print("time:",globalTime2-globalTime1)
                evaluationData["timeList_global"].append([nLayer,nEps,nIterations,0,globalTime2-globalTime1])
                printTopic(evaluationData,"time")
                ################################



                ################################
                # count total entries in muYAtomicList:
                nrEntries=0
                for a in muYAtomicDataList:
                    nrEntries+=a.shape[0]
                print("muYAtomicEntries: {:d}".format(nrEntries))
                evaluationData["sparsity_muYAtomicEntries"].append([nLayer,nEps,nIterations,1,nrEntries])
                ################################


                ################################
                # iteration B
                time1=time.time()

                if params["parallel_iteration"]:
                    DomDecParallelMPI.ParallelIterate(comm,muYL,posYL,eps,\
                            partitionDataBCompCells,partitionDataBCompCellIndices,\
                            muYAtomicDataList,muYAtomicIndicesList,\
                            muXBList,posXBList,alphaBList,betaBDataList,betaBIndexList,\
                            SinkhornSubSolver=params["sinkhorn_subsolver"], SinkhornError=params["sinkhorn_error"], SinkhornErrorRel=params["sinkhorn_error_rel"],\
                            MPIchunksize=params["MPI_chunksize"],MPIprobetime=params["MPI_probetime"])
                else:
                    DomDec.Iterate(muYL,posYL,eps,\
                            partitionDataBCompCells,partitionDataBCompCellIndices,\
                            muYAtomicDataList,muYAtomicIndicesList,\
                            muXBList,posXBList,alphaBList,betaBDataList,betaBIndexList,\
                            SinkhornSubSolver=params["sinkhorn_subsolver"], SinkhornError=params["sinkhorn_error"], SinkhornErrorRel=params["sinkhorn_error_rel"])
                                    
                time2=time.time()
                evaluationData["time_iterate"]+=time2-time1
                ################################

                ################################
                # balancing B
                time1=time.time()

                if params["parallel_balancing"]:
                    DomDecParallelMPI.ParallelBalanceMeasures(comm,muYAtomicDataList,atomicCellMasses,partitionDataBCompCells,\
                            MPIchunksize=params["MPI_chunksize"],MPIprobetime=params["MPI_probetime"])
                else:
                    DomDec.BalanceMeasuresMultiAll(muYAtomicDataList,atomicCellMasses,partitionDataBCompCells,verbose=False)

                time2=time.time()
                evaluationData["time_measureBalancing"]+=time2-time1
                ################################


                ################################
                # truncation B
                time1=time.time()
                if params["parallel_truncation"]:
                    DomDecParallelMPI.ParallelTruncateMeasures(comm,muYAtomicDataList,muYAtomicIndicesList,1E-15,\
    	                    MPIchunksize=params["MPI_chunksize"],MPIprobetime=params["MPI_probetime"])
                else:
                    for i in range(len(atomicCells)):
                        muYAtomicDataList[i],muYAtomicIndicesList[i]=Common.truncateSparseVector(muYAtomicDataList[i],muYAtomicIndicesList[i],1E-15)
                time2=time.time()
                evaluationData["time_measureTruncation"]+=time2-time1
                ################################

                ################################
                # global time
                globalTime2=time.time()
                print("time:",globalTime2-globalTime1)
                evaluationData["timeList_global"].append([nLayer,nEps,nIterations,1,globalTime2-globalTime1])
                printTopic(evaluationData,"time")
                ################################

        # refine

        nLayer+=1


    #################################
    # dump finest
    if params["aux_dump_finest"]:
        print("dumping to file: aux_dump_finest...")
        with open(params["setup_dumpfile_finest"], 'wb') as f:
            pickle.dump([muYL,posYL,eps,\
                    partitionDataA,partitionDataB,\
                    muYAtomicDataList,muYAtomicIndicesList,\
                    muXAList,posXAList,alphaAList,\
                    muXBList,posXBList,alphaBList,\
                    betaBDataList,betaBIndexList,\
                    betaADataList,betaAIndexList],f,2)
        print("dumping done.")


    #####################################
    # evaluate primal and dual score
    if params["aux_evaluate_scores"]:
        solutionInfos,muYAList=DomDec.getPrimalInfos(muYL,posYL,posXAList,muXAList,alphaAList,betaADataList,betaAIndexList,eps,\
                getMuYList=True)

        alphaFieldEven,alphaGraph=DomDec.getAlphaFieldEven(alphaAList,alphaBList,\
                partitionDataA[0],partitionDataB[0],shapeXL,metaCellShape,params["domdec_cellsize"],\
                muX=muXL,requestAlphaGraph=True)

        betaFieldEven=DomDec.glueBetaList(\
                betaADataList,betaAIndexList,muXL.shape[0],offsets=alphaGraph.ravel(),muYList=muYAList,muY=muYL)
                
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
        datDual=DomDec.getHierarchicalKernel(MultiScaleSetupX,MultiScaleSetupY,\
                params["setup_dim"],params["hierarchy_depth"],eps,alphaFieldEven,betaFieldEven,nIter=0)
        print("entries in dual kernel: ",datDual[0].shape[0])
        
        piDual=scipy.sparse.csr_matrix(datDual,shape=(alphaFieldEven.shape[0],betaFieldEven.shape[0]))
        muYEff=np.array(piDual.sum(axis=0)).ravel()
        vRel=np.minimum(1,muYL/muYEff)
        
        # TODO: What is this?
        solutionInfos["scoreDual"]=np.sum(muX*alphaFieldEven)+np.sum(muYL*(betaFieldEven+eps*np.log(vRel)))-eps*np.sum(muYEff*vRel)


        solutionInfos["scoreGap"]=solutionInfos["scorePrimal"]-solutionInfos["scoreDual"]
        solutionInfos["scoreGapRel"]=solutionInfos["scoreGap"]/solutionInfos["scorePrimal"]

        print(solutionInfos)

        for k in solutionInfos.keys():
            evaluationData["solution_"+k]=solutionInfos[k]


    #####################################
    # dump evaluationData into json result file:
    # print(evaluationData)
    with open(params["setup_resultfile"],"w") as f:
        json.dump(evaluationData,f)
    print(evaluationData)

finally:
    # in case an exception is raised, one still needs to free the worker processes
    ParallelMap.Close(comm)

