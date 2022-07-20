def benchmark2D(f,config,testID,error = 0.0001):

  timeStamp1 = time.time()

  with open(config) as f2:
    data = json.load(f)

  n = data.N
  cellsize = data.domdec_cellsize
  batchsize = data.keops_batchsize
  innerIter = data.sinkhorn_inner_iter
  maxIter = data.sinkhorn_max_iter
  
  timeStamp1 = time.time()

  dim = 2
  eps = 2*(1024/n)**2

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
  x_pad = np.arange(-cellsize*dx, (n+cellsize)*dx, dx)
  X0_pad = np.repeat(x_pad.reshape(-1,1), (n+2*cellsize), axis = 1).reshape(-1,1)
  X1_pad = np.repeat(x_pad.reshape(1, -1),(n+2*cellsize), axis = 0).reshape(-1,1)
  posX = np.concatenate((X0_pad, X1_pad), axis = 1)

  #get the padded plan
  linear_indices = np.arange((n+2*cellsize)**2)
  cartesian0 = linear_indices % (n+2*cellsize)-cellsize 
  cartesian1 = linear_indices // (n+2*cellsize)-cellsize 
  original_indices = linear_indices[(0 <= cartesian0) & (cartesian0 < n) & (0 <= cartesian1) & (cartesian1 < n)]

  padded_plan = sp.csr_matrix(((n+2*cellsize)**2, n**2))
  padded_plan[original_indices, :] = pi_sp

  alphapad = np.full(((n+2*cellsize), (n+2*cellsize)),1e-40)
  alphapad[cellsize:n+cellsize, cellsize:n+cellsize] = alpha.reshape(n, n)
  alpha = alphapad.ravel()

  muX = np.array(np.sum(pi_sp,axis = 1)).ravel()
  muY = np.array(np.sum(pi_sp,axis = 0)).ravel()

  # pad MuX
  muXpad = np.full(((n+2*cellsize), (n+2*cellsize)),1e-40)
  muXpad[cellsize:n+cellsize, cellsize:n+cellsize] = muX.reshape(n, n)
  muX = muXpad.ravel()

  shapeY = (n,n)
  shapeX = (n+2*cellsize, n+2*cellsize)
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

  muYAtomicDataList = []
  muYAtomicIndicesList = []
  for i in range(len(atomicCells)):
    muYAtomicData = np.zeros(padded_plan.shape[1])
    UnorderedIndices = []
    for j in atomicCells[i]:
        nuj = padded_plan[j]
        muYAtomicData[nuj.indices] += nuj.data
        UnorderedIndices = UnorderedIndices + nuj.indices.tolist()
    muYAtomicIndices = sorted(list(set(UnorderedIndices)))
    muYAtomicDataList.append(np.array(muYAtomicData[muYAtomicIndices]))
    muYAtomicIndicesList.append(np.array(muYAtomicIndices))

  

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
                    "SolveOnCellKeopsGrid", error,\
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
                    "SolveOnCellKeopsGrid", error,\
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
                    "SolveOnCellKeopsGrid", error,\
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
                    "SolveOnCellKeopsGrid", error,\
                    False,\
                    SinkhornMaxIter = maxIter, BatchSize = batchsize
                    )
  
  visualize_deformation_map(muYAtomicIndicesList, muYAtomicDataList, partitionDataBCompCells, muY, n)

  # balancing
  DomDec.BalanceMeasuresMultiAll(muYAtomicDataList,atomicCellMasses,partitionDataBCompCells,verbose=True)
  # truncation
  for i in range(len(atomicCells)):
    muYAtomicDataList[i],muYAtomicIndicesList[i]=Common.truncateSparseVector(muYAtomicDataList[i],muYAtomicIndicesList[i],1E-15)

  timeStamp6 = time.time()

  largedict = {
      "muYAtomicDataList": muYAtomicDataList,
      "muYAtomicIndicesList": muYAtomicIndicesList,
      "alphaBList": alphaBList,
      "betaBDataList": betaBDataList,
      "betaBIndexList": betaBIndexList,
  }

  smalldict = {
      "start": timeStamp1,
      "setupEnd": timeStamp2,
      "AIteration1": timeStamp3,
      "BIteration1": timeStamp4,
      "AIteration2": timeStamp5,
      "BIteration2": timeStamp6,
      "primalScore": getPrimalInfos(muY,posY,posXAList,muXAList,alphaAList,betaADataList,betaAIndexList,eps,getMuYList=False)
  }

  #where to dump?

  with open("experiments/"+"_dump_"+".dat""ABAB_{:d}_A".format(n), 'wb') as f:
    pickle.dump([largedict,smalldict],f)

  return
