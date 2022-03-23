import numpy as np
import scipy
import scipy.io as sciio
import os
import h5py

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


def getDumpFilename(basename,digits=4):
    """Return next unused dump-filename to store next batch of iterations."""
    cont=True
    i=0
    while cont:
        fn=(basename+"_{:0"+str(digits)+"d}").format(i)+".dat"
        if os.path.isfile(fn):
            i+=1
        else:
            cont=False
    # here, i is index of first file that does not exist
    # fn stores name of first non-existing file
    return fn

def getPrevDumpFilename(basename,digits=4):
    """Return last used dump-filename to load previous iteration. If none exists, return None."""
    cont=True
    i=0
    while cont:
        fn=(basename+"_{:0"+str(digits)+"d}").format(i)+".dat"
        if os.path.isfile(fn):
            i+=1
        else:
            cont=False
    # here, i is index of first file that does not exist
    if i==0:
        return None
    else:
        fn=(basename+"_{:0"+str(digits)+"d}").format(i-1)+".dat"
        return fn


def getDumpFilenameList(basename,digits=4):
    """Return list of all existing dump filenames."""
    cont=True
    i=0
    result=[]
    while cont:
        fn=(basename+"_{:0"+str(digits)+"d}").format(i)+".dat"
        if os.path.isfile(fn):
            i+=1
            result.append(fn)
        else:
            cont=False
    return result

