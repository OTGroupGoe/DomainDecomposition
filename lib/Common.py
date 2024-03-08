import numpy as np
import scipy
import scipy.io as sciio
import pickle

def importMeasure(fn,subslice=None):
    if fn.split(".")[-1] == "pickle":
         return importMeasure_pickle(fn)
    else:
        dat=np.array(sciio.loadmat(fn)["a"],dtype=np.double,order="C")
        if subslice is not None:
            dat=dat[subslice]
        mu,pos=processDensity_Grid(dat,totalMass=1.)
        return (mu,pos,dat.shape)

def importMeasure_pickle(fn):
    # New load function using pickle
    with open(fn, "rb") as f:
        d = pickle.load(f)
    mu = d["mu"]
    pos = d["pos"]
    shape = d["shape"]
    return mu, pos, shape

def getPoslistNCube(shape,dtype=np.double):
	"""Create list of positions in an n-dimensional cube of size shape."""
	ndim=len(shape)

	axGrids=[np.arange(i,dtype=dtype) for i in shape]
	prePos=np.array(np.meshgrid(*axGrids,indexing='ij'),dtype=dtype )
	# the first dimension of prepos is the dimension of the  posvector, the successive dimensions are in the cube
	# so need to move first axis to end, and then flatten
	pos=np.rollaxis(prePos,0,ndim+1)
	# flattening
	newshape=(-1,ndim)
	return (pos.reshape(newshape)).copy()

def processDensity_Grid(x,totalMass=None,constOffset=None,keepZero=True,zeroThresh=1E-14):

	# process actual density
	
	# copy, cast to double and reshape
	img=x.astype(dtype=np.double).copy()
	shape=img.shape
	nPoints=np.prod(shape)
	dim=len(shape)
	img=img.reshape((nPoints))
	
	processDensity(img,totalMass=totalMass,constOffset=constOffset)

	# get grid pos information
	posList=getPoslistNCube(shape,dtype=np.int64)
	posList=posList.reshape((nPoints,dim))

	# if desired, throw away points with zero mass
	if not keepZero:
		nonZeroPos=np.nonzero(img>zeroThresh)
		img=img[nonZeroPos]
		posList=posList[nonZeroPos]
		
		# if necessary, rescale mass once more
		processDensity(img, totalMass=totalMass, constOffset=None)

	return (img,posList)

def processDensity(x, totalMass=None, constOffset=None):
	# re-normalize and add offset if required
	if totalMass is not None:
		x[:]=totalMass*x/np.sum(x)
		if constOffset is not None:
			x[:]=x+constOffset
			x[:]=totalMass*x/np.sum(x)
	else:
		if constOffset is not None:
			x[:]=x+constOffset
			#x[:]=x/np.sum(x)

def getEuclideanCostFunction(x1,x2,p=2.):
	c=-2*np.einsum(x1,[0,2],x2,[1,2],[0,1])
	x1Sqr=np.sum(x1*x1,axis=1)
	x2Sqr=np.sum(x2*x2,axis=1)
	for i in range(x1.shape[0]):
		c[i,:]+=x1Sqr[i]+x2Sqr
	return np.power(c,p/2.)


def GetCostEntriesFromKernel(kernel,posX,posY):
    dim=posY.shape[1]
    xres=kernel.shape[0]
    nonZeros=kernel.data.shape[0]
    
    result=np.zeros((nonZeros,),dtype=np.double)
    for x in range(xres):
        activeSlice=slice(kernel.indptr[x],kernel.indptr[x+1])
        activeIndicesY=kernel.indices[activeSlice]

        result[activeSlice]=np.sum((posX[x]-posY[activeIndicesY])**2,axis=1)
    return result

def GetRelativeEntropyFromKernel(kernel,muX,muY):
    xres=kernel.shape[0]
    nonZeros=kernel.data.shape[0]
    
    result=np.zeros((nonZeros,),dtype=np.double)
    for x in range(xres):
        activeSlice=slice(kernel.indptr[x],kernel.indptr[x+1])
        activeIndicesY=kernel.indices[activeSlice]

        #result[activeSlice]=np.sum((posX[x]-posY[activeIndicesY])**2,axis=1)
        result[activeSlice]=muX[x]*muY[activeIndicesY]
    return np.sum(kernel.data*np.log(kernel.data/result))

def GetRegularizedCost(pi,muX,muY,posX,posY,eps):
    costVector=GetCostEntriesFromKernel(pi,posX,posY)
    
    return np.sum(costVector*pi.data)+eps*GetRelativeEntropyFromKernel(pi,muX,muY)

def GetImage(dat):
    return (cm.viridis(dat/np.max(dat))[...,:3]*255).astype(np.uint8)
    

def GetPartialYMarginal(piCell,XCellInCell):
    return np.array(piCell[XCellInCell].sum(axis=0)).ravel()


def GlueKernels(pi12,pi23,X1in12):
    return scipy.sparse.vstack((pi12[X1in12],pi23))


def getSparseVector(vec,thresh,subIndices=None,subShape=None):
    ind=np.where(vec>=thresh)[0]
    data=vec[ind]
    ptr=np.array([0,len(ind)],dtype=np.int64)
    if subIndices is not None:
        ind=subIndices[ind]
        shp=subShape
    else:
        shp=vec.shape[0]
    
    return scipy.sparse.csr_matrix((data,ind,ptr),(1,shp))

def truncateSparseVector(data,indices,thresh):
    """vec is a sparse.csr_matrix with one row. Returns a sparse.csr_matrix with one row
    where only entries >=thresh will be kept."""
    
    keep=np.where(data>=thresh)[0]
    return (data[keep],indices[keep])


def getEpsListDefault(hierarchy_depth,hierarchy_top,\
        epsBase,layerFactor,layerSteps,stepsFinal,
        nIterations=1,nIterationsLayerInit=2,nIterationsGlobalInit=4,nIterationsFinal=1):
    result=[]
    for i in range(hierarchy_top):
        result.append([])
    
    stepFactor=layerFactor**(1/layerSteps)
    
    for i in range(hierarchy_top,hierarchy_depth+1):
        iReverse=hierarchy_depth-i
        resultLayer=[]
        for j in range(layerSteps+1):
            
            # determine number of iterations for given eps
            if j==0:
                if i==hierarchy_top:
                    # global initialization
                    nIt=nIterationsGlobalInit
                else:
                    # initialization of new layer
                    nIt=nIterationsLayerInit
            else:
                # standard case
                nIt=nIterations
            resultLayer.append([epsBase*layerFactor**iReverse*stepFactor**(layerSteps-j),nIt])
            
        # additional steps on finest layer
        if i==hierarchy_depth:
            for j in range(stepsFinal-1):
                resultLayer.append([epsBase*stepFactor**(-j-1),nIterations])
            for j in [stepsFinal-1]:
                resultLayer.append([epsBase*stepFactor**(-j-1),nIterationsFinal])
        result.append(resultLayer)
    
    return result


def getDiscreteGradient(nx,ny):
    """Return discrete gradient matrices for a grid graph with nx by ny nodes (rows x columns).
    Returns two sparse matrices, gradH and gradV. First is gradient along rows, second along columns."""
    indexArray=np.arange(nx*ny).reshape((nx,ny))

    # vertical
    indptr=np.arange((nx-1)*ny+1)*2
    data=np.full((nx-1)*ny*2,1.,dtype=np.double)
    data[::2]=-1
    indices=np.zeros((nx-1)*ny*2,dtype=np.int32)
    indices[1::2]=(indexArray[1:,:]).ravel()
    indices[::2]=indices[1::2]-ny
    
    gradV=scipy.sparse.csr_matrix((data,indices,indptr),shape=((nx-1)*ny,nx*ny))
    
    # horizontal
    indptr=np.arange((nx)*(ny-1)+1)*2
    data=np.full((nx)*(ny-1)*2,1.,dtype=np.double)
    data[::2]=-1
    indices=np.zeros((nx)*(ny-1)*2,dtype=np.int32)
    indices[1::2]=(indexArray[:,1:]).ravel()
    indices[::2]=indices[1::2]-1
    
    gradH=scipy.sparse.csr_matrix((data,indices,indptr),shape=((nx)*(ny-1),nx*ny))
    
    #return scipy.sparse.vstack((gradH,gradV))
    return (gradH,gradV)
    
    
    
