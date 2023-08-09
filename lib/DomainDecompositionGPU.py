import numpy as np
import torch
# from . import MultiScaleOT
from .LogSinkhorn import LogSinkhorn as LogSinkhorn
import LogSinkhornGPU
from . import DomainDecomposition as DomDec

#########################################################
# Bounding box utils
# Cast all cell problems to a common size and coordinates
#########################################################


def pad_array(a, padding, pad_value=0):
    """
    Pad array `a.reshape(*shape)` with a margin of `padding` width, filled with `pad_value`. 
    If the original shape of `a` was not `shape`, return raveled padded array.
    """
    shape = a.shape
    new_shape = tuple(s + 2*padding for s in shape)
    original_locations = tuple(slice(padding, s+padding) for s in shape)
    b = np.full(new_shape, pad_value, dtype=a.dtype)
    b[original_locations] = a
    return b


def reformat_indices_2D(index, shapeY):
    """
    Takes linear indices and tuple of the total shapeY, and returns parameters 
    encoding the bounding box.
    """
    # TODO: generalize to 3D
    _, n = shapeY
    idx, idy = index // n, index % n

    left = np.min(idx)
    right = np.max(idx)
    bottom = np.min(idy)
    top = np.max(idy)
    width = right - left + 1  # box width
    height = top - bottom + 1  # box height
    # Turn idx and idy into relative indices
    idx = idx - left
    idy = idy - bottom

    return left, bottom, width, height, idx, idy


def batch_cell_marginals_2D(marg_indices, marg_data, shapeY, muY):
    """
    Reshape marginals to a rectangle, find the biggest of all of these and 
    concatenate all marginals along a batch dimension. 
    Copy reference measure `muY` to these bounding boxes.
    """
    # TODO: generalize to 3D
    # Initialize structs to hold the data of all marginals
    B = len(marg_indices)
    left = np.zeros(B, dtype=np.int32)
    bottom = np.zeros(B, dtype=np.int32)
    width = np.zeros(B, dtype=np.int32)
    height = np.zeros(B, dtype=np.int32)
    idx = []
    idy = []

    # Get bounding boxes for all marginals
    for (i, indices) in enumerate(marg_indices):
        left[i], bottom[i], width[i], height[i], idx_i, idy_i = \
            reformat_indices_2D(indices, shapeY)
        idx.append(idx_i)
        idy.append(idy_i)

    # Compute common bounding box
    max_width = np.max(width)
    max_height = np.max(height)

    # Init batched marginal
    Nu_box = np.zeros((B, max_width, max_height))
    # Init batched nuref
    muY = muY.reshape(shapeY)
    Nuref_box = np.zeros((B, max_width, max_height))

    # Fill NuJ
    for (i, data) in enumerate(marg_data):
        Nu_box[i, idx[i], idy[i]] = data
        # Copy reference measure
        i0, i1 = left[i], left[i] + width[i]
        j0, j1 = bottom[i], bottom[i] + height[i]
        Nuref_box[i, :, :] = muY[i0:i1, j0:j1]

    # Return NuJ together with bounding box data (they will be needed for unpacking)
    return Nu_box, Nuref_box, left, bottom, max_width, max_height


def unpack_cell_marginals_2D(Nu_basic, left, bottom, shapeY, threshold=1e-15):
    """
    Un-batch all the cell marginals and truncate entries below `threshold`.
    """
    # TODO: generalize to 3D
    # TODO: incorporate balancing
    # Nu_basic is of size (B, 4, w, h), because there are 4 basic cells per
    # composite cell
    _, n = shapeY
    # Do all the process in the cpu
    B = Nu_basic.shape[0]
    sizeJ = Nu_basic.shape[1]
    marg_indices = []
    marg_data = []
    # NOTE: extraction could be done for the whole Nu_basic at once,
    # but then how to distinguish between slices.
    for k in range(B):
        for i in range(sizeJ):  # TODO: allow different dimension
            # idx, idy = np.nonzero(Nu_basic[k, i])
            idx, idy = np.nonzero(Nu_basic[k, i] > threshold)
            linear_id = (idx+left[k])*n + idy + bottom[k]
            marg_indices.append(linear_id)
            marg_data.append(Nu_basic[k, i, idx, idy])
    return marg_indices, marg_data


def unpack_cell_marginals_2D_gpu(Nu_basic, left, bottom, shapeY, threshold=1e-15):
    """
    Un-batch all the cell marginals and truncate entries below `threshold`.
    """
    # TODO: generalize to 3D
    # Assumes Nu_basic still in GPU
    # Nu_basic is of size (B, 4, w, h), because there are 4 basic cells per
    # composite cell
    _, n = shapeY
    # Do all the process in the cpu
    B = Nu_basic.shape[0]
    sizeJ = Nu_basic.shape[1]
    marg_indices = []
    marg_data = []
    idB, idx, idy = torch.where(Nu_basic > threshold)
    nu_entries = Nu_basic[idB, idx, idy]
    # Where batch index changes slice
    steps = torch.where(torch.diff(idB))[0]+1
    # Add start and end
    steps = np.hstack(([0], steps.cpu().numpy(), [len(idB)]))
    # TODO: continue here

    # -------
    # NOTE: extraction could be done for the whole Nu_basic at once,
    # but then how to distinguish between slices.
    for k in range(B):
        for i in range(sizeJ):  # TODO: allow different dimension
            # idx, idy = np.nonzero(Nu_basic[k, i])
            idx, idy = np.nonzero(Nu_basic[k, i] > threshold)
            linear_id = (idx+left[k])*n + idy + bottom[k]
            print(left.dtype, bottom.dtype, linear_id.dtype)
            marg_indices.append(linear_id)
            marg_data.append(Nu_basic[k, i, idx, idy])
    return marg_indices, marg_data


def batch_balance(muXCell, Nu_basic):
    B, M, _ = muXCell.shape
    s = M//2
    atomic_mass = muXCell.view(B, 2, s, 2, s).permute(
        (0, 1, 3, 2, 4)).contiguous().sum(dim=(3, 4))
    atomic_mass = atomic_mass.view(B, -1).cpu().numpy()
    for k in range(B):
        status, Nu_basic[k] = DomDec.BalanceMeasuresMulti(
            Nu_basic[k], atomic_mass[k], 1e-12, 1e-7
        )
    return Nu_basic


def get_grid_cartesian_coordinates(left, bottom, w, h, dx):
    """
    Generate the cartesian coordinates of the boxes with bottom-left corner 
    given by (left, bottom) (vectors), with width w, height h and spacing dx.
    i0 : torch.tensor
    j0 : torch.tensor
    w : int
    h : int
    dx : float
    """
    # TODO: generalize to 3D
    B = len(left)
    device = left.device
    x1_template = torch.arange(0, w*dx, dx, device=device)
    x2_template = torch.arange(0, h*dx, dx, device=device)
    x1 = left.view(-1, 1)*dx + x1_template.view(1, -1)
    x2 = bottom.view(-1, 1)*dx + x2_template.view(1, -1)
    return x1, x2


def batch_shaped_cartesian_prod(xs):
    """
    For xs = (x1, ..., xd) a tuple of tensors of shape (B, M1), ... (B, Md), 
    form the tensor X of shape (B, M1, ..., Md, d) such that
    `X[i] = torch.cartesian_prod(xs[i],...,xs[i]).view(M1, ..., Md, d)`
    """
    B = xs[0].shape[0]
    for x in xs:
        assert B == x.shape[0], "All xs must have the same batch dimension"
        assert len(x.shape) == 2, "xi must have shape (B, Mi)"
    Ms = tuple(x.shape[1] for x in xs)
    dim = len(xs)
    device = xs[0].device

    shapeX = (B, ) + Ms + (dim,)
    X = torch.empty(shapeX, device=device)
    for i in range(dim):
        shapexi = (B,) + (1,)*i + (Ms[i],) + (1,)*(dim-i-1)
        X[..., i] = xs[i].view(shapexi)
    return X


def compute_offsets_sinkhorn_grid(xs, ys, eps):
    """
    Compute offsets
    xs and ys are d-tuples of tensors with shape (B, Mi) where B is the batch 
    dimension and Mi the size of the grid in that coordinate
    # TODO: ref
    """
    # Get cartesian prod
    X = batch_shaped_cartesian_prod(xs)
    Y = batch_shaped_cartesian_prod(ys)
    shapeX = X.shape
    B, Ms, dim = shapeX[0], shapeX[1:-1], shapeX[-1]
    Ns = Y.shape[1:-1]

    # Get "bottom left" corner coordinates: select slice (:, 0, ..., 0, :)
    X0 = X[(slice(None),) + (0,)*dim + (slice(None),)] \
        .view((B,) + (1,)*dim + (dim,))  # NOTE alternatively: use unpack op.
    Y0 = Y[(slice(None),) + (0,)*dim + (slice(None),)] \
        .view((B,) + (1,)*dim + (dim,))  # NOTE alternatively: use unpack op.

    # Use the formulas in [TODO: ref] to compute the offset
    offsetX = torch.sum(2*(X-X0)*(Y0-X0), dim=-1)/eps
    offsetY = torch.sum(2*(Y-Y0)*(X0-Y0), dim=-1)/eps
    offset_constant = -torch.sum((X0-Y0)**2, dim=-1)/eps
    return offsetX, offsetY, offset_constant

##############################################################
# Dedicated CUDA solver for DomDec:
# Assumes B rectangular problems with same size
##############################################################


class LogSinkhornCudaImageOffset(LogSinkhornGPU.AbstractSinkhorn):
    """
    Online Sinkhorn solver for standard OT on images with separable cost, 
    custom CUDA implementation. 
    Each Sinkhorn iteration has complexity N^(3/2), instead of the usual N^2. 

    Attributes
    ----------
    mu : torch.Tensor 
        of size (B, M1, M2)
        First marginals
    nu : torch.Tensor 
        of size (B, N1, N2)
        Second marginals 
    C : tuple 
        of the form ((x1, x2), (y1, y2))
        Grid coordinates
    eps : float
        Regularization strength
    muref : torch.Tensor 
        with same dimensions as mu (except axis 0, which can have len = 1)
        First reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    nuref : torch.Tensor 
        with same dimensions as nu (except axis 0, which can have len = 1)
        Second reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    alpha_init : torch.Tensor 
        with same dimensions as mu, or None
        Initialization for the first Sinkhorn potential
    """

    def __init__(self, mu, nu, C, eps, **kwargs):
        (xs, ys) = C
        zs = xs + ys  # Have all coordinates in one tuple
        x1 = zs[0]
        B = LogSinkhornGPU.batch_dim(mu)
        # Check whether xs have a batch dimension
        if len(x1.shape) == 1:
            for z in zs:
                assert len(z) == 1, \
                    "dimensions of grid coordinates must be consistent"
            C = tuple(tuple(xi.view(1, -1) for xi in X) for X in C)
        else:
            for z in zs:
                assert len(z.shape) == 2, \
                    "coordinates can just have one spatial dimension"
                assert z.shape[0] == B, \
                    "batch dimension of all coordinates must coincide"

        # Now all coordinates have a batch dimension of either B or 1.
        # Check that all coordinates have same grid spacing
        dx = x1[0, 1]-x1[0, 0]
        for z in zs:
            assert torch.max(torch.abs(torch.diff(z, dim=-1)-dx)) < 1e-6, \
                "Grid is not equispaced"

        # Check geometric dimensions
        Ms = LogSinkhornGPU.geom_dims(mu)
        Ns = LogSinkhornGPU.geom_dims(nu)
        assert len(Ms) == len(Ns) == 2, "Shapes incompatible with images"

        # Compute the offsets
        self.offsetX, self.offsetY, self.offset_const = \
            compute_offsets_sinkhorn_grid(xs, ys, eps)

        # Save xs and ys in case they are needed later
        self.xs = xs
        self.ys = ys

        C = (dx, Ms, Ns)

        super().__init__(mu, nu, C, eps, **kwargs)

    def get_new_alpha(self):
        dx, Ms, Ns = self.C
        h = self.beta / self.eps + self.lognuref + self.offsetY
        return - self.eps * (
            LogSinkhornGPU.softmin_cuda_image(h, Ms, Ns, self.eps, dx)
            + self.offsetX + self.offset_const + self.logmuref - self.logmu
        )

    def get_new_beta(self):
        dx, Ms, Ns = self.C
        h = self.alpha / self.eps + self.logmuref + self.offsetX
        return - self.eps * (
            LogSinkhornGPU.softmin_cuda_image(h, Ns, Ms, self.eps, dx)
            + self.offsetY + self.offset_const + self.lognuref - self.lognu
        )

    def get_dense_cost(self, ind=None):
        """
        Get dense cost matrix of given problems. If no argument is given, all 
        costs are computed. Can be memory intensive, so it is recommended to do 
        small batches at a time.
        `ind` must be slice or iterable, not int.
        """

        if ind == None:
            ind = slice(None,)

        xs = tuple(x[ind] for x in self.xs)
        ys = tuple(y[ind] for y in self.ys)
        X = batch_shaped_cartesian_prod(xs)
        Y = batch_shaped_cartesian_prod(ys)
        B = X.shape[0]
        dim = X.shape[-1]
        C = ((X.view(B, -1, 1, dim) - Y.view(B, 1, -1, dim))**2).sum(dim=-1)
        return C, X, Y

    def get_dense_plan(self, ind=None, C=None):
        """
        Get dense plans of given problems. If no argument is given, all plans 
        are computed. Can be memory intensive, so it is recommended to do small 
        batches at a time.
        `ind` must be slice or iterable, not int.
        """
        if ind == None:
            ind = slice(None,)

        if C == None:
            C, _, _ = self.get_dense_cost(ind)

        B = C.shape[0]
        alpha, beta = self.alpha[ind], self.beta[ind]
        muref, nuref = self.muref[ind], self.nuref[ind]

        pi = torch.exp(
            (alpha.view(B, -1, 1) + beta.view(B, 1, -1) - C) / self.eps
        ) * muref.view(B, -1, 1) * nuref.view(B, 1, -1)
        return pi

    def change_eps(self, new_eps):
        """
        Change the regularization strength `self.eps`.
        In this solver this also involves renormalizing the offsets.
        """
        self.Niter = 0
        self.current_error = self.max_error + 1.
        scale = self.eps / new_eps
        self.offsetX = self.offsetX * scale
        self.offsetY = self.offsetY * scale
        self.offset_const = self.offset_const * scale
        self.eps = new_eps

    def get_dx(self):
        return self.C[0]


def convert_to_basic_2D(A, basic_grid_shape, cellsize):
    """
    A is a tensor of shape (B, m1, m2, *(rem_shape)), where
    B = prod(basic_grid_shape)/4
    m1 = m2 = 2*cellsize
    """
    B, m1, m2 = A.shape[:3]
    rem_shape = A.shape[3:]  # Rest of the shape, that we will not change
    sb1, sb2 = basic_grid_shape
    assert 4*B == sb1*sb2, "Batchsize does not match basic grid"
    assert m1 == m2 == 2*cellsize, "Problem size does not mach cellsize"
    new_shape = (sb1, sb2, cellsize, cellsize, *rem_shape)
    # Permute dimensions to make last X dimension that inner to cell
    A_res = A.view(sb1//2, sb2//2, 2, cellsize, 2,
                   cellsize, -1).permute(0, 2, 1, 4, 3, 5, 6)
    A_res = A_res.reshape(new_shape)
    return A_res


def convert_to_batch_2D(A):
    """
    A is a tensor of shape (sb1, sb2, cellsize, cellsize, *(rem_shape))
    """
    sb1, sb2, cellsize = A.shape[:3]
    rem_shape = A.shape[4:]
    new_shape = (sb1*sb2//4, 2*cellsize, 2*cellsize, *rem_shape)
    A_res = A.view(sb1//2, 2, sb2//2, 2, cellsize, cellsize, -1)
    A_res = A_res.permute(0, 2, 1, 4, 3, 5, 6).reshape(new_shape)
    return A_res


def get_cell_marginals(muref, nuref, alpha, beta, xs, ys, eps):
    """
    Get cell marginals directly using duals and logsumexp reductions, 
    without building the transport plans. 
    The mathematical formulation is covered in [TODO: ref]
    Returns tensor of size (B, n_basic, Ns), where B is the batch dimension and 
    n_basic the number of basic cells per composite cell.
    """
    Ms = LogSinkhornGPU.geom_dims(muref)
    Ns = LogSinkhornGPU.geom_dims(nuref)

    # Deduce cellsize
    s = Ms[0]//2
    dim = len(xs)
    assert dim == 2, "Not implemented for dimension rather than 2"
    n_basic = 2**dim  # number of basic cells per composite cell, depends on dim
    dx = (xs[0][0, 1] - xs[0][0, 0]).item()  # TODO: check consistency

    # Perform permutations and reshapes in X data to turn them
    # into B*n_cells problems of size (s,s)
    alpha_b = alpha.view(-1, 2, s, 2, s) \
        .permute((0, 1, 3, 2, 4)).reshape(-1, s, s)
    mu_b = muref.view(-1, 2, s, 2, s) \
        .permute((0, 1, 3, 2, 4)).reshape(-1, s, s)
    new_Ms = (s, s)
    logmu_b = LogSinkhornGPU.log_dens(mu_b)

    x1, x2 = xs
    x1_b = torch.repeat_interleave(x1.view(-1, 2, s, 1), 2, dim=3)
    x1_b = x1_b.permute((0, 1, 3, 2)).reshape(-1, s)
    x2_b = torch.repeat_interleave(x2.view(-1, 1, 2, s), 2, dim=1)
    x2_b = x2_b.reshape(-1, s)
    xs_b = (x1_b, x2_b)

    # Duplicate Y data to match X data
    y1, y2 = ys
    y1_b = torch.repeat_interleave(y1, n_basic, dim=0)
    y2_b = torch.repeat_interleave(y2, n_basic, dim=0)
    xs_b = (x1_b, x2_b)
    ys_b = (y1_b, y2_b)

    # Perform a reduction to get a second dual for each basic cell
    offsetX, offsetY, offset_const = compute_offsets_sinkhorn_grid(
        xs_b, ys_b, eps)
    h = alpha_b / eps + logmu_b + offsetX
    beta_hat = - eps * (
        LogSinkhornGPU.softmin_cuda_image(h, Ns, new_Ms, eps, dx)
        + offsetY + offset_const
    )

    # Build cell marginals
    nu_basic = nuref[:, None] * torch.exp(
        (beta[:, None] - beta_hat.view(-1, n_basic, *Ns))/eps
    )
    return nu_basic


def BatchIterate(
    muY, posY, dx, eps,
    # TODO: remove this unnecesary input
    partitionDataCompCells, partitionDataCompCellIndices,
    muYAtomicDataList, muYAtomicIndicesList,
    muXJ, posXJ, alphaJ, betaDataList, betaIndexList, shapeY,
    SinkhornError=1E-4, SinkhornErrorRel=False, SinkhornMaxIter=None,
    SinkhornInnerIter=100, BatchSize=np.inf
):

    BatchTotal = muXJ.shape[0]
    if BatchSize == np.inf:
        BatchSize = BatchTotal

    # Divide problem data into batches of size BatchSize
    for i in range(0, BatchTotal, BatchSize):
        # Prepare batch of problem data
        muXBatch = muXJ[i:i+BatchSize]
        # posXJ is list of coordinates along dimensions
        posXBatch = tuple(x[i:i+BatchSize] for x in posXJ)
        alphaBatch = alphaJ[i:i+BatchSize]
        muYData = muYAtomicDataList
        # partitionDataCompCellIndicesBatch = partitionDataCompCellIndices[i:i+BatchSize]
        partitionDataCompCellsBatch = partitionDataCompCells[i:i+BatchSize]
        current_batch = len(partitionDataCompCellsBatch)

        # Solve batch
        resultAlpha, resultBeta, resultMuYAtomicDataList, \
            resultMuYCellIndicesList, solver = BatchDomDecIteration_CUDA(
                SinkhornError, SinkhornErrorRel, muY, posY, dx, eps, shapeY,
                muXBatch, posXBatch, alphaBatch,
                [(muYAtomicDataList[j] for j in J)
                 for J in partitionDataCompCellsBatch],
                [(muYAtomicIndicesList[j] for j in J)
                 for J in partitionDataCompCellsBatch],
                # partitionDataCompCellIndicesBatch[i],
                SinkhornMaxIter, SinkhornInnerIter, current_batch
            )

        # Extract results
        alphaJ[i:i+BatchSize] = resultAlpha
        # TODO: how to extract beta
        # betaDataList[i*BatchSize+k]=resultBeta[k]
        # betaIndexList[i*BatchSize+k]=muYCellIndices.copy()[k]
        # Extract basic cell marginals
        # All composite cells have same size
        lenJ = len(partitionDataCompCellsBatch[0])
        assert lenJ * current_batch == len(resultMuYAtomicDataList)
        # Iterate over composite cells...
        for (k, J) in enumerate(partitionDataCompCellsBatch):
            # ...and then over basic cells inside each composite cell
            for (j, b) in enumerate(J):
                muYAtomicDataList[b] = resultMuYAtomicDataList[lenJ*k + j]
                muYAtomicIndicesList[b] = resultMuYCellIndicesList[lenJ*k + j]

    return alphaJ, muYAtomicIndicesList, muYAtomicDataList, solver
    # for jsub,j in enumerate(partitionDataCompCellsBatch):
    #     muYAtomicDataList[j]=resultMuYAtomicDataList[jsub]
    #     muYAtomicIndicesList[j]=muYCellIndices.copy()


def BatchDomDecIteration_CUDA(
        SinkhornError, SinkhornErrorRel, muY, posYCell, dx, eps, shapeY,
        # partitionDataCompCellIndices,
        muXCell, posXCell, alphaCell, muYAtomicListData, muYAtomicListIndices,
        SinkhornMaxIter, SinkhornInnerIter, BatchSize, balance=True):

    # 1: compute composite cell marginals
    muYCellData = []
    muYCellIndices = []
    for i in range(BatchSize):
        arrayAdder = LogSinkhorn.TSparseArrayAdder()
        for x, y in zip(muYAtomicListData[i], muYAtomicListIndices[i]):
            arrayAdder.add(x, y)
        muYCellData.append(arrayAdder.getDataTuple()[0])
        muYCellIndices.append(arrayAdder.getDataTuple()[1])

    # 2: compute bounding box and copy data to it. Also get reference measure
    # in boxes
    # TODO: for Sang: muYCell are not relevant for unbalanced domdec, what one
    # needs here are the nu_minus. subMuY is the reference measure
    muYCell, subMuY, left, bottom, width, height = batch_cell_marginals_2D(
        muYCellIndices, muYCellData, shapeY
    )
    # Turn muYCell, left and bottom into tensor
    device = muXCell.device
    muYCell = torch.tensor(muYCell, device=device, dtype=torch.float32)
    left_cuda = torch.tensor(left, device=device)
    bottom_cuda = torch.tensor(bottom, device=device)

    # 3: get physical coordinates of bounding box for each batched problem
    posYCell = get_grid_cartesian_coordinates(
        left_cuda, bottom_cuda, width, height, dx
    )

    # 4. Solve problem
    msg, resultAlpha, resultBeta, Nu_basic, solver = BatchSolveOnCell_CUDA(
        muXCell, muYCell, posXCell, posYCell, eps, alphaCell, subMuY,
        SinkhornError, SinkhornErrorRel, SinkhornMaxIter=SinkhornMaxIter,
        SinkhornInnerIter=SinkhornInnerIter
    )

    # 5. Turn back to numpy
    Nu_basic = Nu_basic.cpu().numpy()

    # 6. Balance. Plain DomDec code works here
    if balance:
        batch_balance(muXCell, Nu_basic)

    # 7. Extract new atomic muY and truncate
    MuYAtomicIndicesList, MuYAtomicDataList = unpack_cell_marginals_2D(
        Nu_basic, left, bottom, shapeY
    )

    # resultMuYAtomicDataList = []
    # # The batched version always computes directly the cell marginals
    # for i in range(BatchSize):
    #     resultMuYAtomicDataList.append([np.array(pi[i,j]) for j in range(pi.shape[1])])

    return resultAlpha, resultBeta, MuYAtomicDataList, MuYAtomicIndicesList, solver


def BatchSolveOnCell_CUDA(
    muXCell, muYCell, posX, posY, eps, alphaInit, muYref,
    SinkhornError=1E-4, SinkhornErrorRel=False, YThresh=1E-14, verbose=True,
    SinkhornMaxIter=10000, SinkhornInnerIter=10
):

    # Retrieve BatchSize
    B = muXCell.shape[0]
    dim = len(posX)
    assert dim == 2, "Not implemented for dimension != 2"

    # Retrieve cellsize
    s = muXCell.shape[-1] // 2

    # Define cost for solver
    C = (posX, posY)

    # Solve problem
    solver = LogSinkhornCudaImageOffset(
        muXCell, muYCell, C, eps, alpha_init=alphaInit, nuref=muYref,
        max_error=SinkhornError, max_error_rel=SinkhornErrorRel,
        max_iter=SinkhornMaxIter, inner_iter=SinkhornInnerIter
    )

    msg = solver.iterate_until_max_error()

    alpha = solver.alpha
    beta = solver.beta
    # Compute cell marginals directly
    pi_basic = get_cell_marginals(
        muXCell, muYref, alpha, beta, posX, posY, eps
    )

    return msg, alpha, beta, pi_basic, solver
