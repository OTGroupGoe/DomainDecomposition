import numpy as np
import torch
# from . import MultiScaleOT
from .LogSinkhorn import LogSinkhorn as LogSinkhorn
import LogSinkhornGPU

from . import DomainDecomposition as DomDec
import time

#########################################################
# Bounding box utils
# Cast all cell problems to a common size and coordinates
#########################################################


def pad_array(a, padding, pad_value=0):
    """
    Pad array `a` with a margin of `padding` width, filled with `pad_value`. 
    """
    shape = a.shape
    new_shape = tuple(s + 2*padding for s in shape)
    original_locations = tuple(slice(padding, s+padding) for s in shape)
    b = np.full(new_shape, pad_value, dtype=a.dtype)
    b[original_locations] = a
    return b


def pad_tensor(a, padding, pad_value=0):
    """
    Pad tensor `a` with a margin of `padding` width, filled with `pad_value`. 
    """
    shape = a.shape
    new_shape = tuple(s + 2*padding for s in shape)
    original_locations = tuple(slice(padding, s+padding) for s in shape)
    b = torch.full(new_shape, pad_value, dtype=a.dtype, device=a.device)
    b[original_locations] = a
    return b


def pad_replicate(a, padding):
    """
    Pad tensor `a` replicating values at boundary `paddding` times.
    """
    shape = a.shape
    if len(shape) == 1:
        replicate = torch.nn.ReplicationPad1d(padding)
    elif len(shape) == 2:
        replicate = torch.nn.ReplicationPad2d(padding)
    elif len(shape) == 3:
        replicate = torch.nn.ReplicationPad3d(padding)
    else:
        NotImplementedError("Not implemented for dim > 3")

    # nn expects batch and channel dimensions
    shape_nn = (1, 1)+shape
    a = a.view(shape_nn)
    if a.dtype == torch.int32:
        # Transform to double without losing precission and then back
        b = replicate(a.double()).int()
    else:
        b = replicate(a)
    b = b.view(tuple(s + 2 for s in shape))
    return b


def reformat_indices_2D(index, shapeY):
    """
    Takes linear indices and tuple of the total shapeY, and returns parameters 
    encoding the bounding box.
    """
    # TODO: generalize to 3D

    _, n = shapeY
    idx, idy = index // n, index % n
    if len(index) == 0:
        left, bottom = shapeY
        width = height = 0
    else:
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
    for (k, data) in enumerate(marg_data):
        Nu_box[k, idx[k], idy[k]] = data
        # Copy reference measure
        i0, w = left[k], width[k]
        j0, h = bottom[k], height[k]
        Nuref_box[k, :w, :h] = muY[i0:i0+w, j0:j0+h]

    # Return NuJ together with bounding box data (they will be needed for unpacking)
    return Nu_box, Nuref_box, left, bottom, max_width, max_height


def unpack_cell_marginals_2D(muY_basic, left, bottom, shapeY, threshold=1e-15):
    """
    Un-batch all the cell marginals and truncate entries below `threshold`.
    """
    # TODO: generalize to 3D
    # TODO: incorporate balancing
    # muY_basic is of size (B, 4, w, h), because there are 4 basic cells per
    # composite cell
    _, n = shapeY
    # Do all the process in the cpu
    B = muY_basic.shape[0]
    sizeJ = muY_basic.shape[1]
    marg_indices = []
    marg_data = []
    # NOTE: extraction could be done for the whole muY_basic at once,
    # but then how to distinguish between slices.
    for k in range(B):
        for i in range(sizeJ):  # TODO: allow different dimension
            # idx, idy = np.nonzero(muY_basic[k, i])
            idx, idy = np.nonzero(muY_basic[k, i] > threshold)
            linear_id = (idx+left[k])*n + idy + bottom[k]
            marg_indices.append(linear_id)
            marg_data.append(muY_basic[k, i, idx, idy])
    return marg_indices, marg_data

# TODO: complete this if possible
# def unpack_cell_marginals_2D_gpu(muY_basic, left, bottom, shapeY, threshold=1e-15):
#     """
#     Un-batch all the cell marginals and truncate entries below `threshold`.
#     """
#     # TODO: generalize to 3D
#     # Assumes muY_basic still in GPU
#     # muY_basic is of size (B, 4, w, h), because there are 4 basic cells per
#     # composite cell
#     _, n = shapeY
#     # Do all the process in the cpu
#     B = muY_basic.shape[0]
#     sizeJ = muY_basic.shape[1]
#     marg_indices = []
#     marg_data = []
#     idB, idx, idy = torch.where(muY_basic > threshold)
#     nu_entries = muY_basic[idB, idx, idy]
#     # Where batch index changes slice
#     steps = torch.where(torch.diff(idB))[0]+1
#     # Add start and end
#     steps = np.hstack(([0], steps.cpu().numpy(), [len(idB)]))
#     # TODO: continue here

#     # -------
#     # NOTE: extraction could be done for the whole muY_basic at once,
#     # but then how to distinguish between slices.
#     for k in range(B):
#         for i in range(sizeJ):  # TODO: allow different dimension
#             # idx, idy = np.nonzero(muY_basic[k, i])
#             idx, idy = np.nonzero(muY_basic[k, i] > threshold)
#             linear_id = (idx+left[k])*n + idy + bottom[k]
#             marg_indices.append(linear_id)
#             marg_data.append(muY_basic[k, i, idx, idy])
#     return marg_indices, marg_data


# def batch_balance(muXCell, muY_basic):
#     B, M, _ = muXCell.shape
#     s = M//2
#     atomic_mass = muXCell.view(B, 2, s, 2, s).permute(
#         (0, 1, 3, 2, 4)).contiguous().sum(dim=(3, 4))
#     atomic_mass = atomic_mass.view(B, -1).cpu().numpy()
#     for k in range(B):
#         status, muY_basic[k] = DomDec.BalanceMeasuresMulti(
#             muY_basic[k], atomic_mass[k], 1e-12, 1e-7
#         )
#         print(status, end = " ")
#     return muY_basic

# More straightforward version
def batch_balance(muXCell, muY_basic):
    B, M, _ = muXCell.shape
    s = M//2
    atomic_mass = muXCell.view(B, 2, s, 2, s).sum(dim=(2, 4))
    atomic_mass = atomic_mass.view(B, -1).cpu().numpy()
    muY_basic_shape = muY_basic.shape
    muY_basic = muY_basic.reshape(B, 4, -1)

    # TODO: Here we turn arrays to double so that LogSinkhorn.balanceMeasures
    # accepts them. This can be done cleaner
    for k in range(B):
        # print(atomic_mass[k])
        # print(muY_basic[k].sum(axis = (-1)))
        muY_basick_double = np.array(muY_basic[k], dtype=np.float64)
        atomic_massk_double = np.array(atomic_mass[k], dtype=np.float64)
        status, muY_basic[k] = DomDec.BalanceMeasuresMulti(
            muY_basick_double, atomic_massk_double, 1e-12, 1e-7
        )
        # print(atomic_mass[k] - muY_basic[k].sum(axis = (-1)))
        # print(status, end = " ")
    # print("")

    return muY_basic.reshape(*muY_basic_shape)


# 

def get_grid_cartesian_coordinates(left, bottom, w, h, dxs):
    """
    Generate the cartesian coordinates of the boxes with bottom-left corner 
    given by (left, bottom) (vectors), with width w, height h and spacing dx.
    i0 : torch.tensor
    j0 : torch.tensor
    w : int
    h : int
    dys : tensor of dy along each dimension
    """
    # TODO: generalize to 3D
    B = len(left)
    dx1, dy2 = dxs
    device = left.device
    x1_template = torch.arange(0, w*dx1, dx1, device=device)
    y2_template = torch.arange(0, h*dy2, dy2, device=device)
    x1 = left.view(-1, 1)*dx1 + x1_template.view(1, -1)
    y2 = bottom.view(-1, 1)*dy2 + y2_template.view(1, -1)
    return x1, y2

# def batch_shaped_cartesian_prod(xs):
#     """
#     For xs = (x1, ..., xd) a tuple of tensors of shape (B, M1), ... (B, Md), 
#     form the tensor X of shape (B, M1, ..., Md, d) such that
#     `X[i] = torch.cartesian_prod(xs[i],...,xs[i]).view(M1, ..., Md, d)`
#     """
#     B = xs[0].shape[0]
#     for x in xs:
#         assert B == x.shape[0], "All xs must have the same batch dimension"
#         assert len(x.shape) == 2, "xi must have shape (B, Mi)"
#     Ms = tuple(x.shape[1] for x in xs)
#     dim = len(xs)
#     device = xs[0].device

#     shapeX = (B, ) + Ms + (dim,)
#     X = torch.empty(shapeX, device=device)
#     for i in range(dim):
#         shapexi = (B,) + (1,)*i + (Ms[i],) + (1,)*(dim-i-1)
#         X[..., i] = xs[i].view(shapexi)
#     return X

# def compute_offsets_sinkhorn_grid(xs, ys, eps):
#     """
#     Compute offsets
#     xs and ys are d-tuples of tensors with shape (B, Mi) where B is the batch 
#     dimension and Mi the size of the grid in that coordinate
#     # TODO: ref
#     """
#     # Get cartesian prod
#     X = batch_shaped_cartesian_prod(xs)
#     Y = batch_shaped_cartesian_prod(ys)
#     shapeX = X.shape
#     B, Ms, dim = shapeX[0], shapeX[1:-1], shapeX[-1]
#     Ns = Y.shape[1:-1]

#     # Get "bottom left" corner coordinates: select slice (:, 0, ..., 0, :)
#     X0 = X[(slice(None),) + (0,)*dim + (slice(None),)] \
#         .view((B,) + (1,)*dim + (dim,))  # NOTE alternatively: use unpack op.
#     Y0 = Y[(slice(None),) + (0,)*dim + (slice(None),)] \
#         .view((B,) + (1,)*dim + (dim,))  # NOTE alternatively: use unpack op.

#     # Use the formulas in [TODO: ref] to compute the offset
#     offsetX = torch.sum(2*(X-X0)*(Y0-X0), dim=-1)/eps
#     offsetY = torch.sum(2*(Y-Y0)*(X0-Y0), dim=-1)/eps
#     offset_constant = -torch.sum((X0-Y0)**2, dim=-1)/eps
#     return offsetX, offsetY, offset_constant




def get_refined_marginals_gpu(muYL, muYL_old, parentsYL,
                              atomic_masses, atomic_masses_old,
                              atomic_cells, atomic_cells_old,
                              atomic_data_old, atomic_indices_old,
                              meta_cell_shape, meta_cell_shape_old):
    # Get "physical" basic cells, where mass actually sits
    # For old cells
    cell_indices_old = np.arange(np.prod(meta_cell_shape_old))
    true_indices_old = cell_indices_old.reshape(meta_cell_shape_old)
    # print(cell_indices_old, true_indices_old)
    # print(len(atomic_data_old), len(atomic_indices_old))
    # TODO: this is 2D
    true_indices_old = true_indices_old[1:-1, 1:-1].ravel()
    true_atomic_cells_old = [atomic_cells_old[i] for i in true_indices_old]
    # TODO: reconcile sparse and gpu versions
    if len(atomic_data_old) == len(true_indices_old):
        true_atomic_data_old = atomic_data_old
        true_atomic_indices_old = atomic_indices_old
    else:
        true_atomic_data_old = [atomic_data_old[i] for i in true_indices_old]
        true_atomic_indices_old = [atomic_indices_old[i]
                                   for i in true_indices_old]

    true_atomic_masses_old = atomic_masses_old[true_indices_old]

    # For new cells
    cell_indices = np.arange(np.prod(meta_cell_shape))
    true_indices = cell_indices.reshape(meta_cell_shape)
    true_indices = true_indices[1:-1, 1:-1].ravel()
    true_atomic_cells = [atomic_cells[i] for i in true_indices]
    true_atomic_masses = atomic_masses[true_indices]
    true_meta_cell_shape = tuple(s-2 for s in meta_cell_shape)

    # Invoke domdec function
    true_atomic_data, true_atomic_indices = \
        DomDec.GetRefinedAtomicYMarginals_SparseY(
            muYL, muYL_old, parentsYL,
            true_atomic_masses, true_atomic_masses_old,
            true_atomic_cells, true_atomic_cells_old,
            true_atomic_data_old, true_atomic_indices_old,
            true_meta_cell_shape
        )
    dtype = true_atomic_data[0].dtype
    atomic_data = [np.array([], dtype=dtype) for _ in cell_indices]
    atomic_indices = [np.array([], dtype=np.int32) for _ in cell_indices]
    for (i, k) in enumerate(true_indices):
        atomic_data[k] = true_atomic_data[i]
        atomic_indices[k] = true_atomic_indices[i]

    return atomic_data, atomic_indices

##############################################################
# Dedicated CUDA solver for DomDec:
# Assumes B rectangular problems with same size
##############################################################


# class LogSinkhornCudaImageOffset(LogSinkhornGPU.AbstractSinkhorn):
#     """
#     Online Sinkhorn solver for standard OT on images with separable cost, 
#     custom CUDA implementation. 
#     Each Sinkhorn iteration has complexity N^(3/2), instead of the usual N^2. 

#     Attributes
#     ----------
#     mu : torch.Tensor 
#         of size (B, M1, M2)
#         First marginals
#     nu : torch.Tensor 
#         of size (B, N1, N2)
#         Second marginals 
#     C : tuple 
#         of the form ((x1, x2), (y1, y2))
#         Grid coordinates
#     eps : float
#         Regularization strength
#     muref : torch.Tensor 
#         with same dimensions as mu (except axis 0, which can have len = 1)
#         First reference measure for the Gibbs energy, 
#         i.e. K = muref \otimes nuref exp(-C/eps)
#     nuref : torch.Tensor 
#         with same dimensions as nu (except axis 0, which can have len = 1)
#         Second reference measure for the Gibbs energy, 
#         i.e. K = muref \otimes nuref exp(-C/eps)
#     alpha_init : torch.Tensor 
#         with same dimensions as mu, or None
#         Initialization for the first Sinkhorn potential
#     """

#     def __init__(self, mu, nu, C, eps, **kwargs):
#         (xs, ys) = C
#         zs = xs + ys  # Have all coordinates in one tuple
#         x1 = zs[0]
#         B = LogSinkhornGPU.batch_dim(mu)
#         # Check whether xs have a batch dimension
#         if len(x1.shape) == 1:
#             for z in zs:
#                 assert len(z) == 1, \
#                     "dimensions of grid coordinates must be consistent"
#             C = tuple(tuple(xi.view(1, -1) for xi in X) for X in C)
#         else:
#             for z in zs:
#                 assert len(z.shape) == 2, \
#                     "coordinates can just have one spatial dimension"
#                 assert z.shape[0] == B, \
#                     "batch dimension of all coordinates must coincide"

#         # Now all coordinates have a batch dimension of either B or 1.
#         # Check that all coordinates have same grid spacing
#         if x1.shape[1] == 1:
#             print("WARNING: x seems to have length 1. Setting dx = 1")
#             dx = 1
#         else:
#             dx = x1[0, 1]-x1[0, 0]
#             for z in zs:
#                 if z.shape[-1] > 1:  # otherwise diff yields shape 0
#                     assert torch.max(torch.abs(torch.diff(z, dim=-1)-dx)) < 1e-6, \
#                         "Grid is not equispaced"

#         # Check geometric dimensions
#         Ms = LogSinkhornGPU.geom_dims(mu)
#         Ns = LogSinkhornGPU.geom_dims(nu)
#         assert len(Ms) == len(Ns) == 2, "Shapes incompatible with images"

#         # Compute the offsets
#         self.offsetX, self.offsetY, self.offset_const = \
#             compute_offsets_sinkhorn_grid(xs, ys, eps)

#         # Save xs and ys in case they are needed later
#         self.xs = xs
#         self.ys = ys

#         C = (dx, Ms, Ns)

#         super().__init__(mu, nu, C, eps, **kwargs)

#     def get_new_alpha(self):
#         dx, Ms, Ns = self.C
#         h = self.beta / self.eps + self.lognuref + self.offsetY
#         return - self.eps * (
#             LogSinkhornGPU.softmin_cuda_image(h, Ms, Ns, self.eps, dx)
#             + self.offsetX + self.offset_const + self.logmuref - self.logmu
#         )

#     def get_new_beta(self):
#         dx, Ms, Ns = self.C
#         h = self.alpha / self.eps + self.logmuref + self.offsetX
#         return - self.eps * (
#             LogSinkhornGPU.softmin_cuda_image(h, Ns, Ms, self.eps, dx)
#             + self.offsetY + self.offset_const + self.lognuref - self.lognu
#         )

#     def get_dense_cost(self, ind=None):
#         """
#         Get dense cost matrix of given problems. If no argument is given, all 
#         costs are computed. Can be memory intensive, so it is recommended to do 
#         small batches at a time.
#         `ind` must be slice or iterable, not int.
#         """

#         if ind == None:
#             ind = slice(None,)

#         xs = tuple(x[ind] for x in self.xs)
#         ys = tuple(y[ind] for y in self.ys)
#         X = batch_shaped_cartesian_prod(xs)
#         Y = batch_shaped_cartesian_prod(ys)
#         B = X.shape[0]
#         dim = X.shape[-1]
#         C = ((X.view(B, -1, 1, dim) - Y.view(B, 1, -1, dim))**2).sum(dim=-1)
#         return C, X, Y

#     def get_dense_plan(self, ind=None, C=None):
#         """
#         Get dense plans of given problems. If no argument is given, all plans 
#         are computed. Can be memory intensive, so it is recommended to do small 
#         batches at a time.
#         `ind` must be slice or iterable, not int.
#         """
#         if ind == None:
#             ind = slice(None,)

#         if C == None:
#             C, _, _ = self.get_dense_cost(ind)

#         B = C.shape[0]
#         alpha, beta = self.alpha[ind], self.beta[ind]
#         muref, nuref = self.muref[ind], self.nuref[ind]

#         pi = torch.exp(
#             (alpha.view(B, -1, 1) + beta.view(B, 1, -1) - C) / self.eps
#         ) * muref.view(B, -1, 1) * nuref.view(B, 1, -1)
#         return pi

#     def change_eps(self, new_eps):
#         """
#         Change the regularization strength `self.eps`.
#         In this solver this also involves renormalizing the offsets.
#         """
#         self.Niter = 0
#         self.current_error = self.max_error + 1.
#         scale = self.eps / new_eps
#         self.offsetX = self.offsetX * scale
#         self.offsetY = self.offsetY * scale
#         self.offset_const = self.offset_const * scale
#         self.eps = new_eps

#     def get_dx(self):
#         return self.C[0]


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

def get_dx(x, B):
    """
    Get spacing between points in `x`, checking that grid is 
    2-dimensional (batch + physical dimension), sharing same 
    batch dimension `B` and equispaced.
    """
    assert len(x.shape) == 2, "x must have a batch and physical dim"
    assert x.shape[0] == B, "x must have `B` as batch dim"
    if x.shape[1] == 1:
        return 1.0
    else:
        # Check that gridpoints are equispaced
        diff = torch.diff(x, dim=1)
        dx = diff[0,0]
        assert torch.allclose(diff, dx, rtol=1e-4), \
            "grid points must be equispaced"
        return dx.item()

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
    B = LogSinkhornGPU.batch_dim(muref)

    # Deduce cellsize
    s = Ms[0]//2
    dim = len(xs)
    assert dim == 2, "Not implemented for dimension rather than 2"
    n_basic = 2**dim  # number of basic cells per composite cell, depends on dim

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
    dxs = torch.tensor([get_dx(xi, xi.shape[0]) for xi in xs_b]).cpu()
    dys = torch.tensor([get_dx(yj, yj.shape[0]) for yj in ys_b]).cpu()

    offsetX, offsetY, offset_const = LogSinkhornGPU.compute_offsets_sinkhorn_grid(
        xs_b, ys_b, eps)
    h = alpha_b / eps + logmu_b + offsetX

    beta_hat = - eps * (
        LogSinkhornGPU.softmin_cuda_image(h, Ns, new_Ms, eps, dys, dxs)
        + offsetY + offset_const
    )

    # Turn to double to improve accuracy
    # beta = beta.double()
    # beta_hat = beta_hat.double()
    # nuref = nuref.double()

    # Build cell marginals
    muY_basic = nuref[:, None] * torch.exp(
        (beta[:, None] - beta_hat.view(-1, n_basic, *Ns))/eps
    )
    return muY_basic


def BatchIterate(
    muY, posY, dxs_dys, eps,
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
            resultMuYCellIndicesList, info = BatchDomDecIteration_CUDA(
                SinkhornError, SinkhornErrorRel, muY, posY, dxs_dys, eps, shapeY,
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

    return alphaJ, muYAtomicIndicesList, muYAtomicDataList, info
    # for jsub,j in enumerate(partitionDataCompCellsBatch):
    #     muYAtomicDataList[j]=resultMuYAtomicDataList[jsub]
    #     muYAtomicIndicesList[j]=muYCellIndices.copy()


def BatchDomDecIteration_CUDA(
        SinkhornError, SinkhornErrorRel, muY, posYCell, dxs_dys, eps, shapeY,
        # partitionDataCompCellIndices,
        muXCell, posXCell, alphaCell, muYAtomicListData, muYAtomicListIndices,
        SinkhornMaxIter, SinkhornInnerIter, BatchSize, balance=True):

    # 1: compute composite cell marginals
    dxs, dys = dxs_dys
    info = dict()
    muYCellData = []
    muYCellIndices = []
    t0 = time.perf_counter()
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
        muYCellIndices, muYCellData, shapeY, muY
    )
    info["time_bounding_box"] = time.perf_counter() - t0
    info["bounding_box"] = (width, height)
    # Turn muYCell, left, bottom and subMuY into tensor
    device = muXCell.device
    dtype = muXCell.dtype
    muYCell = torch.tensor(muYCell, device=device, dtype=dtype)
    subMuY = torch.tensor(subMuY, device=device, dtype=dtype)
    left_cuda = torch.tensor(left, device=device)
    bottom_cuda = torch.tensor(bottom, device=device)

    # 3: get physical coordinates of bounding box for each batched problem
    posYCell = get_grid_cartesian_coordinates(
        left_cuda, bottom_cuda, width, height, dys
    )

    # 4. Solve problem

    t0 = time.perf_counter()
    resultAlpha, resultBeta, muY_basic, info_solver = BatchSolveOnCell_CUDA(
        muXCell, muYCell, posXCell, posYCell, eps, alphaCell, subMuY,
        SinkhornError, SinkhornErrorRel, SinkhornMaxIter=SinkhornMaxIter,
        SinkhornInnerIter=SinkhornInnerIter
    )
    info["time_sinkhorn"] = time.perf_counter() - t0
    # Renormalize muY_basic
    # print("shape nu basic", muY_basic.shape)
    muY_basic = muY_basic * \
        (muYCell / (muY_basic.sum(dim=1) + 1e-30))[:, None, :, :]

    # # 5. Turn back to numpy
    # muY_basic = muY_basic.cpu().numpy()

    # # 6. Balance. Plain DomDec code works here
    # t0 = time.perf_counter()
    # if balance:
    #     batch_balance(muXCell, muY_basic)
    # info["time_balance"] = time.perf_counter() - t0

    # 5. CUDA balance
    t0 = time.perf_counter()
    if balance:
        CUDA_balance(muXCell, muY_basic)
    info["time_balance"] = time.perf_counter() - t0

    # 6. Turn back to numpy
    muY_basic = muY_basic.cpu().numpy()

    # 7. Extract new atomic muY and truncate
    t0 = time.perf_counter()
    MuYAtomicIndicesList, MuYAtomicDataList = unpack_cell_marginals_2D(
        muY_basic, left, bottom, shapeY
    )
    info["time_truncation"] = time.perf_counter() - t0

    # resultMuYAtomicDataList = []
    # # The batched version always computes directly the cell marginals
    # for i in range(BatchSize):
    #     resultMuYAtomicDataList.append([np.array(pi[i,j]) for j in range(pi.shape[1])])
    info = {**info, **info_solver}
    return resultAlpha, resultBeta, MuYAtomicDataList, MuYAtomicIndicesList, info


def BatchSolveOnCell_CUDA(
    muXCell, muYCell, posX, posY, eps, alphaInit, muYref,
    SinkhornError=1E-4, SinkhornErrorRel=False, YThresh=1E-14, verbose=True,
    SinkhornMaxIter=10000, SinkhornInnerIter=10
):

    # Retrieve BatchSize
    # TODO: clean up
    B = muXCell.shape[0]
    dim = len(posX)
    assert dim == 2, "Not implemented for dimension != 2"

    # Retrieve cellsize
    s = muXCell.shape[-1] // 2

    # Define cost for solver
    C = (posX, posY)
    # Solve problem
    solver = LogSinkhornGPU.LogSinkhornCudaImageOffset(
        muXCell, muYCell, C, eps, alpha_init=alphaInit, nuref=muYref,
        max_error=SinkhornError, max_error_rel=SinkhornErrorRel,
        max_iter=SinkhornMaxIter, inner_iter=SinkhornInnerIter
    )

    msg = solver.iterate_until_max_error()

    alpha = solver.alpha
    beta = solver.beta
    # Compute cell marginals directly
    muY_basic = get_cell_marginals(
        muXCell, muYref, alpha, beta, posX, posY, eps
    )

    # Wrap solver and possibly runtime info into info dictionary
    info = {
        "solver": solver,
        "msg": msg
    }

    return alpha, beta, muY_basic, info


# def BatchSolveOnCell_CUDA_float(
#     muXCell, muYCell, posX, posY, eps, alphaInit, muYref,
#     SinkhornError=1E-4, SinkhornErrorRel=False, YThresh=1E-14, verbose=True,
#     SinkhornMaxIter=10000, SinkhornInnerIter=10
# ):
#     """
#     Solve, converting first to float, and then undoing the conversion
#     """

#     # Retrieve BatchSize
#     B = muXCell.shape[0]
#     dim = len(posX)
#     assert dim == 2, "Not implemented for dimension != 2"

#     # Retrieve cellsize
#     s = muXCell.shape[-1] // 2

#     # Define cost for solver
#     C = (tuple(xi.float() for xi in posX), tuple(yi.float() for yi in posY))
#     # Solve problem
#     solver = LogSinkhornGPU.LogSinkhornCudaImageOffset(
#         muXCell.float(), muYCell.float(), C, eps, alpha_init=alphaInit.float(),
#         nuref=muYref.float(), max_error=SinkhornError,
#         max_error_rel=SinkhornErrorRel, max_iter=SinkhornMaxIter,
#         inner_iter=SinkhornInnerIter
#     )

#     msg = solver.iterate_until_max_error()

#     # print(f"solving with {solver.alpha.dtype}")
#     alpha = solver.alpha
#     beta = solver.beta
#     # Compute cell marginals directly
#     pi_basic = get_cell_marginals(
#         solver.muref, solver.nuref, alpha, beta, C[0], C[1], eps
#     )
#     # print(f"pi is {pi_basic.dtype}")

#     # Wrap solver and possibly runtime info into info dictionary
#     info = {
#         "solver": solver,
#         "msg": msg
#     }

#     return alpha.double(), beta.double(), pi_basic.double(), info

#######
# Duals
#######


def get_alpha_field_gpu(alpha, shape, cellsize):
    # TODO: generalize to 3D
    comp_shape = tuple(s // (2*cellsize) for s in shape)
    alpha_field = alpha.view(*comp_shape, 2*cellsize, 2*cellsize) \
                       .permute(0, 2, 1, 3).contiguous().view(shape)
    return alpha_field


def get_alpha_field_even_gpu(alphaA, alphaB, shapeXL, shapeXL_pad,
                             cellsize, basic_shape, muX=None,
                             requestAlphaGraph=False):
    """
    Uses alphaA, alphaB and getAlphaGraph to compute one global dual variable
    alpha from alphaAList and alphaBList."""
    dim = len(alphaA.shape)-1
    alphaA_field = get_alpha_field_gpu(alphaA, shapeXL, cellsize)
    alphaB_field = get_alpha_field_gpu(alphaB, shapeXL_pad, cellsize)
    # Remove padding
    # TODO: generalize to 3D
    s1, s2 = shapeXL
    alphaB_field = alphaB_field[cellsize:-cellsize, cellsize:-cellsize]
    alphaDiff = (alphaA_field-alphaB_field).cpu().numpy().ravel()
    alphaGraph = DomDec.getAlphaGraph(
        alphaDiff, basic_shape, cellsize, muX
    )

    # Each offset in alphaGraph is for one of the batched problems in alphaA
    alphaGraphGPU = torch.tensor(
        alphaGraph, device=alphaA.device, dtype=alphaA.dtype
    )
    alphaAEven = alphaA - alphaGraphGPU.view(-1, *np.ones(dim, dtype=np.int32))
    # alphaFieldEven = alphaA_field.cpu().numpy()
    # for a, c in zip(alphaGraph.ravel(), cellsA):
    #     print(c)
    #     alphaFieldEven[np.array(c)] -= a
    alphaFieldEven = get_alpha_field_gpu(alphaAEven, shapeXL, cellsize)
    # alphaFieldEven = alphaFieldEven.cpu().numpy().ravel()

    if requestAlphaGraph:
        return (alphaFieldEven, alphaGraph)

    return alphaFieldEven


###################################################
# Try to do everything in the bounding box format
###################################################

# CUDA version of balancing
def CUDA_balance(muXCell, muY_basic):
    B, M, _ = muXCell.shape
    s = M//2
    atomic_mass = muXCell.view(B, 2, s, 2, s).sum(dim=(2, 4))
    atomic_mass = atomic_mass.view(B, -1)
    muY_basic_shape = muY_basic.shape
    muY_basic = muY_basic.view(B, 4, -1)
    atomic_mass_nu = muY_basic.sum(-1)
    mass_delta = atomic_mass_nu - atomic_mass
    # print(f"balancing with {muY_basic.dtype}")
    threshold = torch.tensor(1e-12)
    # Call LogSinkhornGPU backend function 
    LogSinkhornGPU.backend.BalanceCUDA(muY_basic, mass_delta, threshold)
    return muY_basic.view(*muY_basic_shape)


# TODO: code for 3D
###############################################
# transform basic cell utilities
###############################################

# CUDA functionality to get relative extents of bounding boxes


# # Deprecated, just works for basic to composite. New works for any
# # combining
# def get_axis_bounds(muY_basic, mask, global_minus, axis):
#     B, C = muY_basic.shape[:2]
#     geom_shape = muY_basic.shape[2:]
#     n = geom_shape[axis]
#     # Put in the position of every point with mass its index along axis
#     index_axis = torch.arange(n, device=muY_basic.device, dtype=torch.int)
#     new_shape_index = [n if i == 2 +
#                        axis else 1 for i in range(len(muY_basic.shape))]
#     index_axis = index_axis.view(new_shape_index)
#     mask_index = mask*index_axis
#     # Get positive extreme
#     basic_plus = mask_index.view(B, C, -1).amax(-1)
#     # Turn zeros to upper bound so that we can get the minimum
#     mask_index[~mask] = n
#     basic_minus = mask_index.view(B, C, -1).amin(-1)
#     basic_extent = basic_plus - basic_minus + 1
#     # Add global offsets
#     global_basic_minus = global_minus + basic_minus
#     global_basic_plus = global_minus + basic_plus
#     # Reduce to composite cell
#     global_composite_minus = global_basic_minus.amin(-1)
#     global_composite_plus = global_basic_plus.amax(-1)
#     composite_extent = global_composite_plus - global_composite_minus + 1
#     # Get dim of bounding box
#     max_composite_extent = torch.max(composite_extent).item()
#     relative_basic_minus = global_basic_minus - \
#         global_composite_minus.view(-1, 1)
#     return relative_basic_minus, basic_minus, basic_extent, global_composite_minus, max_composite_extent

def get_axis_bounds(muY_basic, mask, global_minus, axis, sum_indices):
    B = muY_basic.shape[0]
    geom_shape = muY_basic.shape[1:]
    n = geom_shape[axis]
    # Put in the position of every point with mass its index along axis
    index_axis = torch.arange(n, device=muY_basic.device, dtype=torch.int32)
    new_shape_index = [
        n if i == 1 + axis else 1 for i in range(len(muY_basic.shape))
    ]
    index_axis = index_axis.view(new_shape_index)
    mask_index = mask*index_axis
    # Get positive extreme
    basic_plus = mask_index.view(B, -1).amax(-1)
    # Turn zeros to upper bound so that we can get the minimum
    mask_index[~mask] = n
    basic_minus = mask_index.view(B, -1).amin(-1)
    basic_extent = basic_plus - basic_minus + 1
    # Add global offsets
    global_basic_minus = global_minus + basic_minus
    global_basic_plus = global_minus + basic_plus

    # Remove -1's in sum_indices
    sum_indices_clean = sum_indices.clone().long()
    idx, idy = torch.where(sum_indices_clean < 0)
    # Each composite cell comes at least from one basic
    sum_indices_clean[idx, idy] = sum_indices_clean[idx, 0]

    # Reduce to composite cell
    global_composite_minus = global_basic_minus[sum_indices_clean].amin(-1)
    global_composite_plus = global_basic_plus[sum_indices_clean].amax(-1)
    composite_extent = global_composite_plus - global_composite_minus + 1
    # print("axis =", axis, "composite_extent =\n", composite_extent)

    # Turn basic_minus and basic_extent into shape of sum_indices
    basic_minus = basic_minus[sum_indices_clean]
    basic_extent = basic_extent[sum_indices_clean]

    # Get dim of bounding box
    max_composite_extent = torch.max(composite_extent).item()
    relative_basic_minus = global_basic_minus[sum_indices_clean] - \
        global_composite_minus.view(-1, 1)
    return (relative_basic_minus, basic_minus, basic_extent,
            global_composite_minus, max_composite_extent)


def combine_cells(muY_basic, global_left, global_bottom, sum_indices, weights=1):
    """
    Combine basic cells in muY_basic, taking into account their global boundaries 
    to yield Nu_comp. Nu_comp[j] is the result of combining all the cells with 
    indices in sum_indices[j], each multiplied by its corresponding weight in
    weights[j].
    """
    mask = muY_basic > 0.0

    if type(weights) in [int, float]:
        torch_options = dict(dtype=muY_basic.dtype, device=muY_basic.device)
        weights = torch.full(sum_indices.shape, weights, **torch_options)

    # Get bounding box parameters
    relative_basic_left, basic_left, basic_width, \
        global_composite_left, composite_width = \
        get_axis_bounds(muY_basic, mask, global_left, 0, sum_indices)

    relative_basic_bottom, basic_bottom, basic_height, \
        global_composite_bottom, composite_height = \
        get_axis_bounds(muY_basic, mask, global_bottom, 1, sum_indices)

    Nu_comp = LogSinkhornGPU.backend.AddWithOffsetsCUDA_2D(
        muY_basic, composite_width, composite_height,
        weights, sum_indices,
        relative_basic_left, basic_left, basic_width,
        relative_basic_bottom, basic_bottom, basic_height
    )

    return Nu_comp, global_composite_left, global_composite_bottom

# # Deprecated: use better the general combine_cells
# def basic_to_composite_CUDA_2D(muY_basic, global_left, global_bottom):
#     B, C = muY_basic.shape[:2]
#     mask = muY_basic > 0.0
#     # Get bounding box parameters
#     relative_basic_left, basic_left, basic_width, \
#         global_composite_left, composite_width = \
#         get_axis_bounds(muY_basic, mask, global_left, 0)
#     relative_basic_bottom, basic_bottom, basic_height, \
#         global_composite_bottom, composite_height = \
#         get_axis_bounds(muY_basic, mask, global_bottom, 1)

#     Nu_comp = LogSinkhornGPU.BasicToCompositeCUDA_2D(
#         muY_basic, composite_width, composite_height,
#         relative_basic_left, basic_left, basic_width,
#         relative_basic_bottom, basic_bottom, basic_height
#     )
#     return Nu_comp, global_composite_left, global_composite_bottom


def basic_to_composite_CUDA_2D(muY_basic, left, bottom, basic_shape, partition):
    # muY_basic is of shape (B, s1, ..., sd)
    # TODO: change signature left,bottom to allow for higher dimensional
    torch_options_int = dict(dtype=torch.int32, device=muY_basic.device)
    torch_options = dict(dtype=muY_basic.dtype, device=muY_basic.device)
    dim = len(basic_shape)
    basic_indices = torch.arange(
        np.prod(basic_shape), **torch_options_int).view(basic_shape)
    weights = torch.ones(basic_shape, **torch_options)

    # print("Basic to composite")
    # For A cells do nothing, for B cells pad
    if partition == "B":
        basic_indices = pad_replicate(basic_indices, 1)
        weights = pad_tensor(weights, 1, pad_value=0.0)
        basic_shape = tuple(s+2 for s in basic_shape)

    # Turn sum_indices and weights to composite shape
    if dim == 1:
        b1 = basic_shape[0]
        sum_indices = basic_indices.view(b1//2, 2)
        weights = weights.view(b1//2, 2)
    if dim == 2:
        b1, b2 = basic_shape
        sum_indices = basic_indices.view(b1//2, 2, b2//2, 2) \
            .permute(0, 2, 1, 3).reshape((b1*b2)//4, 4)
        weights = weights.view(b1//2, 2, b2//2, 2) \
            .permute(0, 2, 1, 3).reshape((b1*b2)//4, 4)
    elif dim == 3:
        b1, b2, b3 = basic_shape
        sum_indices = basic_indices.view(b1//2, 2, b2//2, 2, b3//2, 2) \
            .permute(0, 2, 4, 1, 3, 5).reshape((b1*b2*b3)//8, 8)
        weights = weights.view(b1//2, 2, b2//2, 2, b3//2, 2) \
            .permute(0, 2, 4, 1, 3, 5).reshape((b1*b2*b3)//8, 8)
    else:
        raise NotImplementedError("not implemented for dim > 3")

    return combine_cells(muY_basic, left, bottom, sum_indices, weights)

    ####################################
    # previous implementation
    # b1, b2 = muY_basic.shape[:2]
    # B = b1*b2
    # # TODO: why are these not possible with view?
    # muY_basic = muY_basic.reshape(B, *muY_basic.shape[2:])
    # left = left.reshape(B)
    # bottom = bottom.reshape(B)

    # # Get indices along which to sum
    # torch_options_int = dict(dtype=torch.int32, device=muY_basic.device)
    # sum_indices = torch.arange(B, **torch_options_int).view(b1//2, 2, b2//2, 2) \
    #     .permute(0, 2, 1, 3).reshape(B//4, 4)

    # return combine_cells(muY_basic, left, bottom, sum_indices)


def unpack_cell_marginals_2D_box(muY_basic, left, bottom, shapeY):
    """
    Un-batch all the cell marginals and truncate entries below `threshold`.
    """
    # TODO: generalize to 3D
    # muY_basic is of size (B, 4, w, h), because there are 4 basic cells per
    # left and bottom are of size (B, 4)
    _, n = shapeY
    # Do all the process in the cpu
    B = muY_basic.shape[0]
    marg_indices = []
    marg_data = []
    # Take to cpu if they are not
    # NOTE: extraction could be done for the whole muY_basic at once,
    # but then how to distinguish between slices.
    if type(muY_basic) == torch.Tensor:
        muY_basic = muY_basic.cpu().numpy()
        left = left.cpu().numpy()
        bottom = bottom.cpu().numpy()
    for k in range(B):
        # idx, idy = np.nonzero(muY_basic[k, i])
        idx, idy = np.nonzero(muY_basic[k] > 0)
        linear_id = (idx+left[k])*n + idy + bottom[k]
        marg_indices.append(linear_id)
        marg_data.append(muY_basic[k, idx, idy])
    return marg_indices, marg_data


def BatchIterateBox(
    muY, posY, dxs_dys, eps,
    muXJ, posXJ, alphaJ,
    muY_basic, left, bottom, shapeY,
    basic_shape, partition="A",
    SinkhornError=1E-4, SinkhornErrorRel=False, SinkhornMaxIter=None,
    SinkhornInnerIter=100, BatchSize=np.inf
):

    # Solve batch
    # TODO: minibatches
    alphaJ, betaJ, muY_basic, left, bottom, info = \
        BatchDomDecIterationBox_CUDA(
            SinkhornError, SinkhornErrorRel, muY, posY, dxs_dys, eps, shapeY,
            muXJ, posXJ, alphaJ,
            muY_basic, left, bottom, basic_shape, partition,
            SinkhornMaxIter, SinkhornInnerIter
        )

    return alphaJ, muY_basic, left, bottom, info
    # for jsub,j in enumerate(partitionDataCompCellsBatch):
    #     muYAtomicDataList[j]=resultMuYAtomicDataList[jsub]
    #     muYAtomicIndicesList[j]=muYCellIndices.copy()


# TODO: finish this
# def BatchIterateBox(
#     muY, posY, dx, eps,
#     muXJ, posXJ, alphaJ,
#     muY_basic, left, bottom, shapeY,
#     partition="A",
#     SinkhornError=1E-4, SinkhornErrorRel=False, SinkhornMaxIter=None,
#     SinkhornInnerIter=100, BatchSize=np.inf
# ):
#     torch_options = dict(device = muY_basic.device, dtype = muY_basic.dtype)
#     # assert BatchSize == np.inf, "not implemented for partial BatchSize"
#     if partition == "A":
#         # Global muY_basic has shape (b1, b2, w, h)
#         # TODO: generalize for 3D
#         muY_basic_part = muY_basic
#         left_part = left
#         bottom_part = bottom
#     else:
#         (b1, b2, w, h) = muY_basic.shape
#         muY_basic_part = torch.zeros((b1+2, b2+2, w, h), **torch_options)
#         muY_basic_part[1:-1,1:-1] = muY_basic
#         # Pad left and bottom with extension of actual values
#         left_part = pad_replicate(left,1)
#         bottom_part = pad_replicate(bottom,1)

#     BatchTotal = muXJ.shape[0]
#     if BatchSize == np.inf:
#         BatchSize = BatchTotal

#     else:
#         # Make sure that BatchSize is an integer number of muY_basic rows
#         print(muY_basic_part.shape)
#         b1, b2 = muY_basic_part.shape[:2]
#         BatchSize = (2*b2)* BatchSize // (2*b2)
#     muY_basic_results = []

#     # Divide problem data into batches of size BatchSize
#     for i in range(0, BatchTotal, BatchSize):
#         print(BatchSize, (i+BatchSize)//b2)
#         muXBatch = muXJ[i:i+BatchSize]
#         posXBatch = tuple(x[i:i+BatchSize] for x in posXJ)
#         alphaBatch = alphaJ[i:i+BatchSize]
#         muY_basic_batch = muY_basic_part[i//b2:(i+BatchSize)//b2]
#         left_batch = left_part[i//b2:(i+BatchSize)//b2].clone()
#         bottom_batch = bottom_part[i//b2:(i+BatchSize)//b2].clone()
#         # Solve batch
#         resultAlpha, resultBeta, muY_basic_batch, left_batch, bottom_batch, info = \
#             BatchDomDecIterationBox_CUDA(
#                 SinkhornError, SinkhornErrorRel, muY, posY, dx, eps, shapeY,
#                 muXBatch, posXBatch, alphaBatch,
#                 muY_basic_batch, left_batch, bottom_batch,
#                 SinkhornMaxIter, SinkhornInnerIter
#             )
#         alphaJ[i:i+BatchSize] = resultAlpha
#         muY_basic_results.append(muY_basic_batch) # must match shape later
#         left_part[i//b2:(i+BatchSize)//b2] = left_batch
#         bottom_part[i//b2:(i+BatchSize)//b2] = bottom_batch

#     # Joint all partial results together
#     max_width = max(nu_i.shape[2] for nu_i in muY_basic_results)
#     max_height = max(nu_i.shape[3] for nu_i in muY_basic_results)
#     nu_result_part = torch.zeros((b1, b2, max_width, max_height), **torch_options)
#     for i in range(0, BatchTotal, BatchSize):
#         nu_result_part[i//b2:(i+BatchSize)//b2] = muY_basic_results[i//BatchSize]

#     if partition == "A":
#         # TODO: generalize for 3D

#         muY_basic = muY_basic_part
#         left = left_part
#         bottom = bottom_part
#     else:
#         muY_basic = muY_basic_part[1:-1,1:-1]
#         left = left_part[1:-1,1:-1]
#         bottom = bottom_part[1:-1,1:-1]

#     return resultAlpha, muY_basic, left, bottom, info
#     # for jsub,j in enumerate(partitionDataCompCellsBatch):
#     #     muYAtomicDataList[j]=resultMuYAtomicDataList[jsub]
#     #     muYAtomicIndicesList[j]=muYCellIndices.copy()


def BatchDomDecIterationBox_CUDA(
        SinkhornError, SinkhornErrorRel, muY, posYCell, dxs_dys, eps, shapeY,
        # partitionDataCompCellIndices,
        muXCell, posXCell, alphaCell,
        muY_basic, left, bottom, basic_shape, partition,
        SinkhornMaxIter, SinkhornInnerIter, balance=True):

    info = dict()
    # 1: compute composite cell marginals
    # Get basic shape size
    dxs, dys = dxs, dys
    t0 = time.perf_counter()
    _, w0, h0 = muY_basic.shape
    torch_options = dict(device=muY_basic.device, dtype=muY_basic.dtype)
    torch_options_int = dict(device=muY_basic.device, dtype=torch.int32)
    b1, b2 = basic_shape
    # muY_basic = muY_basic.reshape(c1, 2, c2, 2, w0, h0).permute(0, 2, 1, 3, 4, 5) \
    #     .reshape(c1*c2, 4, w0, h0)
    # left = left.reshape(c1, 2, c2, 2).permute(0, 2, 1, 3).reshape(c1*c2, 4)
    # bottom = bottom.reshape(c1, 2, c2, 2).permute(0, 2, 1, 3).reshape(c1*c2, 4)

    # Get composite marginals as well as new left and right
    # print("Basic to composite")
    # torch.set_printoptions(profile="full")
    muYCell, left, bottom = basic_to_composite_CUDA_2D(
        muY_basic, left, bottom, basic_shape, partition)
    info["time_bounding_box"] = time.perf_counter() - t0
    # torch.set_printoptions(profile="default")  # reset

    # Get subMuY
    subMuY = crop_measure_to_box(muYCell, left, bottom, muY)
    # 2. Get bounding box dimensions
    w, h = muYCell.shape[1:]
    info["bounding_box"] = (w, h)

    # 3: get physical coordinates of bounding box for each batched problem
    posYCell = get_grid_cartesian_coordinates(
        left, bottom, w, h, dys
    )

    # 4. Solve problem

    t0 = time.perf_counter()
    # print(muXCell.shape, muYCell.shape, posXCell[0].shape, posYCell[0].shape)
    resultAlpha, resultBeta, muY_basic, info_solver = \
        BatchSolveOnCell_CUDA(  # TODO: solve balancing problems in BatchSolveOnCell_CUDA
            muXCell, muYCell, posXCell, posYCell, eps, alphaCell, subMuY,
            SinkhornError, SinkhornErrorRel, SinkhornMaxIter=SinkhornMaxIter,
            SinkhornInnerIter=SinkhornInnerIter
        )

    # Renormalize muY_basic
    # Here muY_basic is still in form (ncomp, C, *geom_shape)
    muY_basic *= (muYCell / (muY_basic.sum(dim=1) + 1e-40))[:, None, :, :]
    info["time_sinkhorn"] = time.perf_counter() - t0

    # NOTE: balancing needs muY_basic in this precise shape. But for outputting
    # we still need to permute

    # Extract

    # # 5. Turn back to numpy
    # muY_basic = muY_basic.cpu().numpy()

    # # 6. Balance. Plain DomDec code works here
    # t0 = time.perf_counter()
    # if balance:
    #     batch_balance(muXCell, muY_basic)
    # info["time_balance"] = time.perf_counter() - t0

    # 5. CUDA balance
    t0 = time.perf_counter()
    if balance:
        CUDA_balance(muXCell, muY_basic)
    info["time_balance"] = time.perf_counter() - t0

    # 7. Truncate
    t0 = time.perf_counter()
    # TODO: if too slow or too much memory turn to dedicated cuda function
    muY_basic[muY_basic <= 1e-15] = 0.0
    info["time_truncation"] = time.perf_counter() - t0

    # Extend left and right to basic cells
    if partition == "B":
        b1, b2 = b1+2, b2+2
    c1, c2 = b1//2, b2//2
    # Permute muY_basic
    _, _, w, h = muY_basic.shape
    muY_basic = muY_basic.view(c1, c2, 2, 2, w, h).permute(
        0, 2, 1, 3, 4, 5).reshape(-1, w, h)
    basic_expander = torch.ones((1, 2, 1, 2), **torch_options_int)
    left = left.reshape(c1, 1, c2, 1) * basic_expander

    left = left.ravel()
    bottom = bottom.reshape(c1, 1, c2, 1) * basic_expander
    bottom = bottom.ravel()
    if partition == "B":
        # trim muY_basic, left and bottom
        basic_indices_true = torch.arange(b1*b2,
                                          device=muY_basic.device, dtype=torch.int64
                                          ).view(b1, b2)[1:-1, 1:-1].ravel()
        muY_basic = muY_basic[basic_indices_true]
        left = left[basic_indices_true]
        bottom = bottom[basic_indices_true]

    # resultMuYAtomicDataList = []
    # # The batched version always computes directly the cell marginals
    # for i in range(BatchSize):
    #     resultMuYAtomicDataList.append([np.array(pi[i,j]) for j in range(pi.shape[1])])
    info = {**info, **info_solver}
    return resultAlpha, resultBeta, muY_basic, left, bottom, info


def slide_marginals_to_corner(muY_basic, global_left, global_bottom):
    # Get smallest possible bounding box
    B = muY_basic.shape[0]
    torch_options_int = dict(device=muY_basic.device, dtype=torch.int)
    sum_indices = torch.arange(B, **torch_options_int).view(-1, 1)
    muY_basic_slide, new_global_left, new_global_bottom = combine_cells(
        muY_basic, global_left, global_bottom, sum_indices
    )
    return muY_basic_slide, new_global_left, new_global_bottom


def crop_measure_to_box(rho_composite, global_left, global_bottom, rho):
    """
    Get the reference measure rho in the same support as rho_composite
    """

    torch_options = dict(device=rho_composite.device,
                         dtype=rho_composite.dtype)
    torch_options_int = dict(device=rho_composite.device, dtype=torch.int32)

    B, w, h = rho_composite.shape
    mask = rho_composite > 0
    sum_indices_comp = torch.arange(B, **torch_options_int).view(-1, 1)

    _, relative_left, comp_width, _, _ = \
        get_axis_bounds(rho_composite, mask, global_left, 0, sum_indices_comp)

    _, relative_bottom, comp_height, _, _ = \
        get_axis_bounds(rho_composite, mask,
                        global_bottom, 1, sum_indices_comp)

    # Reshape indices for AddWithOffsets
    global_left = global_left.view(-1, 1) + relative_left
    global_bottom = global_bottom.view(-1, 1) + relative_bottom
    comp_width = comp_width.view(-1, 1)
    comp_height = comp_height.view(-1, 1)

    # relative_left = relative_bottom = torch.zeros((B, 1), **torch_options_int)
    sum_indices_rho = torch.zeros((B, 1), **torch_options_int)
    weights = torch.ones((B, 1), **torch_options)

    reference_rho = LogSinkhornGPU.backend.AddWithOffsetsCUDA_2D(
        rho.view(1, *rho.shape), w, h,
        weights, sum_indices_rho,
        relative_left, global_left, comp_width,
        relative_bottom, global_bottom, comp_height
    )

    return reference_rho


def refine_marginals_CUDA(muY_basic, global_left, global_bottom,
                          basic_mass_coarse, basic_mass_fine, nu_coarse, nu_fine):

    # Slide marginals to the corner
    muY_basic, global_left, global_bottom = slide_marginals_to_corner(
        muY_basic, global_left, global_bottom)

    # Y marginals
    # Get refinement weights for each Y point
    s1, s2 = nu_coarse.shape
    refinement_weights_Y = nu_fine.view(s1, 2, s2, 2).permute(1, 3, 0, 2) \
        .reshape(-1, s1, s2) / nu_coarse[None, :, :]

    B = muY_basic.shape[0]
    C = refinement_weights_Y.shape[0]
    torch_options_int = dict(device=muY_basic.device, dtype=torch.int)
    torch_options = dict(device=muY_basic.device, dtype=muY_basic.dtype)

    # Get axes bounds
    mask = muY_basic > 0
    sum_indices_basic = torch.arange(B, **torch_options_int).view(-1, 1)

    relative_basic_left, basic_left, basic_width, \
        global_composite_left, w = \
        get_axis_bounds(muY_basic, mask, global_left, 0, sum_indices_basic)

    relative_basic_bottom, basic_bottom, basic_height, \
        global_composite_bottom, h = \
        get_axis_bounds(muY_basic, mask, global_bottom, 1, sum_indices_basic)

    # Indices for refinement mask

    template_indices = torch.arange(C, **torch_options_int)
    sum_indices_refine = torch.tile(template_indices, (B,)).view(-1, 1)
    basic_left_refine = torch.repeat_interleave(global_left, C).view(-1, 1)
    basic_bottom_refine = torch.repeat_interleave(global_bottom, C).view(-1, 1)
    basic_width_refine = torch.repeat_interleave(
        basic_width.ravel(), C).view(-1, 1)
    basic_height_refine = torch.repeat_interleave(
        basic_height.ravel(), C).view(-1, 1)
    relative_basic_left_refine = relative_basic_bottom_refine = torch.zeros(
        (B*C, 1), **torch_options_int)

    weights = torch.ones((B*C, 1), **torch_options)

    refinement_weights_Y_box = LogSinkhornGPU.backend.AddWithOffsetsCUDA_2D(
        refinement_weights_Y, w, h,
        weights, sum_indices_refine,
        relative_basic_left_refine, basic_left_refine, basic_width_refine,
        relative_basic_bottom_refine, basic_bottom_refine, basic_height_refine
    )

    # Refine nu basic by multiplying it with the refinement weights
    muY_basic_refine_Y = refinement_weights_Y_box.view(
        B, C, w, h) * muY_basic.view(B, 1, w, h)
    muY_basic_refine_Y = muY_basic_refine_Y.view(
        B, 2, 2, w, h).permute(0, 3, 1, 4, 2).reshape(B, 2*w, 2*h)

    # Refine muX
    b1, b2 = basic_mass_coarse.shape
    refinement_weights_X = basic_mass_fine.view(b1, 2, b2, 2) \
        / basic_mass_coarse.view(b1, 1, b2, 1)
    muY_basic_refine = muY_basic_refine_Y.view(b1, 1, b2, 1, 2*w, 2*h) \
        * refinement_weights_X.view(b1, 2, b2, 2, 1, 1)
    muY_basic_refine = muY_basic_refine.view(4*B, 2*w, 2*h)

    # Refine left and bottom
    expand = torch.ones((1, 2, 1, 2), **torch_options_int)
    global_left_refine = 2*global_left.view(b1, 1, b2, 1) * expand
    global_left_refine = global_left_refine.view(-1)
    global_bottom_refine = 2*global_bottom.view(b1, 1, b2, 1) * expand
    global_bottom_refine = global_bottom_refine.view(-1)

    return muY_basic_refine, global_left_refine, global_bottom_refine


def get_current_Y_marginal(muY_basic, global_left, global_bottom, shapeY):
    B = muY_basic.shape[0]
    torch_options_int = dict(device=muY_basic.device, dtype=torch.int)
    torch_options = dict(device=muY_basic.device, dtype=muY_basic.dtype)
    sum_indices_basic = torch.arange(B, **torch_options_int).view(-1, 1)

    # Get axis bounds
    mask = muY_basic > 0
    _, basic_left, basic_width, global_left, _ = \
        get_axis_bounds(muY_basic, mask, global_left, 0, sum_indices_basic)

    _, basic_bottom, basic_height, global_bottom, _ = \
        get_axis_bounds(muY_basic, mask, global_bottom, 1, sum_indices_basic)

    sum_indices_global = torch.arange(B, **torch_options_int).view(1, -1)
    weights = torch.ones((1, B), **torch_options)

    muY_sum = LogSinkhornGPU.backend.AddWithOffsetsCUDA_2D(
        muY_basic, *shapeY,
        weights, sum_indices_global,
        global_left.view(1, -1), basic_left.view(1, -
                                                 1), basic_width.view(1, -1),
        global_bottom.view(1, -1), basic_bottom.view(1, -
                                                     1), basic_height.view(1, -1)
    )

    return muY_sum.squeeze()


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


############################################
# Minibatches
############################################

def MiniBatchIterate(
    muY, posY, dxs_dys, eps,
    muXJ, posXJ, alphaJ,
    muY_basic, left, bottom, shapeY, partition,
    SinkhornError=1E-4, SinkhornErrorRel=False, SinkhornMaxIter=None,
    SinkhornInnerIter=100, batchsize=np.inf, clustering=False, N_clusters="smart"
):
    # partition is now of shape (B, C)
    # partition[i,:] are the basic cells in composite cell i
    # there can be -1's to encode for smaller cells
    # Clustering if one wants to apply a cluster algorithm for the minibatching

    torch_options = dict(device=muY_basic.device, dtype=muY_basic.dtype)

    

    # Compute minibatches
    # N_problems = partition.shape[0]
    # if batchsize > N_problems:
    #     batchsize = N_problems

    # N_batches = int(np.ceil(N_problems / batchsize))
    # # Get uniform minibatches of size maybe smaller than batchsize
    # actual_batchsize = int(np.ceil(N_problems / N_batches))
    # if clustering:
    #     minibatches = get_minibatches_clustering(muY_basic, left, bottom,
    #                                              partition, N_batches)
    t0 = time.perf_counter()
    N_problems = partition.shape[0]
    if clustering:
        if N_clusters == "smart":
            # N_clusters = int(min(10, max(1, np.sqrt(N_problems)/32))) # N = 1024 -> 4 clusters
            N_clusters = int(min(10, max(1, np.sqrt(N_problems)/16))) # N = 1024 -> 8 clusters
            print(f"N_clusters = {N_clusters}")
        else:
            N_clusters = min(N_clusters, N_problems)
        minibatches = get_minibatches_clustering(muY_basic, left, bottom,
                                                 partition, N_clusters)
    else:
        if batchsize == np.inf:
            batchsize = N_problems
        N_batches = int(np.ceil(N_problems / batchsize))
        # Get uniform minibatches of size maybe smaller than batchsize
        actual_batchsize = int(np.ceil(N_problems / N_batches))
        minibatches = [
            torch.arange(i*actual_batchsize,
                         min((i+1)*actual_batchsize, N_problems),
                         device=muY_basic.device, dtype=torch.int64)
            for i in range(N_batches)
        ]
        # print([len(batch) for batch in minibatches])
    time_clustering = time.perf_counter() - t0
    N_batches = len(minibatches)  # If some cluster was empty it was removed
    batch_muY_basic_list = []
    info = None
    dims_batch = np.zeros((N_batches, 2), dtype=np.int64)
    # print(N_batches, minibatches)
    for (i, batch) in enumerate(minibatches):
        posXJ_batch = tuple(xi[batch] for xi in posXJ)
        alpha_batch, basic_idx_batch, muY_basic_batch, left_batch,\
            bottom_batch, info_batch = BatchDomDecIterationMinibatch_CUDA(
                SinkhornError, SinkhornErrorRel, muY, posY, dxs_dys, eps, shapeY,
                muXJ[batch], posXJ_batch, alphaJ[batch],
                muY_basic, left, bottom, partition[batch],
                SinkhornMaxIter, SinkhornInnerIter
            )
        if info is None:
            info = info_batch
            info["solver"] = [info["solver"]]
        else:
            for key in info_batch.keys():
                if key[:4] == "time":
                    info[key] += info_batch[key]
            info["solver"].append(info_batch["solver"])
        # Slide marginals to corner to get the smallest bbox later
        t0 = time.perf_counter()
        muY_basic_batch, left_batch, bottom_batch = \
            slide_marginals_to_corner(
                muY_basic_batch, left_batch, bottom_batch)
        info["time_bounding_box"] += time.perf_counter() - t0

        # Write results that are easy to overwrite
        # But do not modify previous tensors
        alphaJ, left, bottom = alphaJ.clone(), left.clone(), bottom.clone()
        alphaJ[batch] = alpha_batch
        left[basic_idx_batch] = left_batch
        bottom[basic_idx_batch] = bottom_batch
        dims_batch[i, :] = muY_basic_batch.shape[1:]

        # Save basic cell marginals for the end
        batch_muY_basic_list.append((basic_idx_batch, muY_basic_batch))
    info["time_clustering"] = time_clustering
    # Combine all batches together
    # Get bounding box

    # Joint problems
    t0 = time.perf_counter()
    w, h = np.max(dims_batch, axis=0)
    # print("dims_batch\n", dims_batch)
    # Continue here
    B = muY_basic.shape[0]
    muY_basic = torch.zeros(B, w, h, **torch_options)
    for (basic_idx, muY_batch), box in zip(batch_muY_basic_list, dims_batch):
        w_i, h_i = box
        # print(basic_idx)
        muY_basic[basic_idx, :w_i, :h_i] = muY_batch
    info["time_join_clusters"] = time.perf_counter() - t0
    return alphaJ, muY_basic, left, bottom, info


def BatchDomDecIterationMinibatch_CUDA(
        SinkhornError, SinkhornErrorRel, muY, posYCell, dxs_dys, eps, shapeY,
        # partitionDataCompCellIndices,
        muXCell, posXCell, alphaCell,
        muY_basic, left, bottom, partition,
        SinkhornMaxIter, SinkhornInnerIter, balance=True):

    info = dict()
    # 1: compute composite cell marginals
    # Get basic shape size
    dxs, dys = dxs_dys

    t0 = time.perf_counter()
    _, w0, h0 = muY_basic.shape
    torch_options = dict(device=muY_basic.device, dtype=muY_basic.dtype)
    torch_options_int = dict(device=muY_basic.device, dtype=torch.int32)
    # muY_basic = muY_basic.reshape(c1, 2, c2, 2, w0, h0).permute(0, 2, 1, 3, 4, 5) \
    #     .reshape(c1*c2, 4, w0, h0)
    # left = left.reshape(c1, 2, c2, 2).permute(0, 2, 1, 3).reshape(c1*c2, 4)
    # bottom = bottom.reshape(c1, 2, c2, 2).permute(0, 2, 1, 3).reshape(c1*c2, 4)

    # Get composite marginals as well as new left and right
    # print("Basic to composite")
    # torch.set_printoptions(profile="full")
    muYCell, left_batch, bottom_batch = basic_to_composite_minibatch_CUDA_2D(
        muY_basic, left, bottom, partition)
    # torch.set_printoptions(profile="default")  # reset

    # Get subMuY
    subMuY = crop_measure_to_box(muYCell, left_batch, bottom_batch, muY)
    # 2. Get bounding box dimensions
    w, h = muYCell.shape[1:]
    info["bounding_box"] = (w, h)

    # 3: get physical coordinates of bounding box for each batched problem
    posYCell = get_grid_cartesian_coordinates(
        left_batch, bottom_batch, w, h, dys
    )
    info["time_bounding_box"] = time.perf_counter() - t0

    # 4. Solve problem

    t0 = time.perf_counter()
    # print(muXCell.shape, muYCell.shape, posXCell[0].shape, posYCell[0].shape)
    resultAlpha, resultBeta, muY_basic, info_solver = \
        BatchSolveOnCell_CUDA(  # TODO: solve balancing problems in BatchSolveOnCell_CUDA
            muXCell, muYCell, posXCell, posYCell, eps, alphaCell, subMuY,
            SinkhornError, SinkhornErrorRel, SinkhornMaxIter=SinkhornMaxIter,
            SinkhornInnerIter=SinkhornInnerIter
        )

    # Renormalize muY_basic
    # Here muY_basic is still in form (ncomp, C, *geom_shape)
    muY_basic *= (muYCell / (muY_basic.sum(dim=1) + 1e-40))[:, None, :, :]
    info["time_sinkhorn"] = time.perf_counter() - t0

    # NOTE: balancing needs muY_basic in this precise shape. But for outputting
    # we still need to permute

    # Extract

    # # 5. Turn back to numpy
    # muY_basic = muY_basic.cpu().numpy()

    # # 6. Balance. Plain DomDec code works here
    # t0 = time.perf_counter()
    # if balance:
    #     batch_balance(muXCell, muY_basic)
    # info["time_balance"] = time.perf_counter() - t0

    # 5. CUDA balance
    t0 = time.perf_counter()
    if balance:
        CUDA_balance(muXCell, muY_basic)
    info["time_balance"] = time.perf_counter() - t0

    # 7. Truncate
    t0 = time.perf_counter()
    # TODO: if too slow or too much memory turn to dedicated cuda function
    muY_basic[muY_basic <= 1e-15] = 0.0
    info["time_truncation"] = time.perf_counter() - t0

    # Extend left_batch and right to basic cells
    t0 = time.perf_counter()
    B, C, w, h = muY_basic.shape
    muY_basic = muY_basic.view(B*C, w, h)
    # Copy left and bottom for beta
    left_beta, bottom_beta = left_batch.clone(), bottom_batch.clone()
    basic_expander = torch.ones((1, C), **torch_options_int)
    left_batch = (left_batch.reshape(B, 1) * basic_expander).ravel()
    bottom_batch = (bottom_batch.reshape(B, 1) * basic_expander).ravel()
    # Get mask with real basic cells
    part_ravel = partition.ravel()
    # print(part_ravel.dtype)
    mask = part_ravel >= 0
    # Transform so that it can be index
    basic_indices = part_ravel[mask].long()
    muY_basic = muY_basic[mask]
    left_batch = left_batch[mask]
    bottom_batch = bottom_batch[mask]
    info["time_bounding_box"] += time.perf_counter() - t0

    info = {**info, **info_solver}
    return resultAlpha, basic_indices, muY_basic, left_batch, bottom_batch, info


def basic_to_composite_minibatch_CUDA_2D(muY_basic, left, bottom, partition):
    # muY_basic is of shape (B, s1, ..., sd)
    # TODO: change signature left,bottom to allow for higher dimensional
    B, C = partition.shape
    sum_indices = partition.clone()
    # There may be -1's in the first position, which `combine_cells` doesn't like
    # To avoid that we set them to whatever the max is in that slice and
    # set the weight to zero
    max_slices = sum_indices.amax(-1).view(-1, 1)
    max_slices = max_slices.repeat((1, C))
    mask = sum_indices < 0
    sum_indices[mask] = max_slices[mask]

    weights = torch.ones(sum_indices.shape, device=muY_basic.device,
                         dtype=muY_basic.dtype)
    weights[mask] = 0.0
    return combine_cells(muY_basic, left, bottom, sum_indices, weights)


########################################
# Clustering problems based on bbox
#########################################

# Adapted from a keops tutorial
def KMeans(x, K=10, Niter=20, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = x.view(N, 1, D)  # (N, 1, D) samples
    c_j = c.view(1, K, D)  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for _ in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) squared distances
        cl = D_ij.argmin(dim=1).view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    return cl, c


# TODO: make all of this more modular. This basically copies get_axis_bounds
# and takes just what is needed
def get_axis_composite_extent(muY_basic, mask, global_minus, axis, sum_indices):
    B = muY_basic.shape[0]
    geom_shape = muY_basic.shape[1:]
    n = geom_shape[axis]
    # Put in the position of every point with mass its index along axis
    index_axis = torch.arange(n, device=muY_basic.device, dtype=torch.int32)
    new_shape_index = [
        n if i == 1 + axis else 1 for i in range(len(muY_basic.shape))
    ]
    index_axis = index_axis.view(new_shape_index)
    mask_index = mask*index_axis
    # Get positive extreme
    basic_plus = mask_index.view(B, -1).amax(-1)
    # Turn zeros to upper bound so that we can get the minimum
    mask_index[~mask] = n
    basic_minus = mask_index.view(B, -1).amin(-1)
    # Add global offsets
    global_basic_minus = global_minus + basic_minus
    global_basic_plus = global_minus + basic_plus

    # Remove -1's in sum_indices
    sum_indices_clean = sum_indices.clone().long()
    idx, idy = torch.where(sum_indices_clean < 0)
    # Each composite cell comes at least from one basic
    sum_indices_clean[idx, idy] = sum_indices_clean[idx, 0]

    # Reduce to composite cell
    global_composite_minus = global_basic_minus[sum_indices_clean].amin(-1)
    global_composite_plus = global_basic_plus[sum_indices_clean].amax(-1)
    composite_extent = global_composite_plus - global_composite_minus + 1
    # print("axis =", axis, "composite_extent =\n", composite_extent)

    return composite_extent


def get_minibatches_clustering(muY_basic, global_left, global_bottom,
                               partition, N_problems):

    # Remove -1's
    B, C = partition.shape
    sum_indices = partition.clone()
    mask_ind = sum_indices < 0
    max_slices = sum_indices.amax(-1).view(-1, 1)
    max_slices = max_slices.repeat((1, C))
    sum_indices[mask_ind] = max_slices[mask_ind]

    # Get extents
    mask = muY_basic > 0.0

    x = get_axis_composite_extent(muY_basic, mask, global_left, 0, sum_indices)

    y = get_axis_composite_extent(muY_basic, mask, global_bottom,
                                  1, sum_indices)

    z = torch.concat((x.view(-1, 1), y.view(-1, 1)), dim=1).double()
    z += torch.rand((B, 2), dtype=torch.float64, device=x.device)
    # Cluster
    cl, _ = KMeans(z, N_problems)
    minibatches = [torch.where(cl == i)[0] for i in range(N_problems)]
    # There exist the possibility that some cluster is empty. Then remove
    minibatches = [batch for batch in minibatches if len(batch) > 0]

    return minibatches
