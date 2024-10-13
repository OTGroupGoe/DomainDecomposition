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

class BoundingBox:
    """
    Holds sparse vectors in bounding box representation. 

    Attributes
    ----------

    dim : int
        Ambient dimension.
    B : int
        Batch dimension; number of structres in the bounding box.
    shape : dim-tuple(int)
        Shape of bounding box.
    global_shape : dim-tuple(int)
        Shape of global marginal.
    data : torch.Tensor of size (B, *shape)
        Holds data for bounding box representation.
    offsets : torch.Tensor of size (B, dim)
        Holds index offset of respective box.
    """
    # TODO: so that BoundingBox is self-contained, it should also somehow
    # encapsulate the coordinates of Y. We leave this for the future.
    def __init__(self, data, offsets, global_shape):
        self.data = data
        self.B = data.shape[0]
        self.box_shape = data.shape[1:]
        self.dim = len(self.box_shape)
        self.offsets = offsets
        self.global_shape = global_shape
        self.options = dict(device = data.device, dtype = data.dtype)
        self.options_int = dict(device = offsets.device, dtype = offsets.dtype)

        # Check consistency
        assert self.offsets.shape[0] == self.B, \
            "First dimension offsets must be B"
        assert self.offsets.shape[1] == self.dim, \
            "First dimension offsets must be dim"
        # assert torch.all(self.offsets >= 0), "Offsets must be non-negative"
        for i in range(self.dim):
            assert torch.all(self.offsets[:,i] < self.global_shape[i]), \
                "Offset must be smaller than global shape"
            
def combine_offsets(*offsets):
    """
    Concatenates inputs into columns.
    """
    return torch.cat(tuple(x[:,None] for x in offsets), dim = 1)

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
    return b.squeeze()

def pad_extrapolate(a, padding):
    # Pad first with zeros
    assert len(a.shape) == 2, "not implemented in higher dimension"
    b = pad_tensor(a, padding)
    # Then fill the rim
    for i in range(padding):
        b[padding-i-1,:] = 2*b[padding-i,:] - b[padding-i+1,:]
        b[-padding+i,:] = 2*b[-padding+i-1,:] - b[-padding+i-2,:]
        b[:,padding-i-1] = 2*b[:,padding-i] - b[:,padding-i+1]
        b[:,-padding+i] = 2*b[:,-padding+i-1] - b[:,-padding+i-2]
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
    Reshape marginals from usual to a rectangle, find the biggest of all of these and 
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
    # return Nu_box, Nuref_box, left, bottom, max_width, max_height
    # Turn stuff to torch
    left = torch.tensor(left)
    right = torch.tensor(right)
    offsets = combine_offsets(left, right)
    Nu_bounding_box = BoundingBox(torch.tensor(Nu_box), offsets, shapeY)
    Nuref_bounding_box = BoundingBox(torch.tensor(Nuref_box), offsets, shapeY)
    return Nu_bounding_box, Nuref_bounding_box

def unpack_cell_marginals_2D(muY_basic, threshold=1e-15):
    """
    Un-batch all the cell marginals from bounding box structure and truncate 
    entries below `threshold`.
    """
    # TODO: generalize to 3D
    # muY_basic is of size (B, 4, w, h), because there are 4 basic cells per
    # composite cell
    _, n = muY_basic.global_shape
    # Do all the process in the cpu
    B = muY_basic.B
    left = muY_basic.offsets[:,0]
    bottom = muY_basic.offsets[:,1]
    marg_indices = []
    marg_data = []
    for k in range(B):
        # idx, idy = np.nonzero(muY_basic[k, i])
        idx, idy = np.nonzero(muY_basic[k] > threshold)
        linear_id = (idx+left[k])*n + idy + bottom[k]
        marg_indices.append(linear_id)
        marg_data.append(muY_basic[k, idx, idy])
    return marg_indices, marg_data

def get_grid_cartesian_coordinates(muYCell, dys):
    """
    Generate the cartesian coordinates of the boxes within a bounding box 
    and spacing dys.
    """
    # TODO: generalize to 3D
    # TODO: use just muYCell.points for this
    if muYCell.dim != 2:
        raise NotImplementedError("Only implemented for dim = 2")
    B = muYCell.B
    left, bottom = muYCell.offsets[:,0], muYCell.offsets[:,1]
    w, h = muYCell.box_shape
    torch_options = (muYCell.options)

    dy1, dy2 = dys
    y1_template = torch.arange(w, **torch_options) * dy1
    y2_template = torch.arange(h, **torch_options) * dy2
    y1 = left.view(-1, 1)*dy1 + y1_template.view(1, -1)
    y2 = bottom.view(-1, 1)*dy2 + y2_template.view(1, -1)
    return y1, y2

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

def get_cell_marginals(muref, nuref, alpha, beta, xs, ys, eps, s = None):
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

    if s is None:
        # Deduce cellsize
        s = Ms[0]//2
    # Get number of basic cells
    b1, b2 = Ms[0]//s, Ms[1]//s
    dim = len(xs)
    assert dim == 2, "Not implemented for dimension rather than 2"
    n_basic = b1*b2  # number of basic cells

    # Perform permutations and reshapes in X data to turn them
    # into B*n_cells problems of size (s,s)
    alpha_b = alpha.view(-1, b1, s, b2, s) \
        .permute((0, 1, 3, 2, 4)).reshape(-1, s, s)
    mu_b = muref.view(-1, b1, s, b2, s) \
        .permute((0, 1, 3, 2, 4)).reshape(-1, s, s)
    new_Ms = (s, s)
    logmu_b = LogSinkhornGPU.log_dens(mu_b)

    x1, x2 = xs
    x1_b = torch.repeat_interleave(x1.view(-1, b1, s, 1), b2, dim=3)
    x1_b = x1_b.permute((0, 1, 3, 2)).reshape(-1, s)
    x2_b = torch.repeat_interleave(x2.view(-1, 1, b2, s), b1, dim=1)
    x2_b = x2_b.reshape(-1, s)
    xs_b = (x1_b, x2_b)

    # Duplicate Y data to match X data
    y1, y2 = ys
    y1_b = torch.repeat_interleave(y1, n_basic, dim=0)
    y2_b = torch.repeat_interleave(y2, n_basic, dim=0)
    xs_b = (x1_b, x2_b)
    ys_b = (y1_b, y2_b)

    # Perform a reduction to get a second dual for each basic cell
    dxs = torch.tensor(np.array([get_dx(xi, xi.shape[0]) for xi in xs_b]))
    dys = torch.tensor(np.array([get_dx(yj, yj.shape[0]) for yj in ys_b]))

    offsetX, offsetY, offset_const = LogSinkhornGPU.compute_offsets_sinkhorn_grid(
        xs_b, ys_b, eps)
    h = alpha_b / eps + logmu_b + offsetX

    # beta_hat = - eps * (
    #     LogSinkhornGPU.softmin_cuda_image(h, Ns, new_Ms, eps, dys, dxs)
    #     + offsetY + offset_const
    # )

    # # Build cell marginals
    # muY_basic = nuref[:, None] * torch.exp(
    #     (beta[:, None] - beta_hat.view(-1, n_basic, *Ns))/eps
    # )
    # Memory friendly implementation

    beta_hat = LogSinkhornGPU.softmin_cuda_image(h, Ns, new_Ms, eps, dys, dxs)
    beta_hat += (offsetY + offset_const)
    beta_hat = beta_hat.view(-1, n_basic, *Ns)
    beta_hat += beta[:,None]/eps 
    muY_basic = beta_hat
    torch.exp(beta_hat, out = muY_basic)
    muY_basic *= nuref[:,None]

    return muY_basic

def BatchSolveOnCell_CUDA(
    muXCell, muYCell, posX, posY, eps, alphaInit, muYref,
    SinkhornError=1E-4, SinkhornErrorRel=False, YThresh=1E-14, verbose=True,
    SinkhornMaxIter=10000, SinkhornInnerIter=10
):
    """
    Solve cell problems. Return optimal potentials and new basic cell 
    marginals.
    """

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

def get_alpha_field_gpu(alpha, shape, cellsize):
    """
    Turn the composite cell potentials `alpha`, composed of many potentials of 
    size (2*cellsize, 2*cellsize), into a global potential of shape `shape`. 
    This new potential may feature big jumps between composite cells; these 
    will be smoothed out by `get_alpha_field_even_gpu`.
    """
    # TODO: generalize to 3D
    assert len(shape) == 2, "Not implemented for dimension rather than 2"
    comp_shape = tuple(s // (2*cellsize) for s in shape)
    alpha_field = alpha.view(*comp_shape, 2*cellsize, 2*cellsize) \
                       .permute(0, 2, 1, 3).contiguous().view(shape)
    return alpha_field


def get_alpha_field_even_gpu(alphaA, alphaB, shapeXL, shapeXL_pad,
                             cellsize, basic_shape, muX=None):
    """
    Uses alphaA, alphaB and getAlphaGraph to compute a global dual potential.
    """
    # TODO: generalize to 3D
    dim = len(alphaA.shape)-1
    assert dim == 2, "Not implemented for dimension rather than 2"
    # Glue alpha batched cell potentials to form two global potentials.
    alphaA_field = get_alpha_field_gpu(alphaA, shapeXL, cellsize)
    alphaB_field = get_alpha_field_gpu(alphaB, shapeXL_pad, cellsize)
    
    # Remove padding
    alphaB_field = alphaB_field[cellsize:-cellsize, cellsize:-cellsize]
    # Compute vertical differences
    alphaDiff = (alphaA_field-alphaB_field).cpu().numpy().ravel()
    # Solve helmholtz problem
    alphaGraph = DomDec.getAlphaGraph(
        alphaDiff, basic_shape, cellsize, muX
    )

    # Each offset in alphaGraph is for one of the batched problems in alphaA
    alphaGraphGPU = torch.tensor(
        alphaGraph, device=alphaA.device, dtype=alphaA.dtype
    )
    # Correct composite cell potentials, each with one offset
    alphaAEven = alphaA - alphaGraphGPU.view(-1, *np.ones(dim, dtype=np.int32))
    
    # Turn corrected cell potentials into global potential
    alphaFieldEven = get_alpha_field_gpu(alphaAEven, shapeXL, cellsize)

    return alphaFieldEven

def CUDA_balance(muXCell, muY_basic):
    """
    Transfer mass between basic cells inside a composite cell until basic 
    masses are correct. CUDA implementation.
    """
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

###############################################
# transform basic cell utilities
###############################################

def get_axis_bounds(muY_basic, global_minus, axis, sum_indices):
    """
    Get relative extents of the bounding box that would result from combining
    the basic cells in muY_basic according to sum_indices.

    Arguments
    ---------
    muY_basic : torch.Tensor
        Data
    global_minus : torch.Tensor(int)
        offsets along axis `axis`
    axis : int
        axis along which the extents are to be computed
    sum_indices : torch.Tensor
        Each row contains the indices of muY_basic that are to be aggregated.
    """
    B = muY_basic.shape[0]
    geom_shape = muY_basic.shape[1:]
    n = geom_shape[axis]
    # Put in the position of every point with mass its index along axis
    index_axis = torch.arange(n, device=muY_basic.device, dtype=torch.int32)
    axis_sum = 2 if axis == 0 else 1 # TODO: generalize for higher dim
    mask = muY_basic.sum(axis_sum) > 0
    mask_index = mask * index_axis.view(1, -1) # mask_index has shape (B, -1)
    # Get positive extreme
    basic_plus = mask_index.amax(-1)
    # Turn zeros to upper bound so that we can get the minimum
    mask_index[~mask] = n
    basic_minus = mask_index.amin(-1)
    basic_extent = basic_plus - basic_minus + 1
    # Add global offsets
    global_basic_minus = global_minus + basic_minus
    global_basic_plus = global_minus + basic_plus

    # NOTE: profiling shows that most time in this function is spent in the 
    # torch.where call below. However, when we try different implementations to
    # obtain the same (i.e., an array `sum_indices` without -1's), we are not 
    # able to remove the overhead. 
    # Remove -1's in sum_indices
    sum_indices_clean = sum_indices.clone().long()
    idx, idy = torch.where(sum_indices_clean < 0)
    # Each composite cell comes at least from one basic
    sum_indices_clean[idx, idy] = sum_indices_clean[idx, 0]

    # Reduce to composite cell
    global_composite_minus = global_basic_minus[sum_indices_clean].amin(-1)
    global_composite_plus = global_basic_plus[sum_indices_clean].amax(-1)
    composite_extent = global_composite_plus - global_composite_minus + 1

    # Turn basic_minus and basic_extent into shape of sum_indices
    basic_minus = basic_minus[sum_indices_clean]
    basic_extent = basic_extent[sum_indices_clean]

    # Get dim of bounding box
    max_composite_extent = torch.max(composite_extent).item()
    relative_basic_minus = global_basic_minus[sum_indices_clean] - \
        global_composite_minus.view(-1, 1)
    return (relative_basic_minus, basic_minus, basic_extent,
            global_composite_minus, max_composite_extent)


def combine_cells(muY_basic_box, sum_indices, weights=1):
    """
    Combine basic cells in muY_basic, taking into account their global boundaries 
    to yield Nu_comp. Nu_comp[j] is the result of combining all the cells with 
    indices in sum_indices[j], each multiplied by its corresponding weight in
    weights[j].
    """
    muY_basic = muY_basic_box.data
    shapeY = muY_basic_box.global_shape
    global_left = muY_basic_box.offsets[:,0]
    global_bottom = muY_basic_box.offsets[:,1]

    if type(weights) in [int, float]:
        torch_options = muY_basic_box.options
        weights = torch.full(sum_indices.shape, weights, **torch_options)

    # Get bounding box parameters
    relative_basic_left, basic_left, basic_width, \
        global_composite_left, composite_width = \
        get_axis_bounds(muY_basic, global_left, 0, sum_indices)

    relative_basic_bottom, basic_bottom, basic_height, \
        global_composite_bottom, composite_height = \
        get_axis_bounds(muY_basic, global_bottom, 1, sum_indices)

    # Previous version
    # Nu_comp = LogSinkhornGPU.backend.AddWithOffsetsCUDA_2D(
    #     muY_basic, composite_width, composite_height,
    #     weights, sum_indices,
    #     relative_basic_left, basic_left, basic_width,
    #     relative_basic_bottom, basic_bottom, basic_height
    # )

    # offsets_comp = combine_offsets(
    #     global_composite_left, global_composite_bottom)
    # muYCell_box = BoundingBox(Nu_comp, offsets_comp, shapeY)

    # New version, output side

    offsets_comp = combine_offsets(
        global_composite_left, global_composite_bottom)
    
    Nu_comp = LogSinkhornGPU.backend.AddWithOffsetsCUDA_2D_OutputSide(
        muY_basic, composite_width, composite_height,
        weights, sum_indices,
        offsets_comp, muY_basic_box.offsets
    )
    # Nu_comp = LogSinkhornGPU.backend.AddWithOffsetsCUDA_2D(
    #     muY_basic, composite_width, composite_height,
    #     weights, sum_indices,
    #     relative_basic_left, basic_left, basic_width,
    #     relative_basic_bottom, basic_bottom, basic_height
    # )

    # print(Nu_comp.shape, shapeY, offsets_comp)

    muYCell_box = BoundingBox(Nu_comp, offsets_comp, shapeY)

    return muYCell_box

def slide_marginals_to_corner(muY_basic_box):
    """
    Get smallest possible bounding box by sliding cell marginals to the 
    bottom-left corner and trimming excess space.
    """
    
    B = muY_basic_box.B
    torch_options_int = muY_basic_box.options_int
    sum_indices = torch.arange(B, **torch_options_int).view(-1, 1)
    return combine_cells(muY_basic_box, sum_indices)


def crop_measure_to_box(rho_composite_box, rho):
    """
    Get the reference measure rho in the same support as rho_composite
    """
    # TODO: implement for 3D
    assert rho_composite_box.dim == 2, "only implemented for 2D"

    torch_options = rho_composite_box.options
    torch_options_int = rho_composite_box.options_int
    rho_composite = rho_composite_box.data
    global_left = rho_composite_box.offsets[:,0].clone()
    global_bottom = rho_composite_box.offsets[:,1].clone()

    torch_options = dict(device=rho_composite.device,
                         dtype=rho_composite.dtype)
    torch_options_int = dict(device=rho_composite.device, dtype=torch.int32)

    B, w, h = rho_composite.shape
    # mask = rho_composite > 0
    sum_indices_comp = torch.arange(B, **torch_options_int).view(-1, 1)

    _, relative_left, comp_width, _, _ = \
        get_axis_bounds(rho_composite, global_left, 0, sum_indices_comp)

    _, relative_bottom, comp_height, _, _ = \
        get_axis_bounds(rho_composite,
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

def refine_marginals_CUDA(muY_basic_box, basic_mass_coarse, basic_mass_fine, 
                          nu_coarse, nu_fine):
    """
    Refine the cell Y-marginals given in the bounding box structure 
    `muY_basic_box` to match the new, fine X and Y marginals.
    """

    # Slide marginals to the corner
    muY_basic_box = slide_marginals_to_corner(muY_basic_box)

    # Y marginals
    # Get refinement weights for each Y point
    s1, s2 = nu_coarse.shape
    refinement_weights_Y = nu_fine.view(s1, 2, s2, 2).permute(1, 3, 0, 2) \
        .reshape(-1, s1, s2) / nu_coarse[None, :, :]

    B = muY_basic_box.B
    C = refinement_weights_Y.shape[0]
    torch_options_int = muY_basic_box.options_int
    torch_options = muY_basic_box.options

    # Get axes bounds
    muY_basic = muY_basic_box.data
    # mask = muY_basic > 0
    global_left = muY_basic_box.offsets[:,0]
    global_bottom = muY_basic_box.offsets[:,1]
    sum_indices_basic = torch.arange(B, **torch_options_int).view(-1, 1)

    relative_basic_left, basic_left, basic_width, \
        global_composite_left, w = \
        get_axis_bounds(muY_basic, global_left, 0, sum_indices_basic)

    relative_basic_bottom, basic_bottom, basic_height, \
        global_composite_bottom, h = \
        get_axis_bounds(muY_basic, global_bottom, 1, sum_indices_basic)

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
    muY_basic_refine_Y = refinement_weights_Y_box.view(B, C, w, h) 
    muY_basic_refine_Y *= muY_basic.view(B, 1, w, h)
    muY_basic_refine_Y = muY_basic_refine_Y.view(
        B, 2, 2, w, h).permute(0, 3, 1, 4, 2).reshape(B, 2*w, 2*h)
    
    # Refine muX
    b1, b2 = basic_mass_coarse.shape
    refinement_weights_X = basic_mass_fine.view(b1, 2, b2, 2) \
        / basic_mass_coarse.view(b1, 1, b2, 1)
    print(muY_basic_refine_Y.shape, refinement_weights_X.shape)
    muY_basic_refine = muY_basic_refine_Y.view(b1, 1, b2, 1, 2*w, 2*h) \
        * refinement_weights_X.view(b1, 2, b2, 2, 1, 1)
    muY_basic_refine = muY_basic_refine.view(4*B, 2*w, 2*h)

    # Refine left and bottom
    expand = torch.ones((1, 2, 1, 2), **torch_options_int)
    global_left_refine = 2*global_left.view(b1, 1, b2, 1) * expand
    global_left_refine = global_left_refine.view(-1)
    global_bottom_refine = 2*global_bottom.view(b1, 1, b2, 1) * expand
    global_bottom_refine = global_bottom_refine.view(-1)

    # Compose new bounding box
    offsets = combine_offsets(global_left_refine, global_bottom_refine)
    shapeY = nu_fine.shape

    # print("shapeY", shapeY, "new offsets", offsets)
    muY_basic_refine_box = BoundingBox(muY_basic_refine, offsets, shapeY)

    return muY_basic_refine_box

def get_current_Y_marginal(muY_basic_box, shapeY, batchshape = None):
    """
    Aggregate cell Y marginals to obtain actual global Y marginal.
    """
    B = muY_basic_box.B
    muY_basic = muY_basic_box.data
    torch_options_int = muY_basic_box.options_int
    torch_options = muY_basic_box.options
    sum_indices_basic = torch.arange(B, **torch_options_int).view(-1, 1)


    sum_indices_global = torch.arange(B, **torch_options_int).view(1, -1)
    weights = torch.ones((1, B), **torch_options)

    # Old AddWithOffsetsCuda
    # global_left = muY_basic_box.offsets[:,0]
    # global_bottom = muY_basic_box.offsets[:,1]

    # # Get axis bounds
    # mask = muY_basic > 0
    # _, basic_left, basic_width, global_left, _ = \
    #     get_axis_bounds(muY_basic, mask, global_left, 0, sum_indices_basic)

    # _, basic_bottom, basic_height, global_bottom, _ = \
    #     get_axis_bounds(muY_basic, mask, global_bottom, 1, sum_indices_basic)
    # muY_sum = LogSinkhornGPU.backend.AddWithOffsetsCUDA_2D(
    #     muY_basic, *shapeY,
    #     weights, sum_indices_global,
    #     global_left.view(1, -1), basic_left.view(1, -
    #                                              1), basic_width.view(1, -1),
    #     global_bottom.view(1, -1), basic_bottom.view(1, -
    #                                                  1), basic_height.view(1, -1)
    # )

    # offsets_comp = torch.zeros((1, 2), **torch_options_int)
    # muY_sum = LogSinkhornGPU.backend.AddWithOffsetsCUDA_2D_OutputSide(
    #     muY_basic, *shapeY, weights, sum_indices_global, 
    #     offsets_comp, muY_basic_box.offsets
    # )

    # Try multiscale approach
    s = 8
    if batchshape is None: 
        b1 = b2 = int(np.sqrt(B)) // s
    else: 
        b1, b2 = batchshape[0]//s, batchshape[1]//s
    if b1 == 0 or b2 == 0: # skip coarse step
        s = 1
        muY_box_coarse = muY_basic_box
    else:
        # Combine fine to coarse 8x8 clusters
        sum_indices_coarse = torch.arange(B, **torch_options_int).reshape(b1, s, b2, s) \
                    .permute(0, 2, 1, 3).reshape(-1, s*s)
        muY_box_coarse = combine_cells(muY_basic_box, sum_indices_coarse)
    # Combine coarse to global
    sum_indices_global = torch.arange(B//(s*s), **torch_options_int).view(1, -1)
    weights = torch.ones((1, B//(s*s)), **torch_options)
    offsets_comp = torch.zeros((1, 2), **torch_options_int)
    muY_sum = LogSinkhornGPU.backend.AddWithOffsetsCUDA_2D_OutputSide(
        muY_box_coarse.data, *shapeY, weights, sum_indices_global, 
        offsets_comp, muY_box_coarse.offsets
    )

    return muY_sum.squeeze()

def get_multiscale_layers(muX, shapeX):
    """
    Get multiscale layers of tensor muX. Currently only works if shape is 
    power of 2.
    """
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
    muY_basic_box, shapeY, partition,
    SinkhornError=1E-4, SinkhornErrorRel=False, SinkhornMaxIter=None,
    SinkhornInnerIter=100, batchsize=np.inf, clustering=False, N_clusters="smart",
    balance = True
):
    """
    Perform a domain decomposition iteration on the composite cells given by 
    partition.

    It divides the partition into smaller minibatches, each of which is 
    processed as a chunk of data on the GPU. The strategy to produce the 
    minibatches can be clustering them according to the problem size (if 
    `clustering` is `True`), or just making chunks of size `batchsize` (if it 
    this parameter is smaller than `np.inf`). `N_clusters` controls the number
    of clusters; it can also be set to "smart"; which adapts it to the 
    resolution.
    """

    torch_options = muY_basic_box.options

    t0 = time.perf_counter()
    N_problems = partition.shape[0]
    B = muY_basic_box.B
    if clustering:
        if N_clusters == "smart":
            N_clusters = int(min(10, max(1, np.sqrt(N_problems)/32))) # N = 1024 -> 4 clusters
            # N_clusters = int(min(10, max(1, np.sqrt(N_problems)/16))) # N = 1024 -> 8 clusters
            print(f"N_clusters = {N_clusters}")
        else:
            N_clusters = min(N_clusters, N_problems)
        minibatches = get_minibatches_clustering(muY_basic_box,
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
                         device=torch_options["device"], dtype=torch.int64)
            for i in range(N_batches)
        ]
    time_clustering = time.perf_counter() - t0
    N_batches = len(minibatches)  # If some cluster was empty it was removed

    # Prepare for minibatch iterations
    new_offsets = torch.zeros_like(muY_basic_box.offsets)
    batch_muY_basic_list = []
    info = None
    dims_batch = np.zeros((N_batches, 2), dtype=np.int64)
    for (i, batch) in enumerate(minibatches):
        posXJ_batch = tuple(xi[batch] for xi in posXJ)
        alpha_batch, basic_idx_batch, muY_basic_box_batch, info_batch = \
            MiniBatchDomDecIteration_CUDA(
                SinkhornError, SinkhornErrorRel, muY, posY, dxs_dys, eps, shapeY,
                muXJ[batch], posXJ_batch, alphaJ[batch],
                muY_basic_box, partition[batch],
                SinkhornMaxIter, SinkhornInnerIter, 
                balance = balance
            )
        if info is None:
            info = info_batch
            info["solver"] = [info["solver"]]
            info["bounding_box"] =[info["bounding_box"]]
        else:
            for key in info_batch.keys():
                if key[:4] == "time":
                    info[key] += info_batch[key]
            info["solver"].append(info_batch["solver"])
            info["bounding_box"].append(info_batch["bounding_box"])

        # Slide marginals to corner to get the smallest bbox later
        t0 = time.perf_counter()
        muY_basic_box_batch = slide_marginals_to_corner(muY_basic_box_batch)
        info["time_bounding_box"] += time.perf_counter() - t0

        # Write results that are easy to overwrite
        # But do not modify previous tensors
        alphaJ[batch] = alpha_batch
        new_offsets[basic_idx_batch] = muY_basic_box_batch.offsets
        dims_batch[i, :] = muY_basic_box_batch.box_shape

        # Save basic cell marginals for combining them at the end
        batch_muY_basic_list.append((basic_idx_batch,muY_basic_box_batch.data))
    info["time_clustering"] = time_clustering
    
    # Prepare combined bounding box
    t0 = time.perf_counter()
    w, h = np.max(dims_batch, axis=0)
    muY_basic = torch.zeros(B, w, h, **torch_options)
    for (basic_idx, muY_batch), box in zip(batch_muY_basic_list, dims_batch):
        w_i, h_i = box
        muY_basic[basic_idx, :w_i, :h_i] = muY_batch
    info["time_join_clusters"] = time.perf_counter() - t0
    # Create bounding box
    muY_basic_box = BoundingBox(muY_basic, new_offsets, shapeY)
    return alphaJ, muY_basic_box, info

def MiniBatchDomDecIteration_CUDA(
        SinkhornError, SinkhornErrorRel, muY, posYCell, dxs_dys, eps, shapeY,
        # partitionDataCompCellIndices,
        muXCell, posXCell, alphaCell,
        muY_basic_box, partition,
        SinkhornMaxIter, SinkhornInnerIter, balance=True):

    """
    Performs a GPU Sinkhorn iteration on the minibatch given by `partition`.
    """
    info = dict()
    # 1: compute composite cell marginals
    # Get basic shape size
    dxs, dys = dxs_dys

    t0 = time.perf_counter()
    torch_options_int = muY_basic_box.options_int

    # Get composite marginals as well as new left and right
    muYCell_box = basic_to_composite_minibatch_CUDA_2D(
        muY_basic_box, partition)

    # Get subMuY
    subMuY = crop_measure_to_box(muYCell_box, muY)
    # 2. Get bounding box dimensions
    w, h = muYCell_box.box_shape
    info["bounding_box"] = (w, h)

    # 3: get physical coordinates of bounding box for each batched problem
    posYCell = get_grid_cartesian_coordinates(
        muYCell_box, dys
    )
    info["time_bounding_box"] = time.perf_counter() - t0

    # 4. Solve problem
    t0 = time.perf_counter()
    # print(muXCell.shape, muYCell.shape, posXCell[0].shape, posYCell[0].shape)
    resultAlpha, resultBeta, muY_basic_batch, info_solver = \
        BatchSolveOnCell_CUDA(  # TODO: solve balancing problems in BatchSolveOnCell_CUDA
            muXCell, muYCell_box.data, posXCell, posYCell, eps, alphaCell, subMuY,
            SinkhornError, SinkhornErrorRel, SinkhornMaxIter=SinkhornMaxIter,
            SinkhornInnerIter=SinkhornInnerIter
        )

    # Renormalize muY_basic_batch
    # Here muY_basic_batch is still in form (ncomp, C, *geom_shape)
    muY_basic_batch *= (muYCell_box.data / (muY_basic_batch.sum(dim=1) + 1e-40))[:, None, :, :]
    info["time_sinkhorn"] = time.perf_counter() - t0

    # NOTE: balancing needs muY_basic_batch in this precise shape. But for outputting
    # we still need to permute

    # 5. CUDA balance
    t0 = time.perf_counter()
    if balance:
        CUDA_balance(muXCell, muY_basic_batch)
    info["time_balance"] = time.perf_counter() - t0

    # 7. Truncate
    t0 = time.perf_counter()
    # TODO: if too slow or too much memory turn to dedicated cuda function
    muY_basic_batch[muY_basic_batch <= 1e-15] = 0.0
    info["time_truncation"] = time.perf_counter() - t0

    # Build bounding box for muY_basic_batch
    t0 = time.perf_counter()
    B, C, w, h = muY_basic_batch.shape
    muY_basic_batch = muY_basic_batch.view(B*C, w, h)
    # Copy left and bottom for beta
    offsets_comp = muYCell_box.offsets.reshape(B, 1, -1)
    offsets_basic = (offsets_comp.expand(-1, C, -1)).reshape(B*C, -1)
    # Get mask with real basic cells
    # Transform so that it can be index
    part_ravel = partition.ravel()
    mask = part_ravel >= 0
    basic_indices = part_ravel[mask].long()
    muY_basic_batch = muY_basic_batch[mask]
    offsets_batch = offsets_basic[mask]

    muY_basic_batch_box = BoundingBox(muY_basic_batch, offsets_batch, shapeY)

    info["time_bounding_box"] += time.perf_counter() - t0

    info = {**info, **info_solver}
    return resultAlpha, basic_indices, muY_basic_batch_box, info


def basic_to_composite_minibatch_CUDA_2D(muY_basic_box, partition):
    """
    Combines basic cells into composite cells according to the minibatch
    specified by `partition`.
    """
    # muY_basic is of shape (B, s1, ..., sd)
    B, C = partition.shape
    sum_indices = partition.clone()
    # There may be -1's in the first position, which `combine_cells` doesn't like
    # To avoid that we set them to whatever the max is in that slice and
    # set the weight to zero
    max_slices = sum_indices.amax(-1).view(-1, 1)
    max_slices = max_slices.repeat((1, C))
    mask = sum_indices < 0
    sum_indices[mask] = max_slices[mask]

    weights = torch.ones(sum_indices.shape, **muY_basic_box.options)
    weights[mask] = 0.0
    return combine_cells(muY_basic_box, sum_indices, weights)



#######################
# Global balance
# For warm-starting with Sinkhorn solution without compromising marginal
#######################
from ortools.graph.python import min_cost_flow

def get_edges_2D(basic_shape):
    sb1, sb2 = basic_shape
    indices = np.arange(sb1*sb2).reshape(sb1, sb2)
    bottom = indices[:,:-1].reshape(-1,1)
    top = indices[:,1:].reshape(-1,1)
    left = indices[:-1,:].reshape(-1,1)
    right = indices[1:,:].reshape(-1,1)
    edges = np.block([
        [bottom, top],
        [left, right],
        [top, bottom],
        [right, left]
    ])
    return edges

def solve_grid_flow_problem(size, supply):
    n1, n2 = size
    N = n1*n2
    
    nodes = np.arange(0,N).reshape(n1, n2)

    # capacity edges
    cap_start = nodes.ravel()
    cap_end = nodes.ravel() + N

    # up edges
    up_start = nodes[:,:-1].ravel() + N
    up_end = nodes[:,1:].ravel()

    # right edges
    right_start = nodes[:-1,:].ravel() + N
    right_end = nodes[1:,:].ravel()

    # down edges
    down_start = nodes[:,1:].ravel() + N
    down_end = nodes[:,:-1].ravel()


    # left edges
    left_start = nodes[1:,:].ravel() + N
    left_end = nodes[:-1,:].ravel()

    # join them
    start_nodes = np.hstack((cap_start, up_start, right_start, down_start, left_start))
    end_nodes = np.hstack((cap_end, up_end, right_end, down_end, left_end))
    
    # turn supply to ints
    res = 1000 / np.max(np.abs(supply))
    supply = (supply*res).astype(np.int64) # supply is now normalized from 0 to 1000
    # Fix numerical error so that it is a balanced problem
    inbalance = supply.sum()
    sign = (inbalance // abs(inbalance))
    if inbalance > len(supply):
        supply -= sign * (abs(inbalance) // len(supply))
    supply[:abs(inbalance)] -=sign


    # capacities
    # TODO: do we need to turn to ints?
    res_cap = 4*1000*N # resolution for capacities
    cap = np.full_like(start_nodes,res_cap, dtype = np.int64)
    
    # cost
    cost = np.ones_like(start_nodes)
    # build solver
    solver_flow = min_cost_flow.SimpleMinCostFlow()
    
    # Add arcs, capacities and costs in bulk using numpy.
    all_arcs = solver_flow.add_arcs_with_capacity_and_unit_cost(
        start_nodes, end_nodes, cap, cost)
    
    solver_flow.set_nodes_supplies(nodes.ravel(), supply)
    
    # solve problem
    status = solver_flow.solve()
    flows = solver_flow.flows(all_arcs)
    nodes_engaged = flows[:N]
    w = flows[N:].astype(np.float64) / res
    # w = flows.astype(np.float64) * (max_cap / res_cap)
    return w

def implement_flow_CUDA(Nu_basic_box, flow, basic_mass, basic_shape, PXpi_basic = None):
    # TODO: generalize for 3D
    b1, b2 = basic_shape
    B = np.prod(basic_shape)
    if PXpi_basic is None:
        PXpi_basic = basic_mass

    # If there's no flow, abort
    dim = len(basic_shape)
    # Each basic cell has at most 2*dim + 1 incoming edges: the loop and two 
    # from every direction
    Nu_basic = Nu_basic_box.data
    torch_options = dict(dtype=Nu_basic.dtype, device=Nu_basic.device)
    torch_options_int = dict(dtype=torch.int32, device=Nu_basic.device)
    basic_index = torch.arange(B, **torch_options_int)
    sum_indices = torch.zeros(B, 2*dim+1, **torch_options_int)
    weights = torch.zeros(B, 2*dim+1, **torch_options)
    idx = torch.div(basic_index, b2, rounding_mode = "trunc")
    idy = (basic_index % b2)

    flow = torch.tensor(flow, **torch_options)

    if flow[B:].sum() == 0:
        return Nu_basic_box
    else: 
        # Capacity edges
        sum_indices[:,0] = basic_index 
        weights[:,0] = basic_mass.ravel() 
        # At the end we will remove the outgoing mass

        cnt_w = 0 # Counter for weights
        # Edges going up: incoming from below
        sum_indices[:,1] = idx*b2 + (idy-1)
        # Cells in the bottom row have no such incoming edge
        remove_edge = idy == 0
        sum_indices[remove_edge, 1] = -1
        n_edges = b1*(b2-1)
        weights[~remove_edge, 1] = flow[cnt_w:cnt_w+n_edges]
        cnt_w = cnt_w + n_edges

        # Edges going right
        sum_indices[:,2] = (idx-1)*b2 + idy
        remove_edge = idx == 0
        sum_indices[remove_edge, 2] = -1
        n_edges = (b1-1)*b2
        weights[~remove_edge,2] = flow[cnt_w:cnt_w+n_edges]
        cnt_w = cnt_w + n_edges

        # Edges going down
        sum_indices[:,3] = idx*b2 + (idy+1)
        remove_edge = idy == b2-1
        n_edges = b1*(b2-1)
        sum_indices[remove_edge, 3] = -1
        weights[~remove_edge,3] = flow[cnt_w:cnt_w+n_edges]
        cnt_w = cnt_w + n_edges

        # Edges going left
        sum_indices[:,4] = (idx+1)*b2 + idy
        remove_edge = idx == b1-1
        sum_indices[remove_edge, 4] = -1
        n_edges = (b1-1)*b2
        weights[~remove_edge,4] = flow[cnt_w:cnt_w+n_edges]
        cnt_w = cnt_w + n_edges

        # Fill weights in 0-th column

        weights[:,0] -= torch.sum(weights[:,1:], dim = 1)

        # TODO: is basic_mass or basic_PXpi
        Nu_basic_box = BoundingBox(Nu_basic_box.data  / PXpi_basic.view(-1, 1, 1), 
                                   Nu_basic_box.offsets, Nu_basic_box.global_shape)
        Nu_basic_box = combine_cells(Nu_basic_box, sum_indices, weights)
        return Nu_basic_box

def global_balance_CUDA(Nu_basic_box, basic_mass, basic_shape):
    # Get induced edge costs
    PXpi_basic = Nu_basic_box.data.sum((1,2))
    mass_delta = basic_mass - PXpi_basic

    # Solve min cost flow problem
    flow = solve_grid_flow_problem(basic_shape, mass_delta.ravel().cpu().numpy())
    # Implement flow
    Nu_basic_box = implement_flow_CUDA(Nu_basic_box, flow, basic_mass, basic_shape, PXpi_basic = PXpi_basic)
    return flow, Nu_basic_box

def sinkhorn_to_domdec(solver, s, balance = True):
    """
    Compute domdec cell marginals, alphas and actual X marginals from a 
    global solver object and cellsize `s`
    """
    xs, ys = solver.xs, solver.ys
    eps = solver.eps
    M1, M2 = LogSinkhornGPU.geom_dims(solver.muref)
    N1, N2 = LogSinkhornGPU.geom_dims(solver.nuref)
    b1, b2 = M1//s, M2//s
    muY_basic = get_cell_marginals(solver.muref, solver.nuref, 
                                        solver.alpha, solver.beta,
                                        xs, ys, eps, s = s).squeeze()
    
    # Truncate
    muY_basic[muY_basic <= 1e-15] = 0.0
    PYpi = solver.get_actual_Y_marginal().squeeze()  # == nu for balanced domdec 
    
    # Get current X marginal by averaging duals
    alpha_new = solver.get_new_alpha()
    solver.alpha = 0.5*solver.alpha + 0.5*alpha_new
    PXpi = solver.get_actual_X_marginal().squeeze()  # == mu for balanced domdec 
    # Renormalize by total muY_basic mass
    PXpi *= PYpi.sum() / PXpi.sum()
    solver.update_beta()

    # Compute current cell-wise score
    # 1. Transport-entropic score
    alpha_score = (solver.alpha * PXpi).view(b1, s, b2, s).sum((1,3)).ravel()
    beta_score = (solver.beta * muY_basic).sum((1,2))
    transport_score = alpha_score + beta_score
    # 2. KL penalties
    margX_score = solver.lam * LogSinkhornGPU.KL(
        PXpi.view(b1, s, b2, s), solver.mu.view(b1, s, b2, s), axis = (1,3)).ravel()
    margY_score = solver.lam * LogSinkhornGPU.KL(PYpi, solver.nu)
    # Wrap in tuple
    basic_cell_score = (transport_score, margX_score, margY_score)

    # Compute atomic mass

    atomic_mass = PXpi.view(b1, s, b2, s).sum((1,3)).view(1, b1*b2)
    
    # Initialize bounding box object
    offsets = torch.zeros((len(muY_basic), 2), 
                      dtype = torch.int32, device = "cuda")
    muY_basic_box = BoundingBox(muY_basic, offsets, (N1, N2))

    # Compress
    muY_basic_box = slide_marginals_to_corner(muY_basic_box)

    # Balance to remove global X error. Specially necessary for unbalanced 
    # domdec with big lambda
    if balance:
        muY_basic_box.data = CUDA_balance(PXpi[None, :, :], muY_basic_box.data)
        # _, muY_basic_box = global_balance_CUDA(muY_basic_box, atomic_mass.ravel(), (b1, b2))

        

    return muY_basic_box, PXpi, atomic_mass, solver.alpha.squeeze(), basic_cell_score

########################################
# Clustering problems based on bbox
#########################################

# Adapted from a keops tutorial
def KMeans(x, K=10, Niter=20, verbose=True):
    """
    Implements Lloyd's algorithm for the Euclidean metric.
    """

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
def get_axis_composite_extent(muY_basic, global_minus, axis, sum_indices):
    """
    Computes the size of the composite extent along a given axis.
    """
    B = muY_basic.shape[0]
    geom_shape = muY_basic.shape[1:]
    n = geom_shape[axis]
    # Put in the position of every point with mass its index along axis
    index_axis = torch.arange(n, device=muY_basic.device, dtype=torch.int32)
    axis_sum = 2 if axis == 0 else 1 # TODO: generalize for higher dim
    mask = muY_basic.sum(axis_sum) > 0
    mask_index = mask * index_axis.view(1, -1) # mask_index has shape (B, -1)
    # Get positive extreme
    basic_plus = mask_index.amax(-1)
    # Turn zeros to upper bound so that we can get the minimum
    mask_index[~mask] = n
    basic_minus = mask_index.amin(-1)
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


def get_minibatches_clustering(muY_basic_box,
                               partition, N_problems):
    """
    Clusters the partition indices according to the size of the 
    composite problem marginal. 
    """

    # Remove -1's
    B, C = partition.shape
    sum_indices = partition.clone()
    mask_ind = sum_indices < 0
    max_slices = sum_indices.amax(-1).view(-1, 1)
    max_slices = max_slices.repeat((1, C))
    sum_indices[mask_ind] = max_slices[mask_ind]

    # Get extents
    muY_basic = muY_basic_box.data
    global_left = muY_basic_box.offsets[:,0]
    global_bottom = muY_basic_box.offsets[:,1]
    # mask = muY_basic > 0.0

    x = get_axis_composite_extent(muY_basic, global_left, 0, sum_indices)

    y = get_axis_composite_extent(muY_basic, global_bottom,
                                  1, sum_indices)

    z = torch.concat((x.view(-1, 1), y.view(-1, 1)), dim=1).double()
    z += torch.rand((B, 2), dtype=torch.float64, device=x.device)
    # Cluster
    cl, _ = KMeans(z, N_problems)
    minibatches = [torch.where(cl == i)[0] for i in range(N_problems)]
    # There exist the possibility that some cluster is empty. Then remove
    minibatches = [batch for batch in minibatches if len(batch) > 0]

    return minibatches
