import numpy as np
import torch
# from . import MultiScaleOT
from .LogSinkhorn import LogSinkhorn as LogSinkhorn
import LogSinkhornGPU
import matplotlib.pyplot as plt

from . import DomainDecomposition as DomDec
from .DomainDecompositionGPU import *
import time
import pickle

class DomDecSinkhornKL(LogSinkhornGPU.UnbalancedPartialSinkhornCudaImageOffset):
    """
    Same as LogSinkhornGPU.UnbalancedPartialSinkhornCudaImageOffset but with 
    KL stopping criterion
    """
    def __init__(self, mu, nu, C, eps, lam, nu_nJ, **kwargs):
        super().__init__(mu, nu, C, eps, lam, nu_nJ, **kwargs)

    def get_current_error(self):
        """
        KL marginal error
        """
        PXpi = self.get_actual_X_marginal()
        PXpi_opt = torch.exp(-self.alpha / self.lam) * self.mu
        # Return KL
        return LogSinkhornGPU.KL(PXpi, PXpi_opt)

def KL_batched(a, b):
    """
    Kullback-Leibler divergence
    """
    B = LogSinkhornGPU.batch_dim(a)
    mask_apos = a != 0
    mask_b0 = b == 0
    mask_inf = (mask_b0) & (mask_apos)
    if mask_inf.sum() > 0:
        # a singular wrt b
        return torch.inf
    else:
        return (a * (LogSinkhornGPU.log_dens(a) - LogSinkhornGPU.log_dens(b)) - a + b).sum((1,2))

class UnbalancedLinftySinkhorn(LogSinkhornGPU.UnbalancedPartialSinkhornCudaImageOffset):
    def __init__(self, mu, nu, C, eps, lam, nu_nJ, **kwargs):
        super().__init__(mu, nu, C, eps, lam, nu_nJ, **kwargs)

    def get_current_error(self):
        """
        Get current error for unbalanced Sinkhorn
        """
        new_alpha = self.get_new_alpha()
        # Compute current marginal
        new_mu = self.mu * torch.exp((self.alpha - new_alpha)/self.eps_lam)
        # Update beta (we get an iteration for free)
        self.alpha = new_alpha
        # Finish this sinkhorn iter
        self.update_beta()
        # Return L1 error
        problem_masses = self.mu.sum(dim = (1,2))
        problem_errors = torch.sum(torch.abs(self.mu - new_mu), dim = (1,2))
        return torch.max(problem_errors / problem_masses)

def MiniBatchIterateUnbalanced(
    muY, posY, dxs_dys, eps, lam,
    muXJ, posXJ, alphaJ,
    muY_basic_box, shapeY, partition, current_basic_score, 
    SinkhornError=1E-4, SinkhornErrorRel=True, SinkhornMaxIter=None,
    SinkhornInnerIter=100, batchsize=np.inf, clustering=False, 
    N_clusters="smart", safeguard_threshold = 0.005, **kwargs
):
    """
    Perform a domain decomposition iteration on the composite cells given by 
    partition.

    It divides the partition into smaller batches, each of which is 
    processed as a chunk of data on the GPU. The strategy to produce the 
    batches can be clustering them according to the problem size (if 
    `clustering` is `True`), or just making chunks of size `batchsize` (if it 
    this parameter is smaller than `np.inf`). `N_clusters` controls the number
    of clusters; it can also be set to "smart"; which adapts it to the 
    resolution.
    """

    torch_options = muY_basic_box.options

    t0 = time.perf_counter()
    N_problems = partition.shape[0]
    B = muY_basic_box.B
    # NOTE: this only works for square problems
    nc1 = nc2 = int(np.sqrt(N_problems))
    comp_ind = torch.arange(nc1*nc2, device = "cuda", dtype = torch.int64
                            ).reshape(nc1, nc2)
    if N_clusters == 1:
        batches = [comp_ind]
    elif N_clusters == 4:
        batches = [
            comp_ind[0::2,0::2], comp_ind[0::2,1::2],
            comp_ind[1::2,0::2], comp_ind[1::2,1::2]
        ]
    elif N_clusters == 8:
        batches = [
            comp_ind[0::4,0::2], comp_ind[0::4,1::2],
            comp_ind[1::4,0::2], comp_ind[1::4,1::2],
            comp_ind[2::4,0::2], comp_ind[2::4,1::2],
            comp_ind[3::4,0::2], comp_ind[3::4,1::2]
        ]
    else: 
        raise NotImplementedError("Only N_clusters in [1,4,8] are implemented.")
    batches_shape = 2*batches[-1].shape[0],2*batches[-1].shape[1]
    batches = [batch.ravel().contiguous() for batch in batches]
    
    # Remove empty batches
    batches = [batch for batch in batches if len(batch) > 0]
    time_clustering = time.perf_counter() - t0
    N_batches = len(batches)  # If some cluster was empty it was removed

    # # Compute current global Y marginal
    PYpi = get_current_Y_marginal(muY_basic_box, shapeY)

    # Save PXpi
    PXpiJ = torch.zeros_like(alphaJ)

    # Prepare for minibatch iterations
    new_offsets = torch.zeros_like(muY_basic_box.offsets)
    batch_muY_basic_list = []
    info = None
    dims_batch = np.zeros((N_batches, 2), dtype=np.int64)
    time_PYpi = 0.0
    time_check_scores = 0.0
    print("batch\ttotal\ttrans\tmargX\tmargY")
    for i,batch in enumerate(batches):
        indices = partition[batch].ravel()
        # Count time towads 
        # Remove minus ones
        t0 = time.perf_counter()
        indices = indices[indices >= 0]
        data_batch = muY_basic_box.data[indices]
        offsets_batch = muY_basic_box.offsets[indices]
        muY_basic_batch = BoundingBox(data_batch, offsets_batch, shapeY)
        PYpi_batch = get_current_Y_marginal(muY_basic_batch, shapeY,
                                                  batchshape=batches_shape)
        
        time_PYpi += time.perf_counter() - t0
        # PYpi_batches.append(batch_Y_marginal)
    
        #axs[0].axis("off")

        #for (i, batch) in enumerate(batches):
        ######################################
        posXJ_batch = tuple(xi[batch] for xi in posXJ)
        alpha_batch, basic_idx_batch, new_muY_basic_batch, info_batch, batch_basic_score = \
            MiniBatchDomDecIterationUnbalanced_CUDA(
                SinkhornError, SinkhornErrorRel, muY, PYpi, posY, 
                dxs_dys, eps, lam, shapeY,
                muXJ[batch], posXJ_batch, alphaJ[batch],
                muY_basic_box, partition[batch],
                SinkhornMaxIter, SinkhornInnerIter,
                balance = kwargs["balance"]
            )
        #####################################
        # Save PXpiJ before info is messed with
        PXpiJ[batch] = info_batch["PXpiCell"]
        if info is None:
            info = info_batch
            # info["solver"] = [info["solver"]]
            info["bounding_box"] =[info["bounding_box"]]
            info["Niter"] =[info["Niter"]]
        else:
            for key in info_batch.keys():
                if key[:4] == "time":
                    info[key] += info_batch[key]
            # info["solver"].append(info_batch["solver"])
            info["bounding_box"].append(info_batch["bounding_box"])
            info["Niter"].append(info_batch["Niter"])
        ##############################################
        
        # Update PYpi

        t0 = time.perf_counter()
        new_PYpi_batch = get_current_Y_marginal(new_muY_basic_batch, shapeY,
                                                  batchshape=batches_shape)
        dPYpi = (new_PYpi_batch - PYpi_batch).contiguous()
        time_PYpi += time.perf_counter() - t0

        # Compare current with previous score
        t0 = time.perf_counter()
        transport_score, margX_score, margY_score = current_basic_score
        old_score = transport_score.sum() + margX_score.sum() + margY_score
        new_transport_score = transport_score.clone()
        new_margX_score = margX_score.clone()
        batch_transport_score, batch_margX_score = batch_basic_score
        # 1. Transport score
        new_transport_score[basic_idx_batch] = batch_transport_score
        # 2. Marginal penalty
        new_margX_score[basic_idx_batch] = batch_margX_score
        new_margY_score = lam*LogSinkhornGPU.KL(PYpi + dPYpi, muY)
        new_score = new_transport_score.sum() + new_margX_score.sum() + new_margY_score

        # print("old, new scores", old_score.round(decimals = 1).item(), new_score.round(decimals = 1).item())
        print(i,
              new_score.round(decimals = 1).item(), 
              new_transport_score.sum().round(decimals = 1).item(), 
              new_margX_score.sum().round(decimals = 1).item(),
              new_margY_score.round(decimals = 1).item(),
              sep = "\t")
        if new_score > (1 + safeguard_threshold*max(1,lam))*old_score:
            # Need to average with previous
            # theta = 1/len(batch)
            theta = 0.25
            new_muY_basic_batch = bounding_box_interpolation(
                muY_basic_batch, new_muY_basic_batch, theta)
            dPYpi *= theta
            print(f"batch {i} set to safe")
        else: 
            current_basic_score = new_transport_score, new_margX_score, new_margY_score

        time_check_scores += time.perf_counter() - t0
        # Update PYpi  
        PYpi += dPYpi

        # Slide marginals to corner to get the smallest bbox later
        t0 = time.perf_counter()
        muY_basic_box_batch = slide_marginals_to_corner(new_muY_basic_batch)
        info["time_bounding_box"] += time.perf_counter() - t0

        # Write results that are easy to overwrite
        # But do not modify previous tensors
        alphaJ[batch] = alpha_batch
        new_offsets[basic_idx_batch] = muY_basic_box_batch.offsets
        dims_batch[i, :] = muY_basic_box_batch.box_shape

        # Save basic cell marginals for combining them at the end
        batch_muY_basic_list.append((basic_idx_batch,muY_basic_box_batch.data))
    info["time_clustering"] = time_clustering
    info["time_PYpi"] = time_PYpi
    info["time_check_scores"] = time_check_scores

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
    
    # Save PXpi in info
    info["PXpiB"] = PXpiJ

    return alphaJ, muY_basic_box, info, current_basic_score

def MiniBatchDomDecIterationUnbalanced_CUDA(
        SinkhornError, SinkhornErrorRel, muY, PYpi, posYCell, dxs_dys, eps, lam,
        shapeY, muXCell, posXCell, alphaCell,
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
    
    # 2. Get bounding box dimensions
    w, h = muYCell_box.box_shape
    info["bounding_box"] = (w, h)

    # Get subMuY
    muYref_box = crop_measure_to_box(muYCell_box, muY)
    # To get nu_{-J} we just need to remove muYCell_box from PXpi
    PYpi_box = crop_measure_to_box(muYCell_box, PYpi)
    muY_nJ = PYpi_box - muYCell_box.data

    # 3: get physical coordinates of bounding box for each batched problem
    posYCell = get_grid_cartesian_coordinates(
        muYCell_box, dys
    )
    info["time_bounding_box"] = time.perf_counter() - t0

    # 4. Solve problem
    t0 = time.perf_counter()
    # print(muXCell.shape, muYCell.shape, posXCell[0].shape, posYCell[0].shape)
    resultAlpha, _, muY_basic_batch, solver = \
        BatchSolveOnCellUnbalanced_CUDA(  
            muXCell, muYref_box, posXCell, posYCell, eps, lam, alphaCell, 
            muYref_box, muY_nJ,
            SinkhornError, SinkhornErrorRel, SinkhornMaxIter=SinkhornMaxIter,
            SinkhornInnerIter=SinkhornInnerIter
        )

    info["time_sinkhorn"] = time.perf_counter() - t0
    info_solver = dict(Niter = solver.Niter)
    
    # Compute cell scores before data is transformed
    s = solver.alpha.shape[-1] // 2
    PXpiCell = solver.get_actual_X_marginal()
    batch_alpha_score = (solver.alpha * PXpiCell).view(-1, 2, s, 2, s).sum((2,4)).ravel()
    batch_margX_score = solver.lam * LogSinkhornGPU.KL(
        PXpiCell.view(-1, 2, s, 2, s), solver.mu.view(-1, 2, s, 2, s), axis = (2, 4)).ravel()
    batch_beta_score = (muY_basic_batch * solver.beta[:,None,:,:]).sum((2,3)).ravel()
    batch_transport_score = batch_alpha_score + batch_beta_score

    # NOTE: trying out crazy balancing idea
    solver.update_alpha()
    PXpiCell = solver.get_actual_X_marginal()
    solver.update_beta()
    # print(PXpiCell.shape, muY_basic_batch.shape)
    # Normalize composite cell X and Y masses to the same mass
    PXpiCell *= (muY_basic_batch.sum((-3, -2, -1)) /PXpiCell.sum((-2, -1))).reshape(-1, 1, 1)
    info_solver["PXpiCell"] = PXpiCell
    # solver.update_beta()
    # 5. CUDA balance
    t0 = time.perf_counter()
    if balance:
        # Get appropriate PXpi
        CUDA_balance(PXpiCell, muY_basic_batch)
    info["time_balance"] = time.perf_counter() - t0
    #info["time_balance"] = 0.0

    # 7. Truncate
    t0 = time.perf_counter()
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

    batch_transport_score = batch_transport_score[mask].contiguous()
    batch_margX_score = batch_margX_score[mask].contiguous()
    batch_basic_score = (batch_transport_score, batch_margX_score)

    muY_basic_batch_box = BoundingBox(muY_basic_batch, offsets_batch, shapeY)

    info["time_bounding_box"] += time.perf_counter() - t0

    info = {**info, **info_solver}
    return resultAlpha, basic_indices, muY_basic_batch_box, info, batch_basic_score

def BatchSolveOnCellUnbalanced_CUDA(
    muXCell, subMuY, posX, posY, eps, lam, alphaInit, muYref, muY_nJ,
    SinkhornError=1E-4, SinkhornErrorRel=True, YThresh=1E-14, verbose=True,
    SinkhornMaxIter=10000, SinkhornInnerIter=10
):
    """
    Solve cell problems. Return optimal potentials and new basic cell 
    marginals.
    """
    # Retrieve BatchSize
    B = muXCell.shape[0]
    dim = len(posX)
    assert dim == 2, "Not implemented for dimension != 2"

    # Retrieve cellsize
    s = muXCell.shape[-1] // 2

    # Define cost for solver
    C = (posX, posY)
    # Solve problem

    solver = DomDecSinkhornKL(
        muXCell, subMuY, C, eps, lam, muY_nJ, alpha_init=alphaInit, nuref=muYref,
        max_error=SinkhornError, max_error_rel=SinkhornErrorRel,
        max_iter=SinkhornMaxIter, inner_iter=SinkhornInnerIter,
        newton_iter = 10
    )

    msg = solver.iterate_until_max_error()

    # Enforce relative error
    if SinkhornErrorRel:
        solver.max_error = SinkhornError*muXCell.sum()

    alpha = solver.alpha
    beta = solver.beta

    # Compute cell marginals directly
    # NOTE: this function seems to be returning the marginal appropriately
    muY_basic = get_cell_marginals(
        muXCell, muYref, alpha, beta, posX, posY, eps
    )

    return alpha, beta, muY_basic, solver

def compute_primal_score(solvers, muY_basic_box, muY):
    # Get primal_score and muX_error
    shapeY = muY_basic_box.global_shape
    lam = solvers[0].lam
    primal_score = 0.0
    for solverB in solvers:
        # Get cost components corresponding to cost function and X marginal
        PXpiJ = solverB.get_actual_X_marginal()
        PYpiJ = solverB.get_actual_Y_marginal()
        primal_score += torch.sum(solverB.alpha * PXpiJ)
        primal_score += torch.sum(solverB.beta * PYpiJ)
        primal_score += lam*LogSinkhornGPU.KL(PXpiJ, solverB.mu)
        # new_alpha = solverB.get_new_alpha()
        # muX_error += torch.sum(torch.abs(solverB.mu - current_mu))
    # Add global marginal penalty
    PYpi = get_current_Y_marginal(muY_basic_box, shapeY)
    primal_score += lam*LogSinkhornGPU.KL(PYpi, muY)
    return primal_score.item()

def compute_primal_score_components(solvers, muY_basic_box, muY):
    # Get primal_score and muX_error
    shapeY = muY_basic_box.global_shape
    lam = solvers[0].lam
    transport_score = 0.0
    margX_score = 0.0
    for solverB in solvers:
        # Get cost components corresponding to cost function and X marginal
        PXpiJ = solverB.get_actual_X_marginal()
        PYpiJ = solverB.get_actual_Y_marginal()
        transport_score += torch.sum(solverB.alpha * PXpiJ)
        transport_score += torch.sum(solverB.beta * PYpiJ)
        margX_score += lam*LogSinkhornGPU.KL(PXpiJ, solverB.mu)
        # new_alpha = solverB.get_new_alpha()
        # muX_error += torch.sum(torch.abs(solverB.mu - current_mu))
    # Add global marginal penalty
    PYpi = get_current_Y_marginal(muY_basic_box, shapeY)
    
    margY_score = lam*LogSinkhornGPU.KL(PYpi, muY)
    return transport_score.item(), margX_score.item(), margY_score.item()

def bounding_box_interpolation(nu1, nu2, theta):
    assert nu1.B == nu2.B, "bounding boxes must have same batch dim"
    w1, h1 = nu1.box_shape
    w2, h2 = nu2.box_shape
    B = nu1.B
    w = max(w1, w2)
    h = max(h1, h2)
    # Copy data
    data = torch.zeros((2*B, w, h), **nu1.options)
    data[:B, :w1, :h1] = nu1.data
    data[B:, :w2, :h2] = nu2.data
    # Combine offsets
    offsets = torch.cat((nu1.offsets, nu2.offsets))
    nu_comb = BoundingBox(data, offsets, nu1.global_shape)
    # Combine cells with weighted sum: should go like:
    # [0, B]
    # [1, B+1]
    # ...
    sum_indices = torch.arange(2*B, **nu1.options_int).view(2, B).permute((1,0)).contiguous()
    weights = torch.tensor([[1-theta, theta]], **nu1.options).expand((B, -1)).contiguous()
    nu_combined = combine_cells(nu_comb, sum_indices, weights)
    return nu_combined