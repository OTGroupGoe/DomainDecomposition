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
    muY_basic_box, shapeY, partition,
    SinkhornError=1E-4, SinkhornErrorRel=True, SinkhornMaxIter=None,
    SinkhornInnerIter=100, batchsize=np.inf, clustering=False, 
    N_clusters="smart", **kwargs
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
    # TODO: we probably need a different clustering strategy for unbalanced
    # for example, jump over one cell
    # if clustering:
    #     if N_clusters == "smart":
    #         # N_clusters = int(min(10, max(1, np.sqrt(N_problems)/32))) # N = 1024 -> 4 clusters
    #         N_clusters = int(min(10, max(1, np.sqrt(N_problems)/16))) # N = 1024 -> 8 clusters
    #         print(f"N_clusters = {N_clusters}")
    #     else:
    #         N_clusters = min(N_clusters, N_problems)
    #     minibatches = get_minibatches_clustering(muY_basic_box,
    #                                              partition, N_clusters)
    # else:
    #     if batchsize == np.inf:
    #         batchsize = N_problems
    #     N_batches = int(np.ceil(N_problems / batchsize))
    #     # Get uniform minibatches of size maybe smaller than batchsize
    #     actual_batchsize = int(np.ceil(N_problems / N_batches))
    #     minibatches = [
    #         torch.arange(i*actual_batchsize,
    #                      min((i+1)*actual_batchsize, N_problems),
    #                      device=torch_options["device"], dtype=torch.int64)
    #         for i in range(N_batches)
    #     ]
    # TODO: this only works for square problems
    nc1 = nc2 = int(np.sqrt(N_problems))
    comp_ind = torch.arange(nc1*nc2, device = "cuda", dtype = torch.int64
                            ).reshape(nc1, nc2)
    minibatches = [
        comp_ind[0::2,0::2].ravel().contiguous(),
        comp_ind[0::2,1::2].ravel().contiguous(),
        comp_ind[1::2,0::2].ravel().contiguous(),
        comp_ind[1::2,1::2].ravel().contiguous()
    ]

    # Reverse
    # minibatches.reverse()
    
    # Remove empty minibatches
    minibatches = [batch for batch in minibatches if len(batch) > 0]
    time_clustering = time.perf_counter() - t0
    N_batches = len(minibatches)  # If some cluster was empty it was removed

    # # Compute current global Y marginal
    # PYpi = get_current_Y_marginal(muY_basic_box, shapeY)
    # Compute global Y marginal for each minibatch
    PYpi_minibatches = []
    # TODO: this might be slow but let's see how it performs for now
    for batch in minibatches:
        indices = partition[batch].ravel()
        # Remove minus ones
        indices = indices[indices >= 0]
        data_batch = muY_basic_box.data[indices]
        offsets_batch = muY_basic_box.offsets[indices]
        muY_basic_batch = BoundingBox(data_batch, offsets_batch, shapeY)
        batch_Y_marginal = get_current_Y_marginal(muY_basic_batch, shapeY)
        PYpi_minibatches.append(batch_Y_marginal)
    
    PYpi = PYpi_minibatches[0].clone()
    for PYpi_partial in PYpi_minibatches[1:]:
        PYpi += PYpi_partial
    PYpi_original = PYpi.clone()
    if "plot" not in kwargs.keys(): kwargs["plot"] = False
    if kwargs["plot"]:
        fig, axs = plt.subplots(figsize = (15,3), ncols = 5)
        axs[0].imshow(PYpi.cpu().T, origin = "lower")
        for i, _ in enumerate(PYpi_minibatches):
            axs[i+1].imshow(PYpi_minibatches[i].cpu().T, origin = "lower")
        plt.suptitle("Old PYpi")
        plt.show()
    
    #axs[0].axis("off")

    # Save PXpi
    PXpiJ = torch.zeros_like(alphaJ)
    dPYpi_list = []

    # Prepare for minibatch iterations
    new_offsets = torch.zeros_like(muY_basic_box.offsets)
    batch_muY_basic_list = []
    info = None
    dims_batch = np.zeros((N_batches, 2), dtype=np.int64)
    for (i, batch) in enumerate(minibatches):
        posXJ_batch = tuple(xi[batch] for xi in posXJ)
        alpha_batch, basic_idx_batch, muY_basic_box_batch, info_batch = \
            MiniBatchDomDecIterationUnbalanced_CUDA(
                SinkhornError, SinkhornErrorRel, muY, PYpi, posY, 
                dxs_dys, eps, lam, shapeY,
                muXJ[batch], posXJ_batch, alphaJ[batch],
                muY_basic_box, partition[batch],
                SinkhornMaxIter, SinkhornInnerIter,
                balance = kwargs["balance"]
            )
        # Get PXpiJ before info is messed with
        PXpiJ[batch] = info_batch["solver"].get_actual_X_marginal()
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
        # Update PYpi
        new_PY_partial = get_current_Y_marginal(muY_basic_box_batch, shapeY)
        dPYpi = new_PY_partial - PYpi_minibatches[i]
        if kwargs["update_PYpi"]:
            PYpi += dPYpi

        dPYpi_list.append(dPYpi.cpu())

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

    if kwargs["plot"]:

        fig, axs = plt.subplots(figsize = (15,3), ncols = 5)

        dPYpi = (PYpi - PYpi_original).cpu()
        cmax = torch.max(torch.abs(dPYpi))
        clim = (-cmax, cmax)
        axs[0].imshow(dPYpi.T, origin = "lower", clim = clim, cmap = "RdBu")
        plt.suptitle("Change to PYpi, " + f"clim = {np.prod(PYpi.shape)*cmax:.4f}")
        for i, dPY in enumerate(dPYpi_list):
            axs[i+1].imshow(dPY.cpu().T, origin = "lower", clim = clim, cmap = "RdBu")
        plt.show()
    
    # Prepare combined bounding box
    # TODO: here we have to evaluate the score of previous and new solution, 
    # and combine them so that they produce an appropriate decrement.
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

    return alphaJ, muY_basic_box, info

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

    # Trace positive entries
    # Get mask where muYCell zero
    # mask = muYCell_box.data == 0
    # muYref_box[mask] = 0
    # PYpi_box[mask] = 0



    # 3: get physical coordinates of bounding box for each batched problem
    posYCell = get_grid_cartesian_coordinates(
        muYCell_box, dys
    )
    info["time_bounding_box"] = time.perf_counter() - t0

    # 4. Solve problem
    t0 = time.perf_counter()
    # print(muXCell.shape, muYCell.shape, posXCell[0].shape, posYCell[0].shape)
    # TODO: probably here we have to pass on more information
    resultAlpha, resultBeta, muY_basic_batch, info_solver = \
        BatchSolveOnCellUnbalanced_CUDA(  
            muXCell, muYref_box, posXCell, posYCell, eps, lam, alphaCell, 
            muYref_box, muY_nJ,
            SinkhornError, SinkhornErrorRel, SinkhornMaxIter=SinkhornMaxIter,
            SinkhornInnerIter=SinkhornInnerIter
        )

    info["time_sinkhorn"] = time.perf_counter() - t0

    # NOTE: trying out crazy balancing idea

    # 5. CUDA balance
    t0 = time.perf_counter()
    if balance:
        # Get appropriate PXpi
        solver = info_solver["solver"]
        solver.update_alpha()
        PXpiCell = solver.get_actual_X_marginal()
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

    muY_basic_batch_box = BoundingBox(muY_basic_batch, offsets_batch, shapeY)

    info["time_bounding_box"] += time.perf_counter() - t0

    info = {**info, **info_solver}
    return resultAlpha, basic_indices, muY_basic_batch_box, info

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
    # TODO: clean up
    B = muXCell.shape[0]
    dim = len(posX)
    assert dim == 2, "Not implemented for dimension != 2"

    # Retrieve cellsize
    s = muXCell.shape[-1] // 2

    # Define cost for solver
    C = (posX, posY)
    # Solve problem
    
    # Max problem error as stopping criterion
    # solver = UnbalancedLinftySinkhorn(
    #     muXCell, subMuY, C, eps, lam, muY_nJ, alpha_init=alphaInit, nuref=muYref,
    #     max_error=SinkhornError, max_error_rel=SinkhornErrorRel,
    #     max_iter=SinkhornMaxIter, inner_iter=SinkhornInnerIter
    # )
    # # Enforce relative error
    # solver.max_error = SinkhornError

    solver = LogSinkhornGPU.UnbalancedPartialSinkhornCudaImageOffset(
        muXCell, subMuY, C, eps, lam, muY_nJ, alpha_init=alphaInit, nuref=muYref,
        max_error=SinkhornError, max_error_rel=SinkhornErrorRel,
        max_iter=SinkhornMaxIter, inner_iter=SinkhornInnerIter
    )
    # Enforce relative error
    if SinkhornErrorRel:
        solver.max_error = SinkhornError*muXCell.sum()


    msg = solver.iterate_until_max_error()
    # Niter = solver.Niter
    # if Niter > 5000:
    #     # Repeat, but saving error in every iterate. 
    #     error_hist = torch.zeros((solver.Niter, B), dtype = muXCell.dtype, device = muXCell.device)
    #     primal_hist = torch.zeros_like(error_hist)
    #     dual_hist = torch.zeros_like(error_hist)
    #     PXpi_hist = torch.zeros((solver.Niter, *solver.alpha.shape), dtype = muXCell.dtype, device = muXCell.device)
    #     PYpi_hist = torch.zeros((solver.Niter, *solver.beta.shape), dtype = muXCell.dtype, device = muXCell.device)
    #     # Reset solver
    #     solver = LogSinkhornGPU.UnbalancedPartialSinkhornCudaImageOffset(
    #         muXCell, subMuY, C, eps, lam, muY_nJ, alpha_init=alphaInit, nuref=muYref,
    #         max_error=SinkhornError, max_error_rel=SinkhornErrorRel,
    #         max_iter=SinkhornMaxIter, inner_iter=SinkhornInnerIter
    #     )
    #     for i in range(Niter):
    #         new_alpha = solver.get_new_alpha()
    #         # Compute current marginal
    #         new_mu = solver.mu * torch.exp((solver.alpha - new_alpha)/solver.eps_lam)
    #         # Update beta (we get an iteration for free)
    #         solver.alpha = new_alpha
    #         # Finish this sinkhorn iter
    #         solver.update_beta()
    #         # Return L1 error for each subproblem
    #         error_hist[i] = torch.sum(torch.abs(solver.mu - new_mu), axis = (1,2))
    
    #         PXpi = solver.get_actual_X_marginal()
    #         PYpi = solver.get_actual_Y_marginal()
    #         PXpi_hist[i] = PXpi
    #         PYpi_hist[i] = PYpi

    #         primal_hist[i] = torch.sum(solver.alpha * PXpi, (1,2)) + torch.sum(solver.beta * PYpi, (1,2)) \
    #             + solver.lam*KL_batched(PXpi, solver.mu) + solver.lam*KL_batched(PYpi + solver.nu_nJ, solver.nu)
        
    #         lam = solver.lam
    #         dual_hist[i] = - lam * torch.sum((torch.exp(-solver.alpha/lam)-1)*solver.mu, (1,2)) \
    #                 - lam * torch.sum((torch.exp(-solver.beta/lam)-1)*solver.nu, (1,2))\
    #                 - torch.sum(solver.beta*solver.nu_nJ, (1,2))
    #     with open("results/scores_temp.pickle", "wb") as f:
    #         pickle.dump(dict(error = error_hist.cpu().numpy(), 
    #         primal = primal_hist.cpu().numpy(), 
    #         dual = dual_hist.cpu().numpy(), 
    #         PXpi = PXpi_hist.cpu().numpy(), 
    #         PYpi = PYpi_hist.cpu().numpy(),
    #         nu_nJ = solver.nu_nJ.cpu().numpy(), 
    #         nuref = solver.nuref.cpu().numpy(), 
    #         muref = solver.muref.cpu().numpy()),f)
    #         # Force an error
    #     assert False

    alpha = solver.alpha
    beta = solver.beta
    # Compute cell marginals directly
    # NOTE: this function seems to be returning the marginal appropriately
    muY_basic = get_cell_marginals(
        muXCell, muYref, alpha, beta, posX, posY, eps
    )

    # Wrap solver and possibly runtime info into info dictionary
    info = {
        "solver": solver,
        "msg": msg
    }
    
    # # NEW: additional sinkhorn iter to even out alpha
    # solver.update_alpha()

    alpha = solver.alpha
    beta = solver.beta

    return alpha, beta, muY_basic, info

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