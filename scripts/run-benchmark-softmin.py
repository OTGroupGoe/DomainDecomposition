import torch
from geomloss import SamplesLoss
import geomloss
import matplotlib.pyplot as plt
import time

from geomloss.utils import *
def softmin_two_grids_dense(eps, C_xy, h_y):
    D = dimension(h_y)
    # B, K, N = h_y.shape[BATCH], h_y.shape[CHANNEL], h_y.shape[WIDTH]
    B, K, *Ns = h_y.shape
    
    # if not keops_available:
    #     raise ImportError("This routine depends on the pykeops library.")

    #x = torch.arange(N).type_as(h_y) / N
    p, dx, Ms = C_xy

    # print("Ms: ", Ms, "; Ns: ", Ns)

    if p == 1: 
        blur = eps
    elif p == 2: 
        blur = np.sqrt(2 * eps)
    else:
        raise NotImplementedError()


    def softmin(a_log, axis): 
        # `a_log` is data, `axis` is dimension along which we compute the softmin
        a_log = a_log.contiguous()
        M = Ms[axis]
        N = Ns[axis]
        dx_eff = torch.tensor(dx/blur).type_as(a_log) # Effective length-scale
        a_log_j = a_log.view(-1, 1, N, 1)
        x_i = torch.arange(M).type_as(a_log) * dx_eff # Assume same spacing for input and output grids
        x_j = torch.arange(N).type_as(a_log) * dx_eff
        x_i = x_i.view(1, M, 1, 1)
        x_j = x_j.view(1, 1, N, 1)
        
        if p == 1:
            kA_log_ij = a_log_j - (x_i - x_j).abs()  # (B * Z, M, N, 1) # Z depends on which permutations were already performed
        elif p == 2:
            kA_log_ij = a_log_j - (x_i - x_j) ** 2  # (B * Z, M, N, 1)

        kA_log = kA_log_ij.logsumexp(dim=2)  # (B * Z, M, 1)
        # print("kA_log: ", kA_log.reshape(-1))
        
        # The softmin is always performed along the last axis. This is because outside of this function the dimensions are permuted.
        # The dimensions after the sofmin can be derived from the permutation pattern in the main function. 
        if D == 1:
            return kA_log.view(B, K, M)
        
        elif D == 2:
            if axis == 1:
                return kA_log.view(B, K, Ns[0], M)
            else: # axis == 0
                return kA_log.view(B, K, Ms[1], M)

        elif D == 3:
            if axis == 2:
                return kA_log.view(B, K, Ns[0], Ns[1], M)
            elif axis == 1:
                return kA_log.view(B, K, Ns[0], Ms[2], M)
            else: # axis == 0
                return kA_log.view(B, K, Ms[2], Ms[1], M)
                
            #return kA_log.view(B, K, N, N, N)

    if D == 1: 
        h_y = softmin(h_y, 0)

    elif D == 2:
        # Below written sizes of the data in h_y
        # (N0, N1)
        h_y = softmin(h_y, 1)  # Act on lines
        # (N0, M1)
        h_y = softmin(h_y.permute([0, 1, 3, 2]), 0).permute([0, 1, 3, 2])  # Act on columns
        # permutation -> (M1, N0) -> softmin -> (M1, M0) -> permutation -> (M0, M1)

    elif D == 3:
        # (N0, N1, N2)
        h_y = softmin(h_y, 2)  # Act on dim 4
        # (N0, N1, M2)
        h_y = softmin(h_y.permute([0, 1, 2, 4, 3]), 1).permute(
            [0, 1, 2, 4, 3]
        )  # Act on dim 3
        # p -> (N0, M2, N1) -> s -> (N0, M2, M1) -> p -> (N0, M1, M2)
        h_y = softmin(h_y.permute([0, 1, 4, 3, 2]), 0).permute(
            [0, 1, 4, 3, 2]
        )  # Act on dim 2
        # p -> (M2, M1, N0) -> s -> (M2, M1, M0) -> p -> (M0, M1, M2)
    # assert False
    return -eps * h_y

Ns = [32, 64, 128, 256]
Bs = [1, 10, 100, 1000]
N_and_batch = [
    (32, 1), (32, 10), (32, 100), (32, 1000),
    (64, 1), (64, 10), (64, 100), (64, 1000),
    (128,1), (128,10), (128,100), (128, 1000),
    (256,1), (256,10), (256,100), (256, 1000)
]
results = np.zeros((3, len(N_and_batch)))

for i, (N, B) in enumerate(N_and_batch):
    print("N = {}, B = {}".format(N, B))
    shapeX = (B, 1, N, N)
    a = torch.rand(shapeX).cuda()
    eps = 0.1
    dx = 0.1
    p = 2
    C_xy = (p, dx, (N, N))
    # Benchmark different methods
    res = %timeit -o geomloss.sinkhorn_images.softmin_two_grids(eps, C_xy, a)
    results[0,i] = res.average
    res = %timeit -o geomloss.sinkhorn_images.softmin_grid(eps, p, a)
    results[1,i] = res.average
    res = %timeit -o softmin_two_grids_dense(eps, C_xy, a)
    results[2,i] = res.average
    torch.cuda.empty_cache()

print(results.tolist())
