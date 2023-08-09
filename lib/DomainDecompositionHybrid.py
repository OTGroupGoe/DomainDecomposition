import numpy as np
from ortools.graph.python import min_cost_flow
import torch
from . import DomainDecompositionGPU as DomDecGPU
import scipy.sparse as sp
from .LogSinkhorn import LogSinkhorn as LogSinkhorn


# Build basic cell problem
def get_gamma_2D(muX_basic, basic_mass, eps_gamma, cellsize, eps_scaling = False):
    """
    Get edge plans for min cost flow
    """
    muX_basic_renorm = muX_basic / basic_mass[:,:,None]
    mu_bottom = muX_basic_renorm[:,:-1].reshape(-1,cellsize,cellsize)
    mu_top    = muX_basic_renorm[:,1:].reshape(-1,cellsize,cellsize)
    mu_left   = muX_basic_renorm[:-1,:].reshape(-1,cellsize,cellsize)
    mu_right  = muX_basic_renorm[1:,:].reshape(-1,cellsize,cellsize)


    mu_gamma = torch.concat((mu_bottom, mu_left))
    nu_gamma = torch.concat((mu_top, mu_right))

    B = mu_gamma.shape[0]
    # Get coordinates
    x_gamma = torch.arange(cellsize, dtype = torch.float32, device = mu_gamma.device).view(1,-1)
    x_gamma = torch.repeat_interleave(x_gamma, B, dim = 0) # Batched version
    xs_gamma = (x_gamma, x_gamma)
    C = (xs_gamma, xs_gamma)

    solver_gamma = DomDecGPU.LogSinkhornCudaImageOffset(mu_gamma, nu_gamma, C, eps_gamma)
    if eps_scaling:
        # TODO: finish eps scaling
        assert False, "Not implemented yet"
    else:
        status = solver_gamma.iterate_until_max_error()
    print("status gamma: ", status)
    gamma = solver_gamma.get_dense_plan()
    return gamma

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


def get_edge_costs_2D(solver_domdec, basic_shape, cellsize, muX_basic, basic_mass, gamma):
    # Recover paramteres
    B = solver_domdec.alpha.shape[0]
    Ms = solver_domdec.alpha.shape[1:]
    Ns = solver_domdec.beta.shape[1:]
    C,X,Y = solver_domdec.get_dense_cost()
    pi = solver_domdec.get_dense_plan(C=C)
    dx = solver_domdec.get_dx()
    dim = X.shape[-1]

    # Get pi, cost, offsets
    # pi
    sb1, sb2 = basic_shape # TODO: needed?
    pi = pi.view(-1, *Ms, *Ns)
    pi_basic = DomDecGPU.convert_to_basic_2D(pi, basic_shape, cellsize)
    # Flatten X and Y dimensions to make easier to handle
    pi_basic = pi_basic.reshape(*basic_shape, cellsize**2, -1)

    # Reshape cost (same as pi)
    C = C.view(-1, *Ms, *Ns)
    C_basic = DomDecGPU.convert_to_basic_2D(C, basic_shape, cellsize)
    # Flatten X and Y dimensions to make easier to handle
    C_basic = C_basic.reshape(*basic_shape, cellsize**2,-1)

    Cpi_basic_renorm = (C_basic*pi_basic).sum(dim = (2,3)) / basic_mass

    # Compute offsets
    dx_offset = cellsize*dx

    offset = 2*(X.view(B, -1, 1, dim) - Y.view(B, 1, -1, dim))*dx_offset
    offset = offset.view(B, *Ms, *Ns, dim)
    offset_basic = DomDecGPU.convert_to_basic_2D(offset, basic_shape, cellsize)
    # Flatten X and Y
    offset_basic = offset_basic.reshape(*basic_shape, cellsize**2, -1, dim)

    # Start computing edges
    N_edges_v = sb1*(sb2-1)
    N_edges_h = (sb1-1)*sb2
    gamma_up = gamma[:N_edges_v].reshape(sb1, sb2-1, cellsize*cellsize, cellsize*cellsize).contiguous()
    gamma_right = gamma[N_edges_v:].reshape(sb1-1, sb2, cellsize*cellsize, cellsize*cellsize).contiguous()
    c_bar_list = []
    pi_i_disint = pi_basic / muX_basic[:,:,:,None]

    # Upward edges
    pi_ij_hat_renorm = torch.einsum('abik,abil->abkl',gamma_up, pi_i_disint[:,:-1]) # each p_ij_hat has mass 1
    C_offset = C_basic[:,:-1] + offset_basic[:,:-1,...,1] + dx_offset**2
    c_bar_ij = (C_offset*pi_ij_hat_renorm).sum(dim = (2,3))  # cost per unit mass in cell j
    c_bar_list.append((c_bar_ij-Cpi_basic_renorm[:,:-1]).ravel())
    
    # Right-ward edges
    pi_ij_hat_renorm = torch.einsum('abik,abil->abkl',gamma_right, pi_i_disint[:-1,:]) # each p_ij_hat has mass 1
    C_offset = C_basic[:-1,:] + offset_basic[:-1,:,...,0] + dx_offset**2
    c_bar_ij = (C_offset*pi_ij_hat_renorm).sum(dim = (2,3))  # cost per unit mass in cell j
    c_bar_list.append((c_bar_ij-Cpi_basic_renorm[:-1,:]).ravel())

    # Downward edges
    pi_ij_hat_renorm = torch.einsum('abki,abil->abkl',gamma_up, pi_i_disint[:,1:]) # lazy transpose gamma
    C_offset = C_basic[:,1:] - offset_basic[:,1:,...,1] + dx_offset**2
    c_bar_ij = (C_offset*pi_ij_hat_renorm).sum(dim = (2,3))  # cost per unit mass in cell j
    c_bar_list.append((c_bar_ij-Cpi_basic_renorm[:,1:]).ravel())

    # Left-ward edges
    pi_ij_hat_renorm = torch.einsum('abki,abil->abkl',gamma_right, pi_i_disint[1:,:]) # lazy transpose gamma
    C_offset = C_basic[1:,:] - offset_basic[1:,:,...,0] + dx_offset**2
    c_bar_ij = (C_offset*pi_ij_hat_renorm).sum(dim = (2,3))  # cost per unit mass in cell j
    c_bar_list.append((c_bar_ij-Cpi_basic_renorm[1:,:]).ravel())

    # Concatenate and return
    c_bar = np.hstack([c_bar_i.cpu().numpy() for c_bar_i in c_bar_list])
    return c_bar

def solve_grid_flow_problem(size, cap_nodes, cost_edges):
    n1, n2 = size
    N = n1*n2
    
    nodes = np.arange(0,N).reshape(n1, n2)

    # capacity edges
    cap_start = np.arange(0, N)
    cap_end = cap_start + N

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
    
    # capacities
    # TODO: do we need to turn to ints?
    res_cap = 2**15 # resolution for capacities
    cap = np.zeros(len(start_nodes), dtype = np.int64)
    max_cap = np.max(cap_nodes)
    cap[:N] = (res_cap * (cap_nodes/max_cap)).astype(np.int64)
    cap[N:] = res_cap
    
    # cost
    res_cost = 2**15
    cost = np.zeros_like(cap)
    max_cost = np.max(np.abs(cost_edges))
    cost[N:] = (res_cost * (cost_edges / max_cost)).astype(np.int64)

    # build solver
    solver = min_cost_flow.SimpleMinCostFlow()
    
    # Add arcs, capacities and costs in bulk using numpy.
    all_arcs = solver.add_arcs_with_capacity_and_unit_cost(
        start_nodes, end_nodes, cap, cost)
    
    # solve problem
    status = solver.solve()
    flows = solver.flows(all_arcs)
    nodes_engaged = flows[:N]
    w = flows[N:].astype(np.float64) * (max_cap / res_cap)
    return w

    
def implement_flow(muYAtomicDataList, muYAtomicIndicesList, edges, w, capacities_nodes, basic_shape):
    # TODO: generalize to 3D
    sb1, sb2 = basic_shape
    I = edges[:,0]
    J = edges[:,1]

    # For this iteration
    n_nodes = np.prod(basic_shape)
    incidence_T = sp.csr_matrix((w, (J, I)), shape = (n_nodes, n_nodes)) # incidence_T[j,i] is the flow from i to j
    mass_out = np.array(incidence_T.sum(axis = 1)).ravel() # outflow from every cell
    basic_cell_index = np.arange((sb1+2)*(sb2+2)).reshape((sb1+2, sb2+2))[1:-1,1:-1].ravel() # consider padding
    
    # These have no padding for simplicity
    new_muYAtomicDataList = []
    new_muYAtomicIndicesList = []

    for j, jpad in enumerate(basic_cell_index):
        arrayAdder=LogSinkhorn.TSparseArrayAdder()
        # Add remaining mass
        # TODO: truncate?
        rem_fraction = (capacities_nodes[j] - mass_out[j])/capacities_nodes[j]
        arrayAdder.add(rem_fraction * muYAtomicDataList[jpad], muYAtomicIndicesList[jpad])
        # Add inflows
        for i in incidence_T[j].indices:
            fraction = incidence_T[j,i]/capacities_nodes[i]
            ipad = basic_cell_index[i]
            arrayAdder.add(fraction * muYAtomicDataList[ipad],  muYAtomicIndicesList[ipad])
        new_muYData, new_muYIndices = arrayAdder.getDataTuple()
        # Save new marginal
        new_muYAtomicDataList.append(new_muYData)
        new_muYAtomicIndicesList.append(new_muYIndices)

    # copy to original position
    for i, ipad in enumerate(basic_cell_index):
        muYAtomicDataList[ipad] = new_muYAtomicDataList[i]
        muYAtomicIndicesList[ipad] = new_muYAtomicIndicesList[i]
    
    return muYAtomicDataList, muYAtomicIndicesList

def flow_update(muYAtomicDataList, muYAtomicIndicesList, last_solver, 
                muX_basic, basic_mass, gamma, basic_shape, 
                capacities_nodes, edges, cellsize):
    # Get induced edge costs
    edge_costs = get_edge_costs_2D(last_solver, basic_shape, cellsize, muX_basic, basic_mass, gamma)
    # Solve min cost flow problem
    w = solve_grid_flow_problem(basic_shape, capacities_nodes, edge_costs)
    # Implement flow
    implement_flow(muYAtomicDataList, muYAtomicIndicesList, edges, w, capacities_nodes, basic_shape)
    return w



