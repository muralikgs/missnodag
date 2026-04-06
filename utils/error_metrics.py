import numpy as np
import networkx as nx
from itertools import combinations

def compute_shd(W_gt, W_est):
    # both W_gt & W_est should be binary matrices
    W_gt = W_gt * 1.0
    W_est = W_est * 1.0
    
    corr_edges = (W_gt == W_est) * W_gt # All the correctly identified edges
    
    W_gt -= corr_edges
    W_est -= corr_edges
    
    R = (W_est.T == W_gt) * W_gt # Reverse edges
    
    W_gt -= R
    W_est -= R.T
    
    E = W_est > W_gt # Extra edges
    M = W_est < W_gt # Missing edges

    return R.sum() + E.sum() + M.sum(), (R.sum(), E.sum(), M.sum())

def norm_shd(W_gt, W_est):
    shd, _ = compute_shd(W_gt, W_est)
    return shd / W_gt.shape[0]

def find_v_structures(G: nx.DiGraph):
    v_structures = []
    for node in G.nodes:
        parents = list(G.predecessors(node))
        for a, c in combinations(parents, 2):
            if not G.has_edge(a, c) and not G.has_edge(c, a):
                v_structures.append((a, node, c))
    return v_structures

def compute_shd_pc(W_gt, W_est):
    missing_edges, extra_edges, reverse_edges = 0, 0, 0

    # convert GT to list of edges
    graph_gt = nx.from_numpy_array(W_gt, create_using=nx.DiGraph)
    gt_edges = graph_gt.edges

    for i, j in gt_edges:
        if W_est[j, i] == 0 and W_est[i, j] == 0:
            missing_edges += 1
        elif (W_est[j, i] == 1 and W_est[i, j] == -1) or W_est[i, j] == -1 and W_est[j, i] == -1:
            W_est[i,j], W_est[j,i] = 0, 0
        elif W_est[i,j] == 1 and W_est[j,i] == -1:
            reverse_edges += 1

    extra_edges = np.abs(W_est).sum() // 2
    return missing_edges + extra_edges + reverse_edges, (reverse_edges, extra_edges, missing_edges)

def compute_shd_cpdag(g1, g2):

    dag_1 = nx.from_numpy_array(g1, create_using=nx.DiGraph)
    dag_2 = nx.from_numpy_array(g2, create_using=nx.DiGraph)

    und_edges_1 = dag_1.to_undirected().edges
    und_edges_2 = dag_2.to_undirected().edges

    extra_missing_edges = len(
        set(und_edges_1).symmetric_difference(set(und_edges_2))
    )

    v_1 = find_v_structures(dag_1)
    v_2 = find_v_structures(dag_2)

    v_structure_mismatch = len(
        set(v_1).symmetric_difference(set(v_2))
    )

    return extra_missing_edges + v_structure_mismatch, (extra_missing_edges, v_structure_mismatch)




