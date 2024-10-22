import numpy as np

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

