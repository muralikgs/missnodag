import torch
import sys
import sklearn.neighbors._base
import numpy as np
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from baselines.optimal_transport.imputers import OTimputer

from missingpy import MissForest

def mean_impute(dataset, missing):
    final_dataset = dataset 
    col_mean = np.nanmean(final_dataset, axis=0)

    inds = np.where(np.isnan(final_dataset))
    final_dataset[inds] = np.take(col_mean, inds[1])
    return final_dataset 

def missforest_impute(dataset, missing):

    tmp = dataset
    tmp[(1 - missing).astype(bool)] = float("NaN")

    imputer = MissForest()
    imp_dataset = imputer.fit_transform(tmp)

    return imp_dataset

def OT_impute(dataset, missing):

    sk_imputer = OTimputer(eps=0.01, batchsize=128, lr=1e-2, niter=2000)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tmp = dataset
    tmp[(1 - missing).astype(bool)] = float("NaN")

    torch_dataset = torch.tensor(tmp, device=device).double()

    imp_dataset = sk_imputer.fit_transform(torch_dataset, verbose=True, report_interval=500)

    return imp_dataset.detach().cpu().numpy()
    
