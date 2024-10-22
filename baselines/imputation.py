import torch
import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from baselines.optimal_transport.imputers import OTimputer

from missingpy import MissForest

def mean_impute(dataset, missing):
    mean = (dataset * missing).sum(axis=0) / missing.sum(axis=0)
    
    dataset_imputed = dataset * missing + (1 - missing) * mean

    return dataset_imputed 

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
    
