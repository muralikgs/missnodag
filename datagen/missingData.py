import numpy as np
from torch.utils.data import Dataset
import torch
from scipy import optimize
import networkx as nx

from datagen.structuralModels import linearSEM

def generate_mar_mask(X, p, p_obs):
    n, d = X.shape

    mask = np.zeros_like(X)

    d_obs = max(int(d * p_obs), 1) 
    d_na = d - d_obs 

    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    coefs = picks_coefs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_random_binary_matrix(rows, cols, max_ones_per_col):
    # Initialize the matrix with zeros
    matrix = np.zeros((rows, cols), dtype=int)

    for i in range(rows):
        # Randomly choose the number of ones for the current row (from 0 to max_ones_per_row)
        num_ones = np.random.randint(2, max_ones_per_col + 1)
        
        # Randomly choose positions to set to 1 in the current row
        if num_ones > 0:
            ones_positions = np.random.choice(cols, num_ones, replace=False)
            matrix[ones_positions, i] = 1
    
    return matrix

def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p

            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p

            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts

class mgraph:

    def __init__(
            self, 
            obs_graph: nx.DiGraph, 
            sem: linearSEM, 
            missing_model='obs-only',
            p=0.2,
            max_variance=2.0,
            is_mcar=False,
            max_child=3,
            scaling_data_given=False,
            scaling_data=None
    ):
        '''
        misssing_model - determines the structure of connections to the missingness indicators. It can be either "full" or "obs-only".
        "obs-only" - In this case, the parents of r_i are restricted to the set x_{-i}. We assume no self-consoring
        "full" - This produces a graph where we have edges from r_{i+1} -> r_i aside from the ones allowed by the setting "obs-only"
        '''
        
        self.obs_graph = obs_graph
        self.sem = sem 
        self.missing_mode = missing_model
        self.p = p
        self.is_mcar = is_mcar

        # define the missingness graph

        # weights between x_i's and r_i's
        self.m_coefs = np.random.randn(sem.n_nodes, sem.n_nodes)
        np.fill_diagonal(self.m_coefs, 0)

        # make the coefficients matrix sparse
        sparsity_mask = generate_random_binary_matrix(self.sem.n_nodes, self.sem.n_nodes, max_ones_per_col=max_child)
        self.m_coefs = self.m_coefs * sparsity_mask

        if self.missing_mode == "full":
            # define the weights between r_i and r_{i+1}
            r_coefs = np.zeros_like(self.m_coefs)
            r_coefs[np.arange(1, self.sem.n_nodes), np.arange(self.sem.n_nodes - 1)] = np.random.randn(self.sem.n_nodes - 1)

            self.m_coefs = np.vstack([self.m_coefs, r_coefs])
        
        # intercepts for the sigmoid function. The intercepts are set such that the probability
        # of a node being missing is on an average equal to the argument "p"
        self.m_intercept = np.log(self.p / (1 - self.p)) * np.ones(self.sem.n_nodes)

        # re-adjusting the parameters such that the variance of W^\top (x,r) is around `max_variance`,
        # currently ignoring the contribution from the parents of r_i that are also missingness indicators. 
        if not scaling_data_given:
            test_data = sem.generateData(n_samples=1000, intervention_set=[None])
        else:
            test_data = scaling_data
        
        wtx = test_data @ self.m_coefs[:self.sem.n_nodes, :]
        arg_var = np.var(wtx, axis=0, keepdims=True)
        self.m_coefs = ((max_variance / arg_var)**0.5) * self.m_coefs 
        
        if self.is_mcar: 
            self.m_coefs = np.zeros_like(self.m_coefs)

    def generatemDataFromSamples(self, X, intervention_set=[None]):

        if self.is_mcar:
            R = generate_mar_mask(X, self.p, p_obs=0.3)
        else:
            p_R_0 = sigmoid(
                X @ self.m_coefs + self.m_intercept
            )

            R = np.random.binomial(n=1, p=1-p_R_0, size=X.shape)
            
            # ensuring that non of the intervened upon nodes are missing
            intervention_mask = np.zeros_like(R)
            if intervention_set[0] != None:
                intervention_mask[:, intervention_set] = 1
            
            R = intervention_mask + (1 - intervention_mask) * R

        data = X.copy()
        data[R == 0] = np.nan

        return R, data

    def generateData(self, n_samples, intervention_set=[None], *args, **kwargs):
        
        X = self.sem.generateData(
            n_samples=n_samples, 
            intervention_set=intervention_set,
            *args, 
            **kwargs
        )

        R, data = self.generatemDataFromSamples(X, intervention_set=intervention_set)

        return X, R, data

class missing_value_dataset(Dataset):

    def __init__(
            self, 
            intervention_datasets, 
            intervention_targets
    ):
        '''
        intervention_datasets is a list of tuples, a tuple corresponding to each intervention,
        and each tuple has three parts shown below: 
        
        (clean data, missing mask, missing data)
        '''
        
        self.intervention_datasets = intervention_datasets 
        self.intervention_targets = intervention_targets 

        self.create_dataset()

    def create_dataset(self):
        
        inter_masks_list = list()
        data_clean_list = list()
        data_miss_list = list()
        miss_indicator_list = list()
        
        for targets, inter_dataset in zip(self.intervention_targets, self.intervention_datasets):

            intervention_mask = np.ones_like(inter_dataset[0])
            if targets[0] != None:
                intervention_mask[:, targets] = 0

            inter_masks_list.append(intervention_mask)
            data_clean_list.append(inter_dataset[0])
            data_miss_list.append(inter_dataset[2])
            miss_indicator_list.append(inter_dataset[1])
        
        self.X_clean = np.vstack(data_clean_list)
        self.X_miss = np.vstack(data_miss_list)
        self.R = np.vstack(miss_indicator_list)
        self.intervention_mask = np.vstack(inter_masks_list)

    def __len__(self):
        return len(self.X_clean)
    
    def __getitem__(self, index):

        return (
            self.X_clean[index],
            self.X_miss[index],
            self.R[index],
            self.intervention_mask[index]
        )
        

