import numpy as np
from torch.utils.data import Dataset
import torch
from scipy import optimize
import networkx as nx
import torch.nn as nn 

from datagen.graph import DirectedGraphGenerator
from datagen.structuralModels import SEM


def generate_R2R_DAG(n_nodes, expected_density=1):
    graph_gen = DirectedGraphGenerator(
        nodes=n_nodes, 
        expected_density=expected_density,
        enforce_dag=True
    )

    return graph_gen()

def make_MNAR_identifiable(r2r_adj, x2r_adj):
    # check for colluders
    colluder_mask = r2r_adj * x2r_adj 

    return r2r_adj * (1 - colluder_mask) # removes the colluders by removing edges from R->R matrix

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

class mgraph:

    def __init__(
            self, 
            obs_graph: nx.DiGraph, 
            sem: SEM, 
            missing_model='obs-only',
            p=0.2,
            max_variance=2.0,
            is_mcar=False,
            max_child=3,
            scaling_data_given=False,
            scaling_data=None,
            mlp=False
    ):
        '''
        Args:
        misssing_model - determines the structure of connections to the missingness indicators. It can be either "full" or "obs-only".
                        Choose between the following options: 
                        "obs-only" - In this case, the parents of r_i are restricted to the set x_{-i}. We assume no self-consoring
                        "full" - This produces a graph where we have edges from r_{i+1} -> r_i aside from the ones 
                                 allowed by the setting "obs-only"
                        "violate" - This produces a graph that violates the identifiability criteria for the missingness mechanism
        '''
        
        self.obs_graph = obs_graph
        self.sem = sem 
        self.missing_mode = missing_model
        self.p = p
        self.is_mcar = is_mcar
        self.mlp = mlp

        # define the missingness graph

        # weights between x_i's and r_i's
        self.m_coefs = np.random.randn(sem.n_nodes, sem.n_nodes)
        np.fill_diagonal(self.m_coefs, 0)

        # make the coefficients matrix sparse
        sparsity_mask = generate_random_binary_matrix(self.sem.n_nodes, self.sem.n_nodes, max_ones_per_col=max_child)
        self.m_coefs = self.m_coefs * sparsity_mask
        self.m_coefs_r2r = np.zeros_like(self.m_coefs)

        if self.missing_mode == "full" or self.missing_mode == "violate":
            r2r_graph = nx.to_numpy_array(generate_R2R_DAG(sem.n_nodes))
            r2r_adj = make_MNAR_identifiable(r2r_graph, 1.0*(np.abs(self.m_coefs) > 0))

            # if "violate" option is chosen, then no-colluder condition need not be satisfied
            self.m_coefs_r2r = r2r_adj if self.missing_mode == "full" else r2r_graph
        
        # intercepts for the sigmoid function. The intercepts are set such that the probability
        # of a node being missing is on an average equal to the argument "p"
        self.m_intercept = np.log(self.p / (1 - self.p)) * np.ones(self.sem.n_nodes)

        # re-adjusting the parameters such that the variance of W^\top (x,r) is around `max_variance`,
        # currently ignoring the contribution from the parents of r_i that are also missingness indicators. 

        if scaling_data_given:
            test_data = scaling_data
        else:
            test_data = sem.generateData(n_samples=1000, intervention_set=[None])
        
        wtx = test_data @ self.m_coefs[:self.sem.n_nodes, :]
        arg_var = np.var(wtx, axis=0, keepdims=True)
        self.m_coefs = ((max_variance / arg_var)**0.5) * self.m_coefs 
        
        if self.is_mcar: 
            self.m_coefs = np.zeros_like(self.m_coefs)

        if self.missing_mode == "violate":
            # incase the "violate" option is chosen, 30% of the nodes (randomly chosen) are made to be self-censored
            censored_nodes = np.random.choice(self.sem.n_nodes, size=int(0.3*self.sem.n_nodes), replace=False)
            self.m_coefs[censored_nodes, censored_nodes] = 0.3

        if self.mlp:
            self.function = nn.Sequential(
                nn.Linear(in_features=2*self.sem.n_nodes, out_features=self.sem.n_nodes),
                nn.Linear(in_features=self.sem.n_nodes, out_features=self.sem.n_nodes),
                nn.Sigmoid()
            )


    def func(self, X, R):
        
        if not self.mlp:
            p_R_0 = sigmoid(
                        X @ self.m_coefs + (1-R) @ self.m_coefs_r2r + self.m_intercept
                    )
        
        else:
            W_xr = torch.tensor(np.abs(self.m_coefs) > 0).float()
            W_rr = torch.tensor(np.abs(self.m_coefs_r2r) > 0).float()

            X, R = torch.tensor(X).float(), torch.tensor(R).float()
            p_R_0 = torch.zeros_like(R)
            for i in range(self.sem.n_nodes):
                par_x = torch.diag(W_xr[:, i])
                par_r = torch.diag(W_rr[:, i])
                
                xr_concat = torch.cat([X @ par_x, R @ par_r], dim=1)
                p_R_0_i = self.function(xr_concat)[:, i]

                p_R_0[:, i] = p_R_0_i.squeeze()
            
            p_R_0 = p_R_0.detach().numpy()

        return p_R_0

    def generatemDataFromSamples(self, X, intervention_set=[None]):

        if self.is_mcar:
            R = generate_mar_mask(X, self.p, p_obs=0.3)
        else:
            if self.missing_mode == "full":
                G = nx.from_numpy_array(self.m_coefs_r2r, create_using=nx.DiGraph)
                topo_order = list(nx.topological_sort(G))
                R = np.ones_like(X)
                for node in topo_order:
                    p_R_0 = self.func(X, R)
                    R_t = np.random.binomial(n=1, p=1-p_R_0, size=X.shape)
                    R[:, node] = R_t[:, node]

            else:
                # defines the probability that a variable is missing, R_i = 0 indicates X_i is missing
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
        

