import networkx as nx
import numpy as np
import torch 
import math 

from models.nodags.functions import indMLPFunction, nonlinearMLP
from models.nodags.resblock import iResBlock

def standard_normal_logprob(z, noise_scale=0.5):
    logZ = -0.5 * math.log(2 * math.pi * noise_scale**2)
    return logZ - z**2 / (2 * noise_scale**2)

def make_non_cotractive(weights):
    s = np.linalg.svd(weights, compute_uv=False)
    scale = 1.0
    if s[0] <= 1.0:
        scale = 2/s[0]
    
    return scale * weights 

def make_contractive(weights):
    s = np.linalg.svd(weights, compute_uv=False)
    scale=1.1
    if s[0] >= 1.0:
        scale = 1.1 * s[0]
    
    return weights/scale
    

class SEM:

    """
    -------------------------------------------------------------------
    This class models a Linear Structural Equation Model (Linear SEM)
    -------------------------------------------------------------------
    The model is initialized with the number of nodes in the graph and
    the absolute minimum and maximum weights for the edges. 
    """
    def __init__(self, 
                 graph, 
                 abs_weight_low=0.2, 
                 abs_weight_high=0.9, 
                 min_noise_scale=0.2, 
                 max_noise_scale=0.5, 
                 contractive=True,  
                 beta=1.0):
        
        self.graph = graph
        self.abs_weight_low = abs_weight_low 
        self.abs_weight_high = abs_weight_high
        self.contractive = contractive

        self.n_nodes = len(graph.nodes)
        
        self.weights = np.random.uniform(self.abs_weight_low, self.abs_weight_high, size=(self.n_nodes, self.n_nodes))
        self.weights *= 2 * np.random.binomial(1, 0.5, size=self.weights.shape) - 1
        self.weights *= nx.to_numpy_array(self.graph)
        self.min_noise_scale = min_noise_scale
        self.max_noise_scale = max_noise_scale
        self.beta = beta # linear-nonlinear mixing factor

        self.noise_scales = self.min_noise_scale + (self.max_noise_scale - self.min_noise_scale) * np.random.rand(self.n_nodes)

        if not self.contractive:
            self.weights = make_non_cotractive(self.weights)
        else:
            self.weights = make_contractive(self.weights)

    def generateData(self, 
                     n_samples, 
                     intervention_set=[None], # set `intervention_set` to [None] for observational data
                     fixed_intervention=False, 
                     return_latents=False, 
                     n_iter=30, 
                     beta_given=False, 
                     beta=1.0,
                     soft_intervention=False, # set to True if performing soft intervention (shift intervention)
                     soft_intervention_mean=1.0 # set the value of the intervention shift
                    ):
        
        
        observed_set = np.setdiff1d(np.arange(self.n_nodes), intervention_set)
        U = np.zeros((self.n_nodes, self.n_nodes))
        U[observed_set, observed_set] = 1

        if soft_intervention:
            E = self.noise_scales.reshape(-1, 1) * np.random.randn(self.n_nodes, n_samples)
            if intervention_set[0] is not None:
                E[intervention_set, :] += soft_intervention_mean
        
            beta_ = beta if beta_given else self.beta 
            wtx = lambda x: self.weights.T @ X 

            X = np.random.randn(self.n_nodes, n_samples)
            for _ in range(n_iter):
                X = ( beta_ * np.tanh(wtx(X)) + (1 - beta_) * wtx(X) ) + E

            # The final data matrix has dimensions - n_samples X self.nodes
            if return_latents:
                return X.T, E.T
                
            return X.T

        else:

            C = np.zeros((self.n_nodes, n_samples))
            if intervention_set[0] != None:
                if fixed_intervention:
                    C[intervention_set, :] = np.random.randn(len(intervention_set), 1)
                else:
                    C[intervention_set, :] = 1.5 * np.random.randn(len(intervention_set), n_samples)

            E = self.noise_scales.reshape(-1, 1) * np.random.randn(self.n_nodes, n_samples)

            beta_ = beta if beta_given else self.beta 
            wtx = lambda x: self.weights.T @ X 


            X = np.random.randn(self.n_nodes, n_samples)
            for _ in range(n_iter):
                X = U @ ( beta_ * np.tanh(wtx(X)) + (1 - beta_) * wtx(X) ) + U @ E + C

            # The final data matrix has dimensions - n_samples X self.nodes
            if return_latents:
                return X.T, E.T
                
            return X.T
    