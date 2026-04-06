import torch
import numpy as np 
import math
import os
from sklearn.linear_model import LogisticRegression 

# import local libraries
from models.nodags.functions import gumbelSoftMLP
from models.nodags.resblock import iResBlock
from models.nodags.missingness_mechanism import *

def standard_normal_logprob(z, noise_scales):
    logZ = -0.5 * torch.log(2 * math.pi * (noise_scales**(2)))
    return logZ - z.pow(2) / (2 * (noise_scales**(2)))

class missModel:

    def __init__(
        self,
        gen_model: iResBlock,
        missing_mech: missMechanism,
        missing_model_type = 'obs-only', 
        is_mcar = False
    ):

        self.gen_model = gen_model
        self.missing_mech = missing_mech
        self.missing_model_type = missing_model_type
        self.n_nodes = self.gen_model.f.n_nodes # number of observations per sample
        self.is_mcar = is_mcar
        
        # initialize the parameters of the connections between missingness indicators and
        # the observation variables

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.coefs = torch.zeros(self.n_nodes, self.n_nodes, device=self.device)
        # self.intercepts = torch.zeros(self.n_nodes, device=self.device)
        
        
    def compute_log_gen_model_prob(
        self, 
        X: torch.Tensor, 
        intervention_mask: torch.Tensor
    ):
        
        lat_std = torch.exp(self.gen_model.var)
        
        # get the latent variables and the log det-gradient
        lats, logdetgrad = self.gen_model.forward(X, intervention_mask, logdet=True, neumann_grad=False)
        
        # compute the log-density of the latent variables
        logpe = (standard_normal_logprob(lats, noise_scales=lat_std) * intervention_mask).sum(1, keepdim=True)
        
        # compute the log-density of the intervened nodes
        # the intervened nodes follow normal distribution with standard 
        # deviation set to 1.5
        logpx_int = (standard_normal_logprob(X, noise_scales=torch.tensor(1.5)) * (1 - intervention_mask)).sum(1, keepdim=True)
        
        # compute the log-density of observation variables
        logpx = logpx_int + logpe + logdetgrad 
        
        return logpx
    
    def compute_log_missing_prob(
        self, 
        X: torch.Tensor, 
        R: torch.Tensor,
        intervention_mask: torch.Tensor
    ):
        
        return self.missing_mech.logprob(X, R, intervention_mask)
        
    def compute_joint_distribution(
        self, 
        X: torch.Tensor,
        R: torch.Tensor,
        intervention_mask: torch.Tensor,
    ):
        
        # compute the log-density of the observation variables
        logpx = self.compute_log_gen_model_prob(X, intervention_mask)
        
        logpr = torch.zeros_like(logpx)
        if not self.is_mcar:
            # compute the log-density of missingness indicators
            logpr = self.compute_log_missing_prob(X, R, intervention_mask)
        
        # compute log joint distribution
        log_joint = logpx + logpr
        
        return log_joint, torch.exp(log_joint)
    
    def save_model(self, path):
        torch.save(self.gen_model.state_dict(), os.path.join(path, "gen-model.pth"))
        torch.save(self.missing_mech.state_dict(), os.path.join(path, "missing-mech.pth"))



        

        
        
        
        
            
        
