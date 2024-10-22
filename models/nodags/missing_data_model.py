import torch
import numpy as np 
import math
import os
from sklearn.linear_model import LogisticRegression 

# import local libraries
from models.nodags.functions import gumbelSoftMLP
from models.nodags.resblock import iResBlock

def standard_normal_logprob(z, noise_scales):
    logZ = -0.5 * torch.log(2 * math.pi * (noise_scales**(2)))
    return logZ - z.pow(2) / (2 * (noise_scales**(2)))

class missModel:

    def __init__(
        self,
        gen_model: iResBlock,
        missing_model_type = 'obs-only', 
        is_mcar = False
    ):

        self.gen_model = gen_model
        self.missing_model_type = missing_model_type
        self.n_nodes = self.gen_model.f.n_nodes # number of observations per sample
        self.is_mcar = is_mcar
        
        # initialize the parameters of the connections between missingness indicators and
        # the observation variables
        #

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.coefs = torch.zeros(self.n_nodes, self.n_nodes, device=self.device)
        self.intercepts = torch.zeros(self.n_nodes, device=self.device)
        
    def learn_miss_mech_separately(
        self, 
        data: np.ndarray, 
        R:np.ndarray, 
        intervention_mask: np.ndarray,
        C=0.5,
        penalty="none",
        verbose=0,
        solver='liblinear', # [choose between liblinear, saga]
        maxiter=1000
    ):
        '''
        In this function the parameters of the missingness mechanism is directly learned 
        from the data using logistic regression. To that end, the function requires 
        observational data as argument (`data`).  
        '''
        
        coefs = np.zeros((self.n_nodes, self.n_nodes))
        intercepts = np.zeros(self.n_nodes)
        
        if self.missing_model_type == "full":
            coefs = np.vstack([coefs, np.zeros((self.n_nodes, self.n_nodes))])
        
        # iterate over each missingness indicator
        for i in range(self.n_nodes):
            i_minus_index = np.arange(self.n_nodes) != i
            
            # isolate the samples where only the i-th observation variable is missing
            sample_index = (R[:, i_minus_index] == 1).all(axis = 1) & (intervention_mask[:, i] == 1)
            # print("Node - {}, Samples: {}".format(i, sample_index.sum()))
            X = data[sample_index, :][:, i_minus_index] # features for Logistic regression
            y = 1 - R[sample_index, i] # target variable for Logistic regression
            
            # running Logistic regression to get the coefficients and intercepts
            learning_model = LogisticRegression(penalty=penalty, C=C, solver=solver, verbose=verbose, max_iter=maxiter)    
            learning_model.fit(X, y)
            
            # store the coefs at the right location
            coefs[i_minus_index, i] = learning_model.coef_ 
            intercepts[i] = learning_model.intercept_.item()
            
        self.coefs = torch.tensor(coefs, dtype=torch.float, device=self.device)
        self.intercepts = torch.tensor(intercepts, dtype=torch.float, device=self.device)
        
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
        
        # compute p(r_i = 0| x_{-i}). i-th column is the i-th component
        indiv_factors_for_missing = torch.sigmoid(
            X @ self.coefs + self.intercepts
        )
        
        indiv_factors = (1 - indiv_factors_for_missing) * R + indiv_factors_for_missing * (1 - R)
        
        # mask out the intervening nodes, set the corresponding prob to 1
        
        indiv_factor_post_intervene = (1 - intervention_mask) + intervention_mask * indiv_factors
        
        return torch.log(indiv_factors).sum(dim=1, keepdim=True)
        
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

        np.save(
            os.path.join(path, "s-weights.npy"),
            self.coefs.detach().cpu().numpy()
        )

        np.save(
            os.path.join(path, "s-intercepts.npy"),
            self.intercepts.detach().cpu().numpy()
        )

        torch.save(
            self.gen_model.state_dict(),
            os.path.join(path, "gen-model.pth")
        )

    def load_model(self, path, map_location=None):

        self.coefs = np.load(
            os.path.join(path, "s-weights.npy")
        )
        self.coefs = torch.tensor(self.coefs, dtype=torch.float, device=self.device)

        self.intercepts = np.load(
            os.path.join(path, "s-intercepts.npy")
        )
        self.intercepts = torch.tensor(self.intercepts, dtype=torch.float, device=self.device)

        if map_location == None: 
            self.gen_model.load_state_dict(torch.load(
                os.path.join(path, "gen-model.pth"), 
            ))

        else: 
            self.gen_model.load_state_dict(torch.load(
                os.path.join(path, "gen-model.pth"), map_location=map_location
            ))

        

        
        
        
        
            
        
