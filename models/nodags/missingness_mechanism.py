import torch 
import os 
import math

from typing import Literal
import torch.nn as nn 
import numpy as np 

from sklearn.linear_model import LogisticRegression

def dag_constraint(W, s=1, method='expm'):
    if method == 'expm':
        return torch.trace(torch.matrix_exp(W * W)) - W.shape[0]
    elif method == 'log-det':
        return -torch.log(s * torch.det(torch.eye(W.shape[0], device=W.device) - W * W)) + W.shape[0] * math.log(s)
    else:
        return torch.tensor(-1.0)
    
def no_colluder_constraint(w1, w2):
    return (w1 * w2).sum()

class missMechanism(nn.Module): 

    def __init__(
            self, 
            n_nodes,
            learning_method="em", # choose between ["em", "initial"]
            constraint_optimization="lagrangian-mul", # choose between ["lagrangian-mul", "aug-lagrangian"]
            C=0.5, # regularization constant for logistic regression
            rho=1.0
    ):
        super(missMechanism, self).__init__()

        self.n_nodes = n_nodes 
        self.learning_method = learning_method
        self.C = C
        self.constraint_optimization = constraint_optimization
        self.lambda_dag_aug = 0.0
        self.lambda_nc_aug = 0.0
        self.rho = rho

    def learn_missing_mech_separately(self, data, R, intervention_mask, penalty, verbose=0, maxiter=100): 
        print("Default Implementation")

    def logprob(self, X: torch.Tensor, R: torch.Tensor, intervention_mask: torch.Tensor): 
        print("Default Implementation")
        return torch.zeros(5)
    
    def loss(self, X, R, intervention_mask, lambda_dag=1.0, lambda_nc=1.0, lambda_spar=1e-2):
        logpr = torch.mean(self.logprob(X, R, intervention_mask))
        nll = -logpr + self.C * self.l1_norm()
        h_w = torch.zeros(1)
        nc_cons = torch.zeros(1)
        return logpr, nll, h_w, nc_cons


    def l1_norm(self):
        print("Default Implementation")
        return torch.zeros(1)

    def save_model(self, path):
        print("Default Implementation")

    def load_model(self, path, map_location=None):
        print("Default Implementation")

class blockParallel(missMechanism):

    def __init__(
            self, 
            n_nodes, 
            learning_method="em", # choose between ["em", "initial"]
            C=0.5 # regularization constant for logistic regression
        ):

        super(blockParallel, self).__init__(n_nodes, learning_method, "lagrangian-mul", C)

        self.coefs = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes, dtype=torch.float))
        self.intercepts = nn.Parameter(torch.zeros(self.n_nodes, dtype=torch.float))

    
    def learn_missing_mech_separately(
        self, 
        data: np.ndarray, 
        R:np.ndarray, 
        intervention_mask: np.ndarray,
        penalty=None,
        verbose=0,
        maxiter=1000
    ):
        '''
        In this function the parameters of the missingness mechanism is directly learned 
        from the data using logistic regression. To that end, the function requires 
        observational data as argument (`data`).  
        '''
        
        coefs = np.zeros((self.n_nodes, self.n_nodes))
        intercepts = np.zeros(self.n_nodes)
        
        
        # iterate over each missingness indicator
        for i in range(self.n_nodes):
            i_minus_index = np.arange(self.n_nodes) != i
            
            # isolate the samples where everything but the i-th node is fully observed and i-th node is not intervened
            # the intervention criteria is necessary because if i-th node is intervened on, then it is always considered 
            # to be observed, regardless of what the missingness indicator says. 
            # sample_index = (R[:, i_minus_index] == 1).all(axis = 1) & (intervention_mask[:, i] == 1)
            sample_index = (R[:, i_minus_index] == 1).all(axis = 1) & (intervention_mask[:, i] == 1)
            X = data[sample_index, :][:, i_minus_index] # features for Logistic regression
            y = 1 - R[sample_index, i] # target variable for Logistic regression
            
            # running Logistic regression to get the coefficients and intercepts
            learning_model = LogisticRegression(penalty=penalty, C=self.C, solver="liblinear", verbose=verbose, max_iter=maxiter)    
            learning_model.fit(X, y)
            
            # store the coefs at the right location
            coefs[i_minus_index, i] = learning_model.coef_ 
            intercepts[i] = learning_model.intercept_.item()
            
        self.coefs = nn.Parameter(torch.tensor(coefs, dtype=torch.float, device=self.coefs.device))
        self.intercepts = nn.Parameter(torch.tensor(intercepts, dtype=torch.float, device=self.intercepts.device))

    def logprob(
            self, 
            X: torch.Tensor, 
            R: torch.Tensor,
            intervention_mask: torch.Tensor
        ):
        
        # Compute P(R_i = 0 | X_{-i}) - i-th column is the i-th component
        diag_mask = torch.ones_like(self.coefs)
        diag_mask.fill_diagonal_(0)
        indiv_factors_for_missing = torch.sigmoid(
            X @ (self.coefs * diag_mask) + self.intercepts
        )  

        indiv_factors = (1 - indiv_factors_for_missing) * R + indiv_factors_for_missing * (1 - R)
        
        # mask out the intervening nodes, set the corresponding prob to 1
        
        indiv_factors_post_intervene = (1 - intervention_mask) + intervention_mask * indiv_factors
        
        return (torch.log(indiv_factors_post_intervene) * intervention_mask).sum(dim=1, keepdim=True)
    
    def prob(
            self, 
            X: torch.Tensor, 
            R: torch.Tensor,
            intervention_mask: torch.Tensor
        ):

        # Compute P(R_i = 0 | X_{-i}) - i-th column is the i-th component
        diag_mask = torch.ones_like(self.coefs_x2r)
        diag_mask.fill_diagonal_(0)
        indiv_factors_for_missing = torch.sigmoid(
            X @ (self.coefs * diag_mask) + self.intercepts
        ) 

        indiv_factors_for_obs = 1 - indiv_factors_for_missing
        
        indiv_factors_post_intervene = (1 - intervention_mask) + intervention_mask * indiv_factors_for_obs

        return indiv_factors_post_intervene
    
    def l1_norm(self):
        return torch.abs(self.coefs).sum()
    

class identifiableMNAR(missMechanism):

    def __init__(
            self, 
            n_nodes,
            learning_method="em", # choose between ["em", "initial"]
            constraint_optimization="lagrangian-mul", # choose between ["lagrangian-mul", "aug-lagrangian"]
            C=0.5, # regularization constant for logistic regression
            rho=2.0
        ):

        super(identifiableMNAR, self).__init__(n_nodes, learning_method, constraint_optimization, C, rho)

        self.coefs_x2r = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes, dtype=torch.float))
        self.coefs_r2r = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes, dtype=torch.float))
        self.intercepts = nn.Parameter(torch.zeros(self.n_nodes, dtype=torch.float))

    def learn_missing_mech_separately(
        self, 
        data: np.ndarray, 
        R:np.ndarray, 
        intervention_mask: np.ndarray,
        penalty=None,
        verbose=0,
        maxiter=1000
    ):
        
        coefs_x2r = np.zeros((self.n_nodes, self.n_nodes))
        coefs_r2r = np.zeros((self.n_nodes, self.n_nodes))
        intercepts = np.zeros(self.n_nodes)

        for i in range(self.n_nodes):
            i_minus_index = np.arange(self.n_nodes) != i 

            # check blockParallel for explanation of the sample filtration below 
            sample_index = (R[:, i_minus_index] == 1).all(axis = 1) & (intervention_mask[:, i] == 1)
            X1 = data[sample_index, :][:, i_minus_index] # observed features for Logistic regression
            R1 = R[sample_index, :][:, i_minus_index] # missingness indicator features for Logistic regression

            X = np.hstack((X1, R1)) # combined features
            y = 1 - R[sample_index, i] # target variable

            learning_model = LogisticRegression(penalty=penalty, C=self.C, solver="liblinear", verbose=verbose, max_iter=maxiter)
            learning_model.fit(X, y)

            # store the coefs at the right location
            coefs_x2r[i_minus_index, i] = learning_model.coef_.squeeze()[:self.n_nodes - 1]
            coefs_r2r[i_minus_index, i] = learning_model.coef_.squeeze()[self.n_nodes - 1:]
            intercepts[i] = learning_model.intercept_.item()

        self.coefs_x2r = nn.Parameter(torch.tensor(coefs_x2r, dtype=torch.float, device=self.coefs_x2r.device))  
        self.coefs_r2r = nn.Parameter(torch.tensor(coefs_r2r, dtype=torch.float, device=self.coefs_r2r.device))
        self.intercepts = nn.Parameter(torch.tensor(intercepts, dtype=torch.float, device=self.intercepts.device))

    def logprob(
            self, 
            X: torch.Tensor, 
            R: torch.Tensor,
            intervention_mask: torch.Tensor
        ):
        
        # Compute P(R_i = 0 | X_{-i}, R_{-i}) - i-th column is the i-th component
        diag_mask = torch.ones_like(self.coefs_x2r)
        diag_mask.fill_diagonal_(0)
        indiv_factors_for_missing = torch.sigmoid(
            X @ (self.coefs_x2r * diag_mask) + R @ (self.coefs_r2r * diag_mask) + self.intercepts
        )     

        indiv_factors = (1 - indiv_factors_for_missing) * R + indiv_factors_for_missing * (1 - R)
        
        # mask out the intervening nodes, set the corresponding prob to 1
        
        indiv_factors_post_intervene = (1 - intervention_mask) + intervention_mask * indiv_factors
        
        return torch.log(indiv_factors_post_intervene).sum(dim=1, keepdim=True)
    
    def l1_norm(self):
        return torch.abs(self.coefs_x2r).sum() + torch.abs(self.coefs_r2r).sum()

    def loss(self, X, R, intervention_mask, lambda_dag=1.0, lambda_nc=1.0, lambda_spar=1e-3):
        logpr = torch.mean(self.logprob(X, R, intervention_mask))
        nll = -logpr + self.C * self.l1_norm()

        h_w = dag_constraint(torch.abs(self.coefs_r2r))
        nc_cons = no_colluder_constraint(torch.abs(self.coefs_r2r), torch.abs(self.coefs_x2r))
        
        nll += lambda_dag * h_w + lambda_nc * nc_cons + lambda_spar * ( self.coefs_x2r.abs().sum() + self.coefs_r2r.abs().sum() ) 

        if self.constraint_optimization == "aug-lagrangian":
            nll += 0.5 * self.rho * (h_w**2 + nc_cons**2)

        return logpr, nll, h_w, nc_cons
    
    def prob(
            self, 
            X: torch.Tensor, 
            R: torch.Tensor,
            intervention_mask: torch.Tensor
        ):

        # Compute P(R_i = 0 | X_{-i}) - i-th column is the i-th component
        diag_mask = torch.ones_like(self.coefs_x2r)
        diag_mask.fill_diagonal_(0)
        indiv_factors_for_missing = torch.sigmoid(
            X @ (self.coefs_x2r * diag_mask) + (1 - R) @ (self.coefs_r2r * diag_mask) + self.intercepts
        ) 

        indiv_factors_for_obs = 1 - indiv_factors_for_missing
        
        indiv_factors_post_intervene = (1 - intervention_mask) + intervention_mask * indiv_factors_for_obs

        return indiv_factors_post_intervene

