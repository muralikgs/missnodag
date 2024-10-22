import torch
import torch.nn as nn 
import numpy as np 
import math 
import time 

from models.nodags.functions import gumbelSoftMLP

def standard_normal_logprob(z, noise_scales):
    logZ = -0.5 * torch.log(2 * math.pi * (noise_scales**(2)))#(noise_scales.pow(2))) # change this back to pow
    return logZ - z.pow(2) / (2 * (noise_scales**(2)))

def dag_constraint(W, s=1, method='log-det'):
    if method == 'expm':
        return torch.trace(torch.matrix_exp(W * W)) - W.shape[0]
    elif method == 'log-det':
        return -torch.log(s * torch.det(torch.eye(W.shape[0], device=W.device) - W * W)) + W.shape[0] * math.log(s)

class iResBlock(nn.Module):
    """
    ----------------------------------------------------------------------------------------
    The class for a single residual map, i.e., (I -f)(x) = e. 
    ----------------------------------------------------------------------------------------
    The forward method computes the residual map and also log-det-Jacobian of the map. 

    Parameters:
    1) func - (nn.Module) - torch module for modelling the function f in (I - f).
    2) n_power_series - (int/None) - Number of terms used for computing determinent of log-det-Jac, 
                                     set it to None to use Russian roulette estimator. 
    3) neumann_grad - (bool) - If True, Neumann gradient estimator is used for Jacobian.
    4) n_dist - (string) - distribution used to sample n when using Russian roulette estimator. 
                           'geometric' - geometric distribution.
                           'poisson' - poisson distribution.
    5) lamb - (float) - parameter of poisson distribution.
    6) geom_p - (float) - parameter of geometric distribution.
    7) n_samples - (int) - number of samples to be sampled from n_dist. 
    8) grad_in_forward - (bool) - If True, it will store the gradients of Jacobian with respect to 
                                  parameters in the forward pass. 
    9) n_exact_terms - (int) - Minimum number of terms in the power series. 
    """
    def __init__(
        self, func:gumbelSoftMLP,
        n_power_series, 
        neumann_grad=True, 
        n_dist='geometric', 
        lamb=2., 
        geom_p=0.5, 
        n_samples=1, 
        grad_in_forward=False, 
        n_exact_terms=2, 
        single_var=False,
        precondition=False, # Must be true when the input is DAG 
        lin_logdet=False, 
        centered=True
    ):
        
        super(iResBlock, self).__init__()
        self.f = func
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1. - geom_p)))
        self.lamb = nn.Parameter(torch.tensor(lamb))
        self.n_dist = n_dist
        self.n_power_series = n_power_series 
        self.neumann_grad = neumann_grad 
        self.grad_in_forward = grad_in_forward
        self.n_exact_terms = n_exact_terms
        self.n_samples = n_samples
        self.precondition = precondition
        self.lin_logdet = lin_logdet
        self.centered = centered
        
        # initialize the variance of latents as a trainable parameter
        if single_var:
            self.var = nn.Parameter(torch.tensor(1.)).float()
        else:
            self.var = nn.Parameter(torch.ones(self.f.n_nodes, dtype=torch.float)) 
        
        # when the data is not centered, learn the means
        if not self.centered:
            self.mu = nn.Parameter(torch.zeros(self.f.n_nodes).float())
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # device = torch.device('cpu')
            self.mu = torch.zeros(self.f.n_nodes).float().to(device)

        if precondition:
            self.Lambda = nn.Parameter(torch.zeros(self.f.n_nodes).float())

    def forward(self, x, mask, logdet=False, neumann_grad=True):
        # set intervention set to [None]
        self.neumann_grad = neumann_grad
        if len(mask.shape) == 1:
            mask = 1
        if not logdet:
            y = x - self.f(x) * mask
            return y
        else:
            if self.precondition:
                Lamb_mat = torch.diag(torch.exp(self.Lambda))
                Lamb_mat_inv = torch.diag(1/torch.exp(self.Lambda))
                # print(x.device, self.mu.device, Lamb_mat.device)
                x_inp = (x - self.mu) @ Lamb_mat
    
            else:
                x_inp = x - self.mu
            f_x, logdetgrad, _ = self._logdetgrad(x_inp, mask)
            if self.precondition:
                return (x - self.mu) - (f_x @ Lamb_mat_inv) * mask, logdetgrad
            else:
                return (x - self.mu) - f_x * mask, logdetgrad 

    def predict_from_latent(self, latent_vec, mask, x_init=None, n_iter=20, threshold=1e-4):
        x = torch.randn(latent_vec.size(), device=latent_vec.device)
        mask_cmp = torch.ones_like(mask) - mask
        c = x_init * mask_cmp

        if self.precondition:
            Lamb_mat = torch.diag(torch.exp(self.Lambda))
            Lamb_mat_inv = torch.diag(1/torch.exp(self.Lambda))

        for _ in range(n_iter):
            x_t = x
            x_inp = x - self.mu 
            if self.precondition:
                x_inp = (x - self.mu) @ Lamb_mat
                f_x = self.f(x_inp) @ Lamb_mat_inv
            else:
                f_x = self.f(x_inp)

            x = f_x * mask + (latent_vec + self.mu) * mask + c 
            if torch.norm(x_t - x) < threshold:
                break
    
        return x 

    def _logdetgrad(self, x, mask):
        with torch.enable_grad():
            if self.n_dist == 'geometric':
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn = lambda m: geometric_sample(geom_p, m)
                rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)
            elif self.n_dist == 'poisson':
                lamb = self.lamb.item()
                sample_fn = lambda m: poisson_sample(lamb, m)
                rcdf_fn = lambda k, offset: poisson_1mcdf(lamb, k, offset)
            
            # if self.training:
            if self.n_power_series is None:
                # Unbiased estimation.
                lamb = self.lamb.item()
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + self.n_exact_terms
                coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * \
                    sum(n_samples >= k - self.n_exact_terms) / len(n_samples)
            else:
                # Truncated estimation.
                n_power_series = self.n_power_series
                coeff_fn = lambda k: 1.

            vareps = torch.randn_like(x)

            if self.lin_logdet:
                estimator_fn = linear_logdet_estimator
            else:
                # if self.training and self.neumann_grad:
                if self.neumann_grad:
                    estimator_fn = neumann_logdet_estimator
                else:
                    estimator_fn = basic_logdet_estimator

            if self.training and self.grad_in_forward:
                f_x, logdetgrad = mem_eff_wrapper(
                    estimator_fn, self.f, x, n_power_series, vareps, coeff_fn, self.training
                )
            else:
                x = x.requires_grad_(True)
                f_x = self.f(x)
                tic = time.time()
                if self.lin_logdet:
                    Weight = self.f.layer.weight
                    self_loop_mask = torch.ones_like(Weight)
                    ind = np.diag_indices(Weight.shape[0])
                    self_loop_mask[ind[0], ind[1]] = 0 
                    logdetgrad = estimator_fn(mask * self_loop_mask * Weight, x.shape[0])
                else:
                    logdetgrad = estimator_fn(f_x * mask, x, n_power_series, vareps, coeff_fn, self.training)
                toc = time.time()
                comp_time = toc - tic 

        return f_x, logdetgrad.view(-1, 1), comp_time
        
    def losses(self, 
               x: torch.Tensor, 
               intervention_mask: torch.Tensor, 
               lambda_c=1e-2, 
               lambda_dag=1.0, 
               obs=False, 
               fun_type='gst-mlp', 
               s=1, 
               method='expm', 
               neumann_grad=True):

        e, logdetgrad = self.forward(x, intervention_mask, logdet=True, neumann_grad=neumann_grad)

        # lat_std = torch.exp(self.var)
        lat_std = torch.exp(self.var*torch.ones(self.f.n_nodes, device=x.device))
        # if len(mask.shape) == 1:
        #     mask = 1
        logpe = (standard_normal_logprob(e, noise_scales=lat_std) * intervention_mask).sum(1, keepdim=True)
        logpx = logpe + logdetgrad

        loss = -torch.mean(logpx)
        if fun_type == 'fac-mlp' or fun_type == 'gst-mlp':
            l1_norm = self.f.get_w_adj().abs().sum()
        else:
            l1_norm = sum(p.abs().sum() for p in self.parameters())

        loss_pen = loss + lambda_c * l1_norm

        if obs:
            if fun_type == "gst-mlp":
                h_w = dag_constraint(self.f.get_w_adj().abs(), s=s, method=method)
            elif fun_type == "lin-mlp":
                w = self.f.layer.weight.T
                h_w = dag_constraint(w, method=method)

            loss_pen += lambda_dag * h_w
            return loss_pen, loss, torch.mean(logdetgrad), h_w

        return loss_pen, loss, torch.mean(logdetgrad)
    
    def neg_log_likelihood(self, x, mask):
        _, nll, _ = self.losses(x, mask, neumann_grad=False)
        return nll

    def get_w_adj(self):
        return self.f.get_w_adj().detach().cpu().numpy()

def basic_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    logdetgrad = torch.tensor(0.).to(x)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
        tr = torch.sum(vjp.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        delta = -1 / k * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad


def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1) * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
    return logdetgrad

def linear_logdet_estimator(W, bs):
    n = W.shape[0]
    I = torch.eye(n, device=W.device)
    return torch.log(torch.det(I - W)) * torch.ones(bs, 1, device=W.device)

def mem_eff_wrapper(): # Function to store the gradients in the forward pass. To be implemented. 
    return 0

def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)

def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)

def poisson_sample(lamb, n_samples):
    return np.random.poisson(lamb, n_samples)

def poisson_1mcdf(lamb, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    sum = 1.
    for i in range(1, k):
        sum += lamb**i / math.factorial(i)
    return 1 - np.exp(-lamb) * sum