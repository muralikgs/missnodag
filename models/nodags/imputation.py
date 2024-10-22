import torch
import numpy as np
from scipy.stats import multivariate_normal
import math
from torch.distributions import MultivariateNormal

from models.nodags.resblock import iResBlock
from models.nodags.missing_data_model import missModel

def standard_normal_logprob(z, noise_scales):
    logZ = -0.5 * torch.log(2 * math.pi * (noise_scales**(2)))#(noise_scales.pow(2))) # change this back to pow
    return logZ - z.pow(2) / (2 * (noise_scales**(2)))

def get_permutation_matrices(missing):

    with torch.no_grad():
        P = torch.zeros(missing.shape[0], missing.shape[1], missing.shape[1], device=missing.device)
        for i, mis in enumerate(missing):
            pi = torch.concatenate((
                torch.nonzero(1 - mis), # missing nodes
                torch.nonzero(mis) # observed nodes
            ))
            P[i, range(len(pi)), pi.squeeze()] = 1

    return P

# As of now torch.nanvar doesn't exist. Found this code as a stop gap solution
# from here - https://github.com/pytorch/pytorch/issues/61474

def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output

def impute_data_linear(x, masks, missing, model, B_gt=None, use_ground_truth=False, lin_approx=False):

    with torch.no_grad():

        observ_mat = torch.diag(masks[0])

        if use_ground_truth:
            B = B_gt @ observ_mat
            Omega = (1/0.25) * torch.eye(x.shape[1], device=x.device).float()

        else:
            if lin_approx:
                # nonlinear data - using the derivative at the mean (0) as an approximation to the weighted adjacency
                # matrix. (First order Taylor series approximation)

                lam_mat_inv = torch.diag(1/torch.exp(model.Lambda))
                func = lambda x: model.f(x) @ lam_mat_inv
                inputs = torch.zeros(1, x.shape[1], device=x.device)

                B = torch.autograd.functional.jacobian(func, inputs).squeeze().T


            else:
                # extract the current adjacency matrix and the model weights

                w_param = model.f.get_w_adj()
                self_loop_mask = torch.ones_like(w_param) - torch.eye(w_param.shape[0], device=w_param.device)
                w_curr = torch.bernoulli(w_param * self_loop_mask)
                B = model.f.layers[0].weight.T * w_curr @ observ_mat

            # extract the variance of latent noise

            lat_var = torch.exp(model.var) ** 2.0
            inv_var_vec = masks[0] * (1/lat_var) * torch.ones(B.shape[0], device=lat_var.device) \
                            + (1 - masks[0]) * (1/0.25) * torch.ones(B.shape[0], device=lat_var.device)
            Omega = torch.diag(inv_var_vec)

        # compute the inverse covariance of x

        I = torch.eye(B.shape[0], device=B.device)
        precision_mat = (I - B) @ Omega @ (I - B.T)

        P = get_permutation_matrices(missing)
        permuted_precisions = torch.einsum("bij,jk,bkl->bil", P, precision_mat, P.permute(0,2,1))

        # perform Cholesky decomposition on the precision matrix

        U = torch.linalg.cholesky(permuted_precisions, upper=True)
        repermuted_U = torch.einsum("bij,bjk,bkl->bil", P.permute(0,2,1), U, P)

        # get k - maximum number of nodes missing per batch

        k = (1 - missing).sum(axis=1).max().detach().int()


        # modify U to compensate for the different number of missing nodes per sample.

        U_mod = torch.zeros(U.shape[0], U.shape[1] + k, U.shape[2] + k, device=U.device)
        U_mod[:] = torch.eye(U.shape[1] + k, device=U_mod.device)
        U_mod[:, :U.shape[1], :U.shape[2]] = repermuted_U

        # modify missing to account for different number of missing nodes per sample.

        mis = (1 - missing).int()
        mis_row_ones = mis.sum(axis=1)
        mis_normalized = torch.zeros(mis.shape[0], mis.shape[1] + k, device=mis.device)
        mis_normalized[:, :mis.shape[1]] = mis

        # naive method - TODO: Think of a more efficient way to do this.

        for i in range(k):
            mat = torch.ones_like(mis_normalized[:, mis.shape[1]:])
            if i > 0:
                mat[:, -i:] = 0
            mis_normalized[:, mis.shape[1]:] += ((mis_row_ones == i) * mat.T).T

        # create a batch of U[mis:mis] matrix

        ind = torch.nonzero(mis_normalized[:, :])[:, 1].view(mis.shape[0], k)
        ind = ind.to(U_mod.device)
        row = torch.tensor([[i] for i in range(len(mis))], device=ind.device)
        U_mis = U_mod[row, ind]
        U_mis_mis = U_mis[row, :, ind].swapaxes(2, 1)

        # create a batch of U[mis:obs] matrix

        obs_normalized = (1 - mis_normalized).int()
        obs_ind = torch.nonzero(obs_normalized[:, :])[:, 1].view(obs_normalized.shape[0], U.shape[1])
        obs_ind = obs_ind.to(U_mod.device)
        U_mis_obs = U_mis[row, :, obs_ind].swapaxes(1, 2)

        # create a batch x_obs matrix

        x_mod = torch.zeros(x.shape[0], x.shape[1] + k, device=x.device)
        x_mod[:, :x.shape[1]] = x
        x_obs = torch.stack(tuple(x_mod[row, obs_ind]))

        # solve for the missing values

        z = torch.randn(x.shape[0], k, device=x.device)
        mu = torch.einsum("bij,bj->bi", U_mis_obs, x_obs)
        x_mis = torch.linalg.solve_triangular(U_mis_mis, (z - mu).view(z.shape[0], 1, -1), upper=True, left=False).squeeze()


        # imputed batch of samples

        x_mod[row, ind] = x_mis.view(len(row), ind.shape[1])
        x_imputed = x_mod[:,:x.shape[1]]

    return x_imputed

def check_acceptance(
    X_imputed: torch.Tensor,
    miss_model: missModel,
    log_q_cond_dis: torch.Tensor, # log prob
    intervention_mask: torch.Tensor, # earlier - mask
    R: torch.Tensor, # missingness indicator
    accepted: torch.Tensor
):

    sample_num = X_imputed.shape[0]

    # compute the joint model distribution (this computes the log-prob)
    log_p_joint_dis, _ = miss_model.compute_joint_distribution(X_imputed, R, intervention_mask)

    # k should satisfy the following inequality
    # k Q(x_m) >= P(x_m, x_o, r)
    # k = torch.exp(log_p_joint_dis.view(-1)[accepted] - log_q_cond_dis[accepted]).max()
    # dele_index = []
    # 
    k = ((1 - accepted*1.0)*torch.exp(log_p_joint_dis.view(-1) - log_q_cond_dis)).max()

    prob_upper_lim = torch.exp(
        (1 - accepted*1.0) * (log_p_joint_dis.view(-1) - log_q_cond_dis - torch.log(k))
    )

    random_u = torch.rand(sample_num, device=log_p_joint_dis.device)
    accepted_samples = random_u <= prob_upper_lim

    return accepted_samples, k

def impute_mcmc_rejection(
    X: torch.Tensor,
    intervention_mask: torch.Tensor, # intervention_mask (earlier - mask)
    R: torch.Tensor, # missingness indicator (earlier - missing)
    miss_model: missModel,
    max_repetitions=1000,
    min_accept_factor=0.5
):

    nr, nc = X.shape
    Mu = torch.nanmean(X, dim=0)
    S_nan = nanvar(X, dim=0)

    X_sampling_list = []
    num_index = []

    Sampling_model = MultivariateNormal(
        loc=Mu,
        covariance_matrix=torch.diag(S_nan)
    )

    R_t = R.clone()

    # initially only the samples without any missing variables are accepted
    accepted = R_t.sum(dim=1) == nc

    # initialize the imputed data to the original missing data
    X_imputed = torch.nan_to_num(X, nan=0)

    k_list = list()

    # define one iteration for now
    for t in range(max_repetitions):
        # generate samples from the sampling distribution
        samples_mispart = Sampling_model.sample(sample_shape=torch.Size((nr,)))

        # compute the density of the generated samples corresponding to the missing coordinates
        log_q_cond_dis = (standard_normal_logprob(samples_mispart, noise_scales=S_nan**0.5) * (1 - R_t)).sum(dim=1)

        # impute the missing values with the samples generated
        X_imputed = X_imputed * R_t + samples_mispart * (1 - R_t)
        
        # accept samples based on acceptance criteria
        accepted, k = check_acceptance(
            X_imputed,
            miss_model,
            log_q_cond_dis,
            intervention_mask,
            R,
            accepted
        )
        
        # k_list.append(k.detach().numpy())

        R_t[accepted] = 1

        if accepted.sum() >= min_accept_factor * nr:
            return X_imputed[accepted], accepted, t, k_list

    return X_imputed[accepted], accepted, t, k_list
