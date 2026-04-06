import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from copy import copy

from models.nodags.missing_data_model import missModel
from models.nodags.imputation import impute_mcmc_rejection
from models.nodags.layers.mlpLipschitz import linearLipschitz
from utils.error_metrics import *

def update_lipschitz(model, n_iterations):
    for m in model.modules():
        if isinstance(m, linearLipschitz):
            m.compute_weight(update=True, n_iterations=n_iterations)

class Trainer:

    def __init__(
        self,
        miss_model: missModel,
        obs=False,
        dag_method='expm',
        missingness_learn="em", # choose between ["initial", "em"]
        s=1,
        lr=1e-2,
        lr_miss=1e-2,
        lambda_c=1e-3,
        lambda_dag=10.0,
        max_epochs=200,
        batch_size=512,
        lambda_nc=10.0,
        lambda_mm_c=1e-2,
        true_adj=None,
        n_lip_iters=5 # controls the number of iterations used to maintain the Lipschitz
                      # constant of the NN weights
    ):

        self.miss_model = miss_model
        self.lr = lr
        self.lr_miss = lr_miss
        self.lambda_c = lambda_c
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_lip_iters = n_lip_iters
        self.lambda_dag = lambda_dag 
        self.obs = obs
        self.dag_method = dag_method 
        self.s = s 
        self.missingness_learn = missingness_learn
        self.lambda_nc = lambda_nc
        self.true_adj = true_adj
        self.lambda_mm_c = lambda_mm_c

        self.optimizer = torch.optim.Adam(self.miss_model.gen_model.parameters(), lr=self.lr)
        self.optimizer_miss_mech = torch.optim.Adam(self.miss_model.missing_mech.parameters(), lr=self.lr_miss)

    def reinit_optimizers(self):
        self.optimizer = torch.optim.Adam(self.miss_model.gen_model.parameters(), lr=self.lr)
        self.optimizer_miss_mech = torch.optim.Adam(self.miss_model.missing_mech.parameters(), lr=self.lr_miss)


    def learn_missingness_mech(
        self,
        data: np.ndarray,
        R: np.ndarray, 
        intervention_mask: np.ndarray,
    ):
        self.miss_model.missing_mech.learn_missing_mech_separately(
            data, 
            R, 
            intervention_mask, 
            penalty="l1",
            maxiter=100
        )

    def train(
        self,
        data: Dataset,
        print_loss=True,
        print_interval=50,
        data_missing=True,
        min_accept_factor=0.5,
        store_grad_norm=False,
        compute_shd_iter=False,
    ):

        logpx_obs_list = list()
        iteration_count = list()
        acceptance_rate_list = list()
        repetitions_list = list()
        h_w_list = list()
        hw_r2r_list = list()
        nc_cons_list = list()
        miss_loss_list = list()
        coefs_norm_list = list()
        miss_mech_grad_norms = list()
        shd_list = list()

        training_dataloader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.coefs_x2r_list = list()
        self.coefs_r2r_list = list()

        for epoch in range(self.max_epochs):
            av_loss = 0
            logpx_obs = 0
            count = 0
            av_accepted_samples = 0
            av_hw = 0
            repetitions = 0
            av_miss_loss = 0
            av_hw_r2r = 0
            av_nc_cons = 0

            for it, batch in enumerate(training_dataloader):

                self.optimizer.zero_grad()
                self.optimizer_miss_mech.zero_grad()

                X_clean, X_miss, R, intervention_mask = batch[0].float().to(device), batch[1].float().to(device), batch[2].float().to(device), batch[3].float().to(device)

                if data_missing:
                    X, accepted, t, k_list = impute_mcmc_rejection(
                        X=X_miss,
                        intervention_mask=intervention_mask,
                        R=R,
                        miss_model=self.miss_model,
                        min_accept_factor=min_accept_factor
                    )
                    R = R[accepted]
                    intervention_mask = intervention_mask[accepted]
                else:
                    X = X_clean
                    t = 0

                av_accepted_samples += len(X)
                repetitions += t

                loss_pen, nll, _, h_w = self.miss_model.gen_model.losses(
                    x = X,
                    intervention_mask=intervention_mask,
                    lambda_c=self.lambda_c,
                    lambda_dag=1.0,
                    obs=self.obs
                )
                av_loss += loss_pen.item()
                logpx_obs += -nll.item()
                count += 1
                av_hw += h_w

                loss_pen.backward()
                self.optimizer.step()
                update_lipschitz(self.miss_model.gen_model, n_iterations=self.n_lip_iters)

                if self.missingness_learn == "em":
                    logpr, nll_loss, h_w_r2r, nc_cons = self.miss_model.missing_mech.loss(
                        X,
                        R, 
                        intervention_mask, 
                        lambda_dag=self.lambda_dag, 
                        lambda_nc=self.lambda_nc,
                        lambda_spar=self.lambda_mm_c
                    )

                    nll_loss.backward()
                    
                    if store_grad_norm:
                        grad_norm_sum = 0.0
                        coefs_norm_sum = 0.0
                        for param in self.miss_model.missing_mech.parameters():
                            if param.grad is not None:
                                grad_norm_sum += param.grad.norm().item()
                                coefs_norm_sum += param.norm().item()
                        miss_mech_grad_norms.append(grad_norm_sum)
                        coefs_norm_list.append(coefs_norm_sum)

                    self.optimizer_miss_mech.step()
                    av_miss_loss += logpr.item()
                    av_hw_r2r += h_w_r2r.item() 
                    av_nc_cons = nc_cons.item()

                    if self.miss_model.missing_mech.constraint_optimization == "aug-lagrangian":
                        with torch.no_grad():
                            self.miss_model.missing_mech.lambda_dag_aug += self.miss_model.missing_mech.rho * h_w_r2r.item()
                            self.miss_model.missing_mech.lambda_nc_aug += self.miss_model.missing_mech.rho * nc_cons.item()

                if print_loss:
                    # normalize the statistics
                    av_loss /= count
                    av_accepted_samples /= count
                    av_hw /= count
                    av_miss_loss /= count 
                    av_hw_r2r /= count 
                    av_nc_cons /= count
                    logpx_obs_list.append(logpx_obs/count)
                    acceptance_rate_list.append(av_accepted_samples / len(X_miss))
                    iteration_count.append(epoch * len(training_dataloader) + count)
                    repetitions_list.append(repetitions / count)
                    h_w_list.append(av_hw)
                    miss_loss_list.append(av_miss_loss)
                    hw_r2r_list.append(av_hw_r2r)
                    nc_cons_list.append(av_nc_cons)
                    # coefs_norm_list.append(torch.norm(self.miss_model.missing_mech.coefs).item())
                    self.coefs_x2r_list.append(copy(self.miss_model.missing_mech.coefs_x2r.detach().cpu().numpy()))
                    self.coefs_r2r_list.append(copy(self.miss_model.missing_mech.coefs_r2r.detach().cpu().numpy()))

                    # Compute the SHD
                    if compute_shd_iter:
                        est_adj = self.miss_model.gen_model.get_w_adj()
                        shd, _ = compute_shd(np.abs(self.true_adj) > 0, np.abs(est_adj) > 0.85)
                        shd_list.append(shd)
                    else:
                        shd_list.append(None)
                        
                    

                    print("Epoch: {}/{}, Iter: {}/{}, Loss: {}, Loss (Miss): {}".format(epoch+1, self.max_epochs, it, len(training_dataloader), av_loss, av_miss_loss), end="\r", flush=True)

                    # Reset the counts and statistics
                    count = 0
                    av_loss = 0
                    logpx_obs = 0
                    av_accepted_samples = 0
                    repetitions = 0
                    av_hw = 0
                    av_miss_loss = 0
                    av_hw_r2r = 0
                    av_nc_cons = 0 

        stats = (logpx_obs_list, iteration_count, acceptance_rate_list, repetitions_list, h_w_list, miss_loss_list, hw_r2r_list, nc_cons_list, coefs_norm_list, miss_mech_grad_norms, shd_list)
        return stats
