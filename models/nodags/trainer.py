import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from models.nodags.missing_data_model import missModel
from models.nodags.imputation import impute_mcmc_rejection
from models.nodags.layers.mlpLipschitz import linearLipschitz

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
        s=1,
        lr=1e-2,
        lambda_c=1e-3,
        lambda_dag=10.0,
        max_epochs=200,
        batch_size=512,
        n_lip_iters=5 # controls the number of iterations used to maintain the lipschitz
                      # constant of the NN weights
    ):

        self.miss_model = miss_model
        self.lr = lr
        self.lambda_c = lambda_c
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_lip_iters = n_lip_iters
        self.lambda_dag = lambda_dag 
        self.obs = obs
        self.dag_method = dag_method 
        self.s = s 

        self.optimizer = torch.optim.Adam(self.miss_model.gen_model.parameters(), lr=self.lr)

    def learn_missingness_mech(
        self,
        data: np.ndarray,
        R: np.ndarray, 
        intervention_mask: np.ndarray,
        C=0.5
    ):
        self.miss_model.learn_miss_mech_separately(
            data, 
            R, 
            intervention_mask, 
            C,
            penalty="l1",
            solver="saga",
            maxiter=100
        )

    def train(
        self,
        data: Dataset,
        print_loss=True,
        print_interval=50,
        data_missing=True,
        min_accept_factor=0.5
    ):

        logpx_obs_list = list()
        iteration_count = list()
        acceptance_rate_list = list()
        repetitions_list = list()
        # k_list_interations = list()

        training_dataloader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for epoch in range(self.max_epochs):
            av_loss = 0
            logpx_obs = 0
            count = 0
            av_accepted_samples = 0
            repetitions = 0
            for it, batch in enumerate(training_dataloader):

                self.optimizer.zero_grad()

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

                # R = R[:len(X)]
                # intervention_mask = intervention_mask[:len(X)]

                if self.obs: 
                    loss_pen, nll, _, _ = self.miss_model.gen_model.losses(
                        x = X,
                        intervention_mask=intervention_mask,
                        lambda_c=self.lambda_c,
                        lambda_dag=self.lambda_dag,
                        obs=self.obs
                    )
                else:
                    loss_pen, nll, _ = self.miss_model.gen_model.losses(
                        x = X,
                        intervention_mask=intervention_mask,
                        lambda_c=self.lambda_c,
                        lambda_dag=self.lambda_dag,
                        obs=self.obs,
                        s=self.s, 
                        method=self.dag_method
                    )
                av_loss += loss_pen.item()
                logpx_obs += -nll.item()
                count += 1

                loss_pen.backward()
                self.optimizer.step()
                update_lipschitz(self.miss_model.gen_model, n_iterations=self.n_lip_iters)

                if print_loss:
                    av_loss /= count
                    av_accepted_samples /= count
                    logpx_obs_list.append(logpx_obs/count)
                    acceptance_rate_list.append(av_accepted_samples / len(X_miss))
                    iteration_count.append(epoch * len(training_dataloader) + count)
                    repetitions_list.append(repetitions / count)
                    print("Epoch: {}/{}, Iter: {}/{}, Loss: {}, Acceptance rate (av): {:.2f}".format(epoch+1, self.max_epochs, it, len(training_dataloader), av_loss, av_accepted_samples), end="\r", flush=True)

                    count = 0
                    av_loss = 0
                    logpx_obs = 0
                    av_accepted_samples = 0
                    repetitions = 0

        return logpx_obs_list, iteration_count, acceptance_rate_list, repetitions_list
