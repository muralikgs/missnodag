import numpy as np
import argparse 
import os
import torch 
from torch.utils.data import DataLoader

from datagen.missingData import incompleteDataset 

from models.nodags.layers.mlpLipschitz import linearLipschitz
from models.nodags.model_missing_data import NODAGSModel, impute_data_linear

def get_dataset(data_path, missing_prob, keep_missing=True):

    intervention_sets = np.load(os.path.join(data_path, "intervention_sets.npy"))
    datasets = [
        np.load(os.path.join(data_path, "dataset_{}.npy".format(i))) for i in range(len(intervention_sets))
    ]
    training_interventions = intervention_sets[:-6]
    training_datasets = datasets[:-6]

    validation_datasets = datasets[-6:]
    validation_interventions = intervention_sets[-6:]

    inter_datasets = list()
    for dataset, targets in zip(training_datasets, training_interventions):
        inter_datasets.append(
            incompleteDataset([dataset], [targets], missing_prob=missing_prob, keep_missing=keep_missing)
        )
    
    val_dataset = incompleteDataset(validation_datasets, validation_interventions, missing_prob=0.2, keep_missing=True)

    return inter_datasets, val_dataset

def get_checkpoint_path(data_path, missing_prob, missing):

    missing_label = int(100 * missing_prob)
    if missing:
        checkpoint_path = os.path.join(data_path, "{}_{}".format(missing_label, "True"))
    else:
        checkpoint_path = os.path.join(data_path, "{}_{}".format(missing_label, "False"))
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    return checkpoint_path

def update_lipschitz(model, n_iterations):
    for m in model.modules():
        if isinstance(m, linearLipschitz):
            m.compute_weight(update=True, n_iterations=n_iterations)

def learn_graph(
        datasets,
        val_dataset,
        checkpoints_path, 
        lr, 
        lambda_c, 
        data_missing=True, 
        act_fun='relu',
        max_epochs=200,
        batch_size=256, 
        lin_approx=False
):
    
    nodags = NODAGSModel(
        n_nodes = 61,
        act_fun = act_fun,
        fun_type='gst-mlp',
        lr = lr,
        lambda_c = lambda_c,
        precondition = True, 
        centered = True,
        obs = False, 
        data_missing = data_missing, 
        use_ground_truth = False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nodags_model = nodags.module
    nodags_model = nodags_model.to(device)

    inter_data_loader = list()
    for dataset in datasets:
        inter_data_loader.append(
            DataLoader(dataset, batch_size=batch_size, num_workers=4)
        )

    optimizer = torch.optim.Adam(nodags_model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        # av_loss_nll = 0
        # count = 0
        for i in range(len(datasets)):
            tot_it = len(inter_data_loader[i])
            for it in range(tot_it):
                update_lipschitz(nodags_model, n_iterations=5)
                optimizer.zero_grad()
                x_cor, masks, _, missing, x_clean = next(iter(inter_data_loader[i]))
                x_cor = x_cor.to(device)
                masks = masks.to(device)
                missing = missing.to(device)
                x_clean = x_clean.to(device)

                if data_missing:
                    x = impute_data_linear(x_cor, masks, missing, nodags_model, lin_approx=lin_approx)
                else:
                    x = x_clean 
                
                loss_pen, nll, _ = nodags_model.losses(
                    x, 
                    masks, 
                    lambda_c=lambda_c, 
                    obs=False
                )

                # av_loss_nll += nll
                # count += 1

                loss_pen.backward()
                optimizer.step()
        
        # av_loss_nll /= count 
                print("Epoch: {}/{}, Inter: {}/{}, NLL: {}".format(epoch+1, max_epochs, i+1, len(datasets), nll), end="\r", flush=True)
    
    print()

    # storing the trained model
    checkpoint_file_path = os.path.join(checkpoints_path, "model.pth")
    torch.save(nodags_model.state_dict(), checkpoint_file_path)

    # storing the validation nll
    _, nll, _ = nodags_model.losses(
        torch.tensor(val_dataset.data_clean).float().cuda(),
        torch.tensor(val_dataset.masks).float().cuda(), 
        neumann_grad=False
    )

    with open(os.path.join(checkpoints_path, "val_nll.txt"), 'w') as file:
        file.write("{}".format(nll/61))
    
    print("Val nll: {}".format(nll/61))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', type=str, required=True, help='data input path')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--lc', type=float, default=1e-2, help='Regularization constant')
    parser.add_argument('--missing', action='store_true', default=False, help='If set then some data samples are corrupted')
    parser.add_argument('--act-fun', type=str, default='none', choices=['none', 'relu', 'tanh'], help='Activation function')
    parser.add_argument('--prob', type=float, default=0.1, help='Missing probability')
    parser.add_argument('--keep-missing', action='store_true', default=False)
    parser.add_argument('--max-epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size for stochastic gradients')
    parser.add_argument('--lin-approx', action='store_true', default=False)

    args = parser.parse_args()

    print("Loading the data")
    dataset, val_dataset = get_dataset(args.d, args.prob, args.keep_missing)
    checkpoint_path = get_checkpoint_path(args.d, args.prob, args.missing)

    print("Learning the graph from data")
    learn_graph(
        datasets=dataset,
        val_dataset=val_dataset, 
        checkpoints_path=checkpoint_path,
        lr=args.lr, 
        lambda_c=args.lc, 
        act_fun=args.act_fun,
        data_missing=args.missing,
        max_epochs=args.max_epochs,
        batch_size=args.bs, 
        lin_approx=args.lin_approx
    )