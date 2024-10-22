import numpy as np
import argparse 
import os
import torch 
from torch.utils.data import DataLoader
import networkx as nx

import pytorch_lightning as pl

from datagen.missingData import mgraph, missing_value_dataset
from datagen.structuralModels import linearSEM

from models.nodags.missing_data_model import missModel
from models.nodags.functions import gumbelSoftMLP
from models.nodags.resblock import iResBlock
from models.nodags.trainer import Trainer

from baselines.imputation import *

chosen_genes = [2, 12, 13, 32, 33, 39, 49, 50, 54, 55]

def get_dataset(data_path, missing_prob, impute_method, keep_missing=True):

    intervention_sets = np.load(os.path.join(data_path, "intervention_sets.npy"))
    datasets = [
        np.load(os.path.join(data_path, "dataset_{}.npy".format(i)))[:, chosen_genes] for i in range(len(intervention_sets))
    ]
    
    datasets = [datasets[i] for i in chosen_genes]
    intervention_sets = [[i] for i, _ in enumerate(chosen_genes)]

    training_interventions = intervention_sets[:-2]
    training_datasets = datasets[:-2]

    testing_datasets = datasets[-2:]
    testing_interventions = intervention_sets[-2:]

    scaling_data = np.vstack(training_datasets)[:1000]

    dummy_graph = nx.DiGraph()
    dummy_graph.add_nodes_from(range(scaling_data.shape[1]))

    miss_graph = mgraph(
        obs_graph=dummy_graph,
        sem=linearSEM(graph=dummy_graph),
        missing_model='obs-only',
        p=missing_prob,
        is_mcar=False, 
        max_child=2,
        max_variance=2.0,
        scaling_data_given=True, 
        scaling_data=scaling_data
    )

    missing_datasets = list()
    missing_masks = list()

    for dataset, targets in zip(training_datasets, training_interventions):
        missing_mask, dataset_missing = miss_graph.generatemDataFromSamples(
            X=dataset, 
            intervention_set=targets
        )

        missing_datasets.append(dataset_missing)
        missing_masks.append(missing_mask)

    final_dataset = np.vstack(tuple(missing_datasets))
    final_missing_masks = np.vstack(tuple(missing_masks))

    if impute_method == "missforest":
        imp_dataset = missforest_impute(final_dataset, final_missing_masks)
    elif impute_method == "mean":
        imp_dataset = mean_impute(final_dataset, final_missing_masks)
    elif impute_method == "optransport":
        imp_dataset = OT_impute(final_dataset, final_missing_masks)

    final_datasets = list()
    run_sum = 0
    for dataset in training_datasets:
        final_datasets.append(
            imp_dataset[run_sum : run_sum + len(dataset)]
        )

        run_sum += len(dataset)

    interventional_datasets = list()

    for dataset, dataset_missing, missing_mask in zip(training_datasets, final_datasets, missing_masks):
        
        interventional_datasets.append(
            (dataset, missing_mask, dataset_missing)
        )

    training_dataset = missing_value_dataset(interventional_datasets, training_interventions)

    return training_dataset, (testing_interventions, testing_datasets)

def get_checkpoint_path(data_path, missing_prob, impute_method):
    
    missing_label = int(100 * missing_prob)
    checkpoint_path = os.path.join(data_path, "{}_{}".format(missing_label, impute_method))
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    return checkpoint_path

def learn_graph(datasets,
                val_dataset,
                checkpoints_path,
                lr, 
                lambda_c, 
                act_fun='none',
                max_epochs=200,
                batch_size=1024
                ):

    causal_mech = gumbelSoftMLP(
        n_nodes=10,
        lip_constant=0.9,
        activation="relu"
    )

    nodags = iResBlock(
        func=causal_mech,
        n_power_series=None,
        precondition=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nodags = nodags.to(device) 

    miss_model = missModel(nodags, is_mcar=False)

    model_trainer = Trainer(
        miss_model=miss_model,
        lr=lr,
        lambda_c=lambda_c,
        max_epochs=max_epochs,
        batch_size=batch_size
    )

    log_px, iter_count, acceptance_rate, repetitions_list = model_trainer.train(
        data=datasets,
        print_loss=False, 
        print_interval=100,
        data_missing=False,
        min_accept_factor=0.5
    )


    test_X = np.vstack(val_dataset[1])
    test_inter_mask = np.ones_like(test_X)

    for targets, dataset in zip(val_dataset[0], val_dataset[1]):
        test_inter_mask[:len(dataset), targets[0]] = 0

    _, nll, _ = nodags.losses(
        x=torch.tensor(test_X).float().cuda(),
        intervention_mask=torch.tensor(test_inter_mask).float().cuda(),
        neumann_grad=False
    )

    with open(os.path.join(checkpoints_path, "val_nll.txt"), 'w') as file:
        file.write("{}".format(nll/10))
    
    print("Val nll: {}".format(nll/10))    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', type=str, required=True, help='data input path')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--lc', type=float, default=1e-2, help='Regularization constant')
    parser.add_argument('--act-fun', type=str, default='none', choices=['none', 'relu', 'tanh'], help='Activation function')
    parser.add_argument('--prob', type=float, default=0.1, help='Missing probability')
    parser.add_argument('--impute-method', type=str, choices=["missforest", "mean", "optransport"], default="missforest")
    parser.add_argument('--max-epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size for stochastic gradients')

    args = parser.parse_args()

    print("Loading the data")
    dataset, val_dataset = get_dataset(args.d, args.prob, args.impute_method)
    checkpoint_path = get_checkpoint_path(args.d, args.prob, args.impute_method)

    print("Learning the graph from data")
    learn_graph(
        datasets=dataset,
        val_dataset=val_dataset, 
        checkpoints_path=checkpoint_path,
        lr=args.lr, 
        lambda_c=args.lc, 
        act_fun=args.act_fun,
        max_epochs=args.max_epochs,
        batch_size=args.bs, 
    )