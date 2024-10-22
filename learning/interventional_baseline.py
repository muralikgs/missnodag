from torch.utils.data import DataLoader
import torch
import networkx as nx
import numpy as np
import os 
import argparse
import pytorch_lightning as pl

from datagen.missingData import mgraph, missing_value_dataset
from datagen.structuralModels import linearSEM 

from baselines.imputation import *

from models.nodags.functions import gumbelSoftMLP
from models.nodags.resblock import iResBlock
from models.nodags.missing_data_model import missModel
from models.nodags.trainer import Trainer 


def get_dataset(data_path, n_nodes, n_samples, missing_prob, impute_method, is_mcar=False, max_parents=2):

    # load the interventional targets
    intervention_set_path = os.path.join(data_path, "intervention-sets.npy")
    intervention_sets = np.load(intervention_set_path)

    # load the clean and dataset and simulate the missingness mechanism
    missing_datasets = list()
    missing_masks = list()
    for i, targets in enumerate(intervention_sets):
        dataset_path = os.path.join(data_path, "dataset-{}.npy".format(i))
        dataset = np.load(dataset_path)
        
        # we only care about the connections between x-nodes and r-nodes as the
        # data samples for x-nodes are already generated. Hence we create a dummy 
        # nx.DiGraph object to feed as input to mgraph()
        dummy_graph = nx.DiGraph()
        dummy_graph.add_nodes_from(range(n_nodes))
        miss_graph = mgraph(
            obs_graph=dummy_graph,
            sem=linearSEM(graph=dummy_graph),
            missing_model='obs-only', 
            p=missing_prob,
            is_mcar=is_mcar,
            max_child=max_parents
        )

        # generate the missing data
        missing_mask, dataset_missing = miss_graph.generatemDataFromSamples(
            X=dataset[:n_samples],
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

    final_datasets = [imp_dataset[i*n_samples:(i+1)*n_samples] for i in range(len(intervention_sets))]
    
    intervention_datasets = [
        (final_datasets[i], missing_masks[i], missing_datasets[i]) for i in range(len(intervention_sets))
    ]

    return missing_value_dataset(intervention_datasets, intervention_sets)


def get_checkpoint_path(data_path, missing_prob, n_samples, max_parents, impute_method, missing_type='mcar', var_samples_exp=False, var_miss_sparse_exp=False):
    
    if var_samples_exp: 
        missing_label = n_samples
    elif var_miss_sparse_exp:
        missing_label = max_parents
    else:
        missing_label = int(100 * missing_prob)

    if var_samples_exp:
        checkpoint_path = os.path.join(data_path, "{}_samples_{}_{}".format(missing_label, impute_method, missing_type))
    elif var_miss_sparse_exp:
        checkpoint_path = os.path.join(data_path, "{}_parents_{}_{}".format(missing_label, impute_method, missing_type))
    else:
        checkpoint_path = os.path.join(data_path, "{}_4_parents_{}_{}".format(missing_label, impute_method, missing_type))
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    return checkpoint_path

def learn_graph(n_nodes, 
                inter_datasets,
                checkpoints_path,
                lr, 
                lambda_c, 
                act_fun='none',
                max_epochs=100,
                batch_size=512,
                print_loss=False
                ):
    
    causal_mech = gumbelSoftMLP(
        n_nodes=n_nodes, 
        lip_constant=0.9,
        activation=act_fun
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
        batch_size=batch_size,
    )

    _, _, _, _ = model_trainer.train(
        data=inter_datasets,
        print_loss=print_loss, 
        data_missing=False
    )

    print()

    miss_model.save_model(checkpoints_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', type=str, required=True, help='data input path')
    parser.add_argument('--n-nodes', type=int, default=20, help='Number of nodes in the graph')
    parser.add_argument('--n-samples', type=int, default=500, help='Number of samples used per dataset')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--lc', type=float, default=1e-2, help='Regularization constant')
    parser.add_argument('--act-fun', type=str, default='none', choices=['none', 'relu', 'tanh'], help='Activation function')
    parser.add_argument('--prob', type=float, default=0.1, help='Missing probability')
    parser.add_argument('--impute-method', type=str, choices=["missforest", "mean", "optransport"], default="missforest")
    parser.add_argument('--max-epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size for stochastic gradients')
    parser.add_argument('--missing-type', type=str, default='mcar', choices=['mcar', 'mnar'])
    parser.add_argument('-p', action="store_true", default=False)
    parser.add_argument("--max-parents", type=int, default=2, help="Max number of parents for each missingness indicator")

    args = parser.parse_args()

    is_mcar = True if args.missing_type == 'mcar' else False

    print("Loading the data")
    dataset = get_dataset(args.d, args.n_nodes, args.n_samples, args.prob, args.impute_method, is_mcar, max_parents=args.max_parents)
    checkpoint_path = get_checkpoint_path(
        args.d,
        args.prob, 
        args.n_samples,
        args.max_parents,
        args.impute_method, 
        args.missing_type,
        var_samples_exp=False
    )

    print("Learning the graph from data")
    learn_graph(
        n_nodes=args.n_nodes,
        inter_datasets=dataset,
        checkpoints_path=checkpoint_path,
        lr=args.lr,
        lambda_c=args.lc,
        max_epochs=args.max_epochs, 
        batch_size=args.bs,
        print_loss=args.p
    )
