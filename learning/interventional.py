from torch.utils.data import DataLoader
import torch
import numpy as np
import networkx as nx 
import os 
import argparse

from datagen.missingData import mgraph, missing_value_dataset
from datagen.structuralModels import linearSEM

from models.nodags.functions import gumbelSoftMLP
from models.nodags.resblock import iResBlock 
from models.nodags.missing_data_model import missModel 
from models.nodags.trainer import Trainer 

def get_dataset(data_path, n_nodes, n_samples, missing_prob, is_mcar=False, max_parents=2, n_inter=10):

    # load the interventional targets
    intervention_set_path = os.path.join(data_path, "intervention-sets.npy")
    intervention_sets = np.load(intervention_set_path)[:n_inter]

    # load the clean and dataset and simulate the missingness mechanism
    intervention_datasets = list()
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

        intervention_datasets.append(
            (dataset[:n_samples], missing_mask, dataset_missing)
        )

    obs_dataset_data = np.load(
        os.path.join(data_path, "obs-dataset.npy")
    )

    obs_missing_mask, obs_dataset_missing = miss_graph.generatemDataFromSamples(
        X=obs_dataset_data,
        intervention_set=[None]
    )

    return missing_value_dataset(intervention_datasets, intervention_sets), miss_graph.m_coefs

def get_checkpoint_path(data_path, missing_prob, n_samples, max_parents, n_inter, is_missing, missing_type='mcar', var_samples_exp=False, var_miss_sparse_exp=False, var_inter_exp=False):
    
    if var_samples_exp:
        missing_label = n_samples 
    elif var_miss_sparse_exp:
        missing_label = max_parents
    elif var_inter_exp:
        missing_label = n_inter
    else:
        missing_label = int(100 * missing_prob)
    missing_string = "missing" if is_missing else "not_missing"

    if var_samples_exp:
        checkpoint_path = os.path.join(data_path, "{}_samples_{}_{}".format(missing_label, missing_string, missing_type))
    elif var_miss_sparse_exp:
        checkpoint_path = os.path.join(data_path, "{}_parents_{}_{}".format(missing_label, missing_string, missing_type))
    elif var_inter_exp:
        checkpoint_path = os.path.join(data_path, "{}_inters_{}_{}_{}".format(missing_label, int(100 * missing_prob), missing_string, missing_type))
    else:
        checkpoint_path = os.path.join(data_path, "{}_4_parents_{}_{}".format(missing_label, missing_string, missing_type))
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    return checkpoint_path

def learn_graph(n_nodes, 
                inter_datasets,
                gt_miss_weights,
                checkpoints_path,
                lr, 
                lambda_c, 
                data_missing=True, 
                act_fun='none',
                max_epochs=100,
                batch_size=512,
                is_mcar=False,
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

    miss_model = missModel(nodags, is_mcar=is_mcar)

    model_trainer = Trainer(
        miss_model=miss_model,
        lr=lr,
        lambda_c=lambda_c, 
        max_epochs=max_epochs, 
        batch_size=batch_size
    )

    inter_X = inter_datasets.X_miss
    inter_R = inter_datasets.R 
    inter_masks = inter_datasets.intervention_mask 

    model_trainer.learn_missingness_mech( 
        data=inter_X,
        R=inter_R,
        intervention_mask=inter_masks,
        C=0.7
    )

    _, _, _, _ = model_trainer.train(
        data=inter_datasets,
        print_loss=print_loss,
        print_interval=100,
        data_missing=data_missing
    )

    print()

    miss_model.save_model(checkpoints_path)
    np.save(os.path.join(checkpoints_path, "gt_miss_weights.npy"), gt_miss_weights)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', type=str, required=True, help='data input path')
    parser.add_argument('--n-nodes', type=int, default=20, help='Number of nodes in the graph')
    parser.add_argument('--n-samples', type=int, default=500, help='Number of samples used per dataset')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--lc', type=float, default=1e-2, help='Regularization constant')
    parser.add_argument('--missing', action='store_true', default=False, help='If set then some data samples are corrupted')
    parser.add_argument('--act-fun', type=str, default='none', choices=['none', 'relu', 'tanh'], help='Activation function')
    parser.add_argument('--prob', type=float, default=0.1, help='Missing probability')
    parser.add_argument('--max-epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size for stochastic gradients')
    parser.add_argument('--missing-type', type=str, default='mcar', choices=['mcar', 'mnar'], help='Type of missingness mechanism')
    parser.add_argument('-p', action='store_true', default=False)
    parser.add_argument("--max-parents", type=int, default=2, help="Maximum parents for each missingness indicator")
    parser.add_argument("--n-inter", type=int, default=10, help="Number of interventions used for training")

    args = parser.parse_args()

    is_mcar = False

    print("Loading the data")
    dataset, gt_miss_weights = get_dataset(args.d, args.n_nodes, args.n_samples, args.prob, is_mcar, args.max_parents, args.n_inter)
    checkpoint_path = get_checkpoint_path(
        args.d, 
        args.prob, 
        args.n_samples,
        args.max_parents, 
        args.n_inter,
        args.missing, 
        missing_type=args.missing_type,
        var_samples_exp=False,
        var_miss_sparse_exp=False,
        var_inter_exp=True
    )

    print("Learning the graph from data")
    learn_graph(
        n_nodes=args.n_nodes,
        inter_datasets=dataset,
        gt_miss_weights=gt_miss_weights,
        checkpoints_path=checkpoint_path,
        lr=args.lr,
        lambda_c=args.lc,
        data_missing=args.missing,
        act_fun=args.act_fun,
        max_epochs=args.max_epochs, 
        batch_size=args.bs,
        is_mcar=is_mcar,
        print_loss=args.p
    )
