import os 
import torch 
import argparse 

import numpy as np
import networkx as nx

from datagen.missingData import mgraph, missing_value_dataset
from datagen.structuralModels import linearSEM

from models.nodags.functions import gumbelSoftMLP
from models.nodags.resblock import iResBlock 
from models.nodags.missing_data_model import missModel 
from models.nodags.trainer import Trainer 

from baselines.imputation import *

def get_dataset(data_path, n_nodes, n_samples, missing_prob, impute_method, is_mcar=False, max_parents=2):

    data_file_path = os.path.join(data_path, "obs-dataset.npy")
    obs_data = np.load(data_file_path)

    dummy_graph = nx.DiGraph()
    dummy_graph.add_nodes_from(range(n_nodes))
    miss_graph = mgraph(
        obs_graph=dummy_graph,
        sem=linearSEM(graph=dummy_graph),
        missing_model="obs-only",
        p=missing_prob,
        is_mcar=is_mcar,
        max_child=max_parents
    )

    # generate the missing data
    missing_mask, dataset_missing = miss_graph.generatemDataFromSamples(
        X=obs_data[:n_samples],
        intervention_set=[None]
    )

    training_dataset = missing_value_dataset([(obs_data[:n_samples], missing_mask, dataset_missing)], [[None]])
    
    if impute_method == "missforest":
        imp_dataset = missforest_impute(training_dataset.X_miss, training_dataset.R)
    elif impute_method == "mean":
        imp_dataset = mean_impute(training_dataset.X_miss, training_dataset.R)
    elif impute_method == 'optransport':
        imp_dataset = OT_impute(training_dataset.X_miss, training_dataset.R)

    training_dataset.X_clean = imp_dataset

    return training_dataset

def get_checkpoint_path(data_path, missing_prob, impute_method, missing_type='mnar'):
    
    missing_label = int(100 * missing_prob)
    checkpoint_path = os.path.join(data_path, "{}_{}_{}".format(missing_label, impute_method, missing_type))
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    return checkpoint_path

def learn_graph(n_nodes, 
                dataset,
                checkpoints_path,
                lr, 
                lambda_c, 
                lambda_dag,
                s=1, 
                method='expm',
                act_fun='none',
                max_epochs=200,
                is_mcar=False, 
                batch_size=1024,
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
        obs=True, 
        lr=lr,
        lambda_c=lambda_c,
        lambda_dag=lambda_dag,
        max_epochs=max_epochs,
        batch_size=batch_size,
        n_lip_iters=5,
        s=s, 
        dag_method=method
    )
    _, _, _, _ = model_trainer.train(
        data=dataset, 
        print_loss=print_loss,
        print_interval=100,
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
    parser.add_argument('--ldag', type=float, default=1, help='Reg. constant for DAG constraint')
    parser.add_argument('--act-fun', type=str, default='none', choices=['none', 'relu', 'tanh'], help='Activation function')
    parser.add_argument('--prob', type=float, default=0.1, help='Missing probability')
    parser.add_argument('--s', type=float, default=1)
    parser.add_argument('--impute-method', type=str, choices=["missforest", "mean", "optransport"], default="missforest")
    parser.add_argument('--method', type=str, default='expm', choices=['expm', 'log-det'], help='DAG constraint to use')
    parser.add_argument('--max-epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size for stochastic gradients')
    parser.add_argument("--missing-type", type=str, default="mnar", choices=["mnar", "mcar"], help='Type of Missingness mechanism')
    parser.add_argument('--max-parents', type=int, default=2, help="Maximum parents for each missingness indicator")
    parser.add_argument('-p', action="store_true", default=False, help="Use this flag to print loss during training")
    args = parser.parse_args()

    print("Loading the data")
    dataset = get_dataset(
        data_path=args.d, 
        n_nodes=args.n_nodes, 
        n_samples=args.n_samples, 
        missing_prob=args.prob, 
        impute_method=args.impute_method, 
        is_mcar=False, 
        max_parents=args.max_parents)
    
    checkpoint_path = get_checkpoint_path(
        data_path=args.d, 
        missing_prob=args.prob, 
        impute_method=args.impute_method, 
        missing_type=args.missing_type
    )

    print("Learning the graph from data")
    learn_graph(
        n_nodes=args.n_nodes, 
        dataset=dataset,
        checkpoints_path=checkpoint_path,
        lr=args.lr, 
        lambda_c=args.lc, 
        lambda_dag=args.ldag,
        s=args.s, 
        method=args.method,
        act_fun=args.act_fun,
        max_epochs=args.max_epochs,
        is_mcar=False, 
        batch_size=args.bs,
        print_loss=args.p
    )
