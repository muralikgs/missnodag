import numpy as np 
import os 
import argparse 
import yaml 
import uuid
import pandas as pd

from time import time 
from tqdm import tqdm

from datagen.missingData import mgraph, missing_value_dataset
from datagen.structuralModels import SEM 

from models.nodags.functions import gumbelSoftMLP
from models.nodags.resblock import iResBlock

from models.nodags.missingness_mechanism import *
from models.nodags.missing_data_model import missModel 
from models.nodags.trainer import Trainer

from baselines.imputation import *
from baselines.enco.causal_graphs.graph_definition import CausalDAGDataset
from baselines.enco.causal_discovery.enco import ENCO

from utils.error_metrics import * 

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import mv_fisherz
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode 

def gen_missing_data(intervention_sets, datasets, data_config, impute=False, impute_method='optransport'):

    dummy_graph = nx.DiGraph()
    dummy_graph.add_nodes_from(range(data_config["n_nodes"]))
    miss_graph = mgraph(
        obs_graph=dummy_graph,
        sem=SEM(graph=dummy_graph),
        scaling_data_given=True, 
        scaling_data = datasets[0],
        missing_model=data_config["missing-mech"], 
        p=data_config["missing-prob"],
        is_mcar=False,
        max_child=2,
        mlp=data_config["mlp"]
    )
    
    missing_datasets = list()
    for dataset, targets in zip(datasets, intervention_sets):
        # generate the missing data
        missing_mask, dataset_missing = miss_graph.generatemDataFromSamples(
            X=dataset[:data_config["n_samples_per_intervention"]],
            intervention_set=targets
        )

        missing_datasets.append(
            (dataset[:data_config['n_samples_per_intervention']], missing_mask, dataset_missing)
        )
    
    if impute:
        final_dataset = np.vstack( tuple([data[2] for data in missing_datasets]) )
        final_missing_masks = np.vstack( tuple(data[1] for data in missing_datasets) )

        if impute_method == "missforest":
            imp_dataset = missforest_impute(final_dataset, final_missing_masks)
        elif impute_method == "mean":
            imp_dataset = mean_impute(final_dataset, final_missing_masks)
        else:
            imp_dataset = OT_impute(final_dataset, final_missing_masks)

        n_samples = data_config["n_samples_per_intervention"]
        final_datasets = [imp_dataset[i*n_samples:(i+1)*n_samples] for i in range(len(intervention_sets))]

        intervention_datasets = [
            (final_datasets[i], missing_datasets[i][1], missing_datasets[i][2]) for i in range(len(intervention_sets))
        ]
    
        return missing_value_dataset(intervention_datasets, intervention_sets), (miss_graph.m_coefs, miss_graph.m_coefs_r2r)

    return missing_value_dataset(missing_datasets, intervention_sets), (miss_graph.m_coefs, miss_graph.m_coefs_r2r)

def train_model(data_config, model_config, datasets, model_choice="missnodag"):
    """
    Wrapper to dispatch training based on model_choice.
    Extra keyword args are forwarded only to the selected trainer.
    """
    if model_choice == "missnodag":
        return train_missnodag(data_config, model_config, datasets)
    elif model_choice in ["optransport", "missforest", "mean"]:
        return train_nodags(data_config, model_config, datasets)
    elif model_choice == "enco":
        return train_enco(data_config, model_config, datasets) 
    elif model_choice == "mvpc":
        return train_jci_mvpc(data_config, model_config, datasets)
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

def train_missnodag(data_config, model_config, datasets):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    causal_mech = gumbelSoftMLP(
        n_nodes=data_config["n_nodes"], 
        lip_constant=0.9,
        activation=model_config["activation"]
    )

    nodags = iResBlock(
        func=causal_mech,
        n_power_series=None
    )
    nodags = nodags.to(device)

    missing_mech = identifiableMNAR(
        n_nodes=data_config["n_nodes"],
        learning_method="em",
        constraint_optimization="aug-lagrangian",
        C=model_config["missing-mech-sparsity-reg"]
    )
    missing_mech = missing_mech.to(device)

    miss_model = missModel(nodags, missing_mech=missing_mech, is_mcar=False)

    model_trainer = Trainer(
        miss_model=miss_model,
        lr=model_config["lr"],
        lr_miss=model_config["lr"],
        lambda_c=model_config["lc"], 
        max_epochs=model_config["max_epochs"], 
        batch_size=model_config["batch_size"],
        missingness_learn="em",
        lambda_dag=model_config["ldag"],
        lambda_nc=model_config["lnc"],
        obs=False
    )

    start = time()

    training_failed = False
    training_error = None
    try:
        _ = model_trainer.train(
            data=datasets,
            print_loss=False,
            print_interval=100,
            data_missing=True,
            min_accept_factor=0.5,
            store_grad_norm=False        
        )
    except Exception as e:
        training_failed = True
        training_error = str(e)

    stop = time()

    return (
        miss_model.gen_model.get_w_adj(), 
        miss_model.missing_mech.coefs_x2r.detach().cpu().numpy(), 
        miss_model.missing_mech.coefs_r2r.detach().cpu().numpy(), 
        stop-start, training_failed, training_error
    )

def train_enco(data_config, model_config, datasets):

    # the missing values are imputed using optransport imputation algorithm
    
    obs_data = datasets.intervention_datasets[0][0]
        
    num_vars = obs_data.shape[1]
    num_samples = obs_data.shape[0]
    
    data_int = np.zeros((num_vars, num_samples, num_vars), dtype=np.float32)
    intervened_vars = []
    
    # Loop through intervention datasets to populate data_int
    for i, targets in enumerate(datasets.intervention_targets):
        tgt = targets[0]
        data_m = datasets.intervention_datasets[i][0]
        
        if tgt is None:
            continue
            
        target_var = int(tgt)
        intervened_vars.append(target_var)
        # Ensure we only take as many samples as initialized
        n_s = min(num_samples, data_m.shape[0])
        data_int[target_var, :n_s, :] = data_m[:n_s]
        
    all_vars = set(range(num_vars))
    exclude_inters = sorted(list(all_vars - set(intervened_vars)))
    
    dataset_obj = CausalDAGDataset(
        adj_matrix=np.zeros((num_vars, num_vars)),
        data_obs=obs_data,
        data_int=data_int,
        exclude_inters=exclude_inters
    )
    
    start = time()
    training_failed = False
    training_error = None
    
    try:
        model = ENCO(
            graph=dataset_obj, 
            sample_size_inters=num_samples, 
            sample_size_obs=num_samples,
        )
                
        if torch.cuda.is_available():
             model.to(torch.device("cuda"))
             
        est_adj_matrix = model.discover_graph(num_epochs=10)
        est_graph = est_adj_matrix.cpu().numpy()
        
    except Exception as e:
        training_failed = True
        training_error = str(e)
        est_graph = np.zeros((num_vars, num_vars))

    stop = time()
    
    # ENCO does not learn missingness mechanism coefficients, return zeros
    return (
        est_graph, 
        np.zeros((num_vars, num_vars)), # x2r
        np.zeros((num_vars, num_vars)), # r2r
        stop-start, training_failed, training_error
    )

def train_nodags(data_config, model_config, datasets):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    causal_mech = gumbelSoftMLP(
        n_nodes=data_config["n_nodes"], 
        lip_constant=0.9,
        activation=model_config["activation"]
    )

    nodags = iResBlock(
        func=causal_mech,
        n_power_series=None
    )
    nodags = nodags.to(device)

    missing_mech = identifiableMNAR(n_nodes=data_config["n_nodes"])
    
    miss_model = missModel(nodags, missing_mech=missing_mech, is_mcar=False)

    model_trainer = Trainer(
        miss_model=miss_model,
        lr=model_config["lr"],
        lr_miss=model_config["lr"],
        lambda_c=model_config["lc"], 
        max_epochs=model_config["max_epochs"], 
        batch_size=model_config["batch_size"],
        missingness_learn="initial",
        obs=False
    )

    start = time()

    training_failed = False
    training_error = None
    try:
        _ = model_trainer.train(
            data=datasets,
            print_loss=False,
            print_interval=100,
            data_missing=False,
            store_grad_norm=False        
        )
    except Exception as e:
        training_failed = True
        training_error = str(e)

    stop = time()

    return (
        miss_model.gen_model.get_w_adj(), 
        np.zeros_like(miss_model.missing_mech.coefs_x2r.detach().cpu().numpy()), 
        np.zeros_like(miss_model.missing_mech.coefs_r2r.detach().cpu().numpy()), 
        stop-start, training_failed, training_error
    )

def train_jci_mvpc(data_config, model_config, datasets):

    # --- STEP 1: DEFINE YOUR METADATA ---
    # Number of actual data variables
    num_vars = data_config["n_nodes"]

    # Dictionary mapping the "Intervention Column Index" to "Target Variable Indices"
    n_interventional_settings = num_vars if data_config["interventions"] == -1 else data_config["interventions"]    

    # Stack all the interventional data
    int_datasets = tuple([datasets[i][2] for i in range(n_interventional_settings+1)])
    int_datasets = np.vstack(int_datasets)
    data = int_datasets

    # Initialize the background knowledge
    bk = BackgroundKnowledge()

    total_cols = num_vars

    if n_interventional_settings > 0:

        interventions = {i+num_vars: [i] for i in range(n_interventional_settings-1)}

        # Total columns in your pooled dataset (Data + Contexts)
        total_cols += len(interventions) 

        # --- STEP 2: CONFIGURE BACKGROUND KNOWLEDGE ---
    
        nodes = [GraphNode(f"X{i+1}") for i in range(total_cols)]

        # We iterate through our known intervention columns
        for context_idx, target_indices in interventions.items():
            
            # A. The Context Variable is Exogenous
            # Nothing causes the intervention (forbid X -> Context)
            for i in range(total_cols):
                if i != context_idx:
                    bk.add_forbidden_by_node(nodes[i], nodes[context_idx])

            # B. Handle Targets (The "Known" Part)
            for var_idx in range(num_vars):
                if var_idx in target_indices:
                    # FORCE edge: Context -> Target
                    # This effectively tells PC: "Don't test this, we know it's true."
                    bk.add_required_by_node(nodes[context_idx], nodes[var_idx])
                else:
                    # FORBID edge: Context -> Non-Target
                    # This assumes perfect interventions (no side effects on other vars)
                    bk.add_forbidden_by_node(nodes[context_idx], nodes[var_idx])

        n_samples = data_config["n_samples_per_intervention"]
        X_w_context = np.zeros((data.shape[0], total_cols))
        X_w_context[:, :data_config["n_nodes"]] = data

        for i in range(n_interventional_settings):
            if i == 0: 
                continue
            
            X_w_context[i*n_samples:(i+1)*n_samples, i+data_config["n_nodes"]-1] = 1

        
    else:
        X_w_context = data


    # --- STEP 3: RUN MVPC ---
    start = time()
    training_failed = False
    training_error = None
    try:
        cg = pc(
            X_w_context,
            alpha=0.05,
            indep_test=mv_fisherz,
            mvpc=True,
            background_knowledge=bk
        )
        # Adjacency-style matrix used internally by causallearn (encodes edge endpoints)
        adj = cg.G.graph  # shape: (total_cols, total_cols)

    except Exception as e:
        training_failed = True
        training_error = str(e)
        adj = np.zeros((total_cols, total_cols))
    
    stop = time()

    return (
        adj, 
        np.zeros((num_vars, num_vars)), 
        np.zeros((num_vars, num_vars)), 
        stop-start, training_failed, training_error
    )

def get_metrics(gt_params, est_params, adj_thresh=0.7, mm_thresh=0.2, model_choice="missnodag"):

    gt_adjacency, gt_miss_mech = gt_params 
    est_adjacency, est_miss_mech = est_params 

    gt_x2r_coefs, gt_r2r_coefs = gt_miss_mech
    est_x2r_coefs, est_r2r_coefs = est_miss_mech

    # target law metrics
    if model_choice == "mvpc":
        shd_tl, _ = compute_shd_pc((np.abs(gt_adjacency) > 0)*1.0, est_adjacency.copy())
    else:
        shd_tl, _ = compute_shd(np.abs(gt_adjacency) > 0, est_adjacency > adj_thresh)

    # missingness mechanism metrics
    shd_x2r, _ = compute_shd(np.abs(gt_x2r_coefs) > mm_thresh, np.abs(est_x2r_coefs) > mm_thresh)
    shd_r2r, _ = compute_shd(np.abs(gt_r2r_coefs) > mm_thresh, np.abs(est_r2r_coefs) > mm_thresh)

    return {
        "shd_tl" : shd_tl,
        "shd_x2r" : shd_x2r,
        "shd_r2r" : shd_r2r
    }

def create_table_row(config, trial_id, metrics, model_choice):

    row = {
        "benchmark_name" : config["name"],
        "setting" : config["setting"],
        "trial_id" : trial_id,
        "run_id" : str(uuid.uuid4()),
        "method" : model_choice
    }

    for key, val in metrics.items():
        row[key] = val

    return row 

def run_setting(data_root_dir, benchmark_root_dir, n_trials=10, verbose=False, model_choice="missnodag", impute_method="optransport"):
    
    with open(os.path.join(data_root_dir, "settings.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    model_config = config["model"]
    data_config = config["data"]

    parquet_path = os.path.join(benchmark_root_dir, f"results.{config['name']}.{config['setting']}.{model_choice}.parquet")
    results_df = pd.DataFrame()

    for trial in range(n_trials):
        print(f"Trial: {trial+1}/{n_trials}")
        artifacts_dir = os.path.join(data_root_dir, f"trials/trial-{trial}/artifacts")

        # load the GT SCM parameters        
        scm = np.load(os.path.join(artifacts_dir, "scm_true.npz"))
        weights = scm["adjacency"]

        # load the data files
        data_files = np.load(os.path.join(artifacts_dir, "train_datasets.npz"), allow_pickle=True)
        targets = [[target.item()] for target in data_files["targets"]]
        datasets = [data_files[f"data_{i}"] for i, _ in enumerate(targets)]

        # generate the missing data
        missing_data, missing_mech = gen_missing_data(
            intervention_sets=targets,
            datasets=datasets, 
            data_config=data_config, 
            impute=model_choice not in ["missnodag", "mvpc"],
            impute_method=impute_method
        )

        # estimated graph, estimated covariance and total training time
        est_graph, est_x2r_coefs, est_r2r_coefs, train_time, training_failed, training_error = train_model(
            data_config, 
            model_config, 
            missing_data, 
            model_choice
        )

        # save the ground truth missingness mechanism
        np.savez(os.path.join(artifacts_dir, "missing_mech_true.npz"), x2r_coefs=missing_mech[0], r2r_coefs=missing_mech[1])
        
        # save the estimated graph and covariance
        np.savez(os.path.join(artifacts_dir, f"scm_est.{model_choice}.npz"), adjacency=est_graph, x2r_coefs=est_x2r_coefs, r2r_coefs=est_r2r_coefs)

        # evaluation
        err_metrics = get_metrics(
            (weights, missing_mech),
            (est_graph, (est_x2r_coefs, est_r2r_coefs)),
            adj_thresh=model_config["adj_threshold"],
            mm_thresh=model_config["mm_threshold"], 
            model_choice=model_choice
        )

        metrics = {
            "shd_tl" : err_metrics["shd_tl"],
            "shd_x2r" : err_metrics["shd_x2r"],
            "shd_r2r" : err_metrics["shd_r2r"],
            "training_time" : train_time,
            "training_failed" : training_failed,
            "training_error" : training_error if training_failed else ""
        }

        result_row = create_table_row(config, trial, metrics, model_choice)
        results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    
    results_df.to_parquet(parquet_path, engine="pyarrow")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", type=str, help='path to the settings file')
    ap.add_argument("--root_out", type=str, help="path to the root folder to store the results")
    ap.add_argument("--n_trials", type=int, default=10, help="number of repeated trials")
    ap.add_argument("--model", type=str, default="missnodag", choices=["missnodag", "mean", "optransport", "missforest", "enco", "mvpc"], help="Model to test")
    ap.add_argument("--verbose", action="store_true", default=False, help="Use the flag for printing the loss during training")
    
    args = ap.parse_args()
    
    settings_path = args.settings
    data_root_dir = os.path.dirname(settings_path)
    benchmark_root_dir = args.root_out
    verbose = args.verbose
    n_trials = args.n_trials
    model_choice = args.model
    
    impute_method = "optransport"
    if model_choice in ["mean", "optransport", "missforest"]:
        impute_method = model_choice
    
    run_setting(data_root_dir, benchmark_root_dir, n_trials, verbose, model_choice, impute_method)
