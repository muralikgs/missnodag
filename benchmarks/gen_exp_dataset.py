import numpy as np 
import os 
import yaml 
import argparse 
from tqdm import tqdm
import random

from datagen.graph import DirectedGraphGenerator, cyclic_graph_generator
from datagen.structuralModels import SEM 

def generate_sem(
        n_nodes=10,
        expected_density=1,
        min_noise_scale=0.2,
        max_noise_scale=0.5,
        num_cycles="random",
        enforce_dag=False,
        contractive=True
    ) -> SEM:

    # Generating the graph for the given specs
    if num_cycles == "random":
        graph_gen = DirectedGraphGenerator(nodes=n_nodes, expected_density=expected_density, enforce_dag=enforce_dag)
        graph = graph_gen()
    else:
        graph = cyclic_graph_generator(n_nodes=n_nodes, expected_density=expected_density, num_cycles=num_cycles)

    sem = SEM(
        graph=graph,
        contractive=contractive,
        min_noise_scale=min_noise_scale,
        max_noise_scale=max_noise_scale
        )

    return sem

def generate_data_from_config(config):

    # print(config["name"])
    data_config = config["data"]
    sem = generate_sem(
        n_nodes = data_config["n_nodes"],
        expected_density=data_config["out_degree"],
        min_noise_scale=data_config["min_noise_scale"],
        max_noise_scale=data_config["max_noise_scale"],
        num_cycles=data_config["cycles"],
        enforce_dag=False, 
        contractive=data_config["contractive"]
    )

    n_interventions = data_config["n_nodes"] if data_config["interventions"] == -1 else data_config["interventions"]

    intervention_sets = [[None]] + [[i-1] for i in range(1, n_interventions+1)]

    datasets = []
    for targets in intervention_sets:
        datasets.append(
            sem.generateData(
                n_samples=data_config["n_samples_per_intervention"],
                intervention_set=targets, 
                beta_given=True,
                beta=data_config["beta"], 
                soft_intervention=data_config["soft-intervention"]
            )
        )

    max_targets = data_config["val_num_targets_per_setting_max"]
    min_targets = data_config["val_num_targets_per_setting_min"]

    val_targets = []
    val_datasets = []
    for _ in range(5): # here we set the number of experiments to 5 for validation data
        
        # randomly pick interventional targets within [min_targets, max_targets]
        n_targets = np.random.randint(min_targets, max_targets+1, 1)
        target_set = np.random.choice(data_config["n_nodes"], n_targets, replace=False)
        val_targets.append(target_set)

        # generate the dataset given the interventional targets
        val_datasets.append(
            sem.generateData(
                n_samples=data_config["n_samples_per_intervention"],
                intervention_set=target_set,
                beta_given=True, 
                beta=data_config["beta"],
                soft_intervention=data_config["soft-intervention"]
            )
        )

    return (intervention_sets, datasets), sem.weights, (val_targets, val_datasets)


def main(benchmark_root, n_trials=10, seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    # Walk through the directory tree and identify the folders containing 
    # settings.yaml files
    benchmark_root = os.path.abspath(benchmark_root)
    if not os.path.isdir(benchmark_root):
        raise FileNotFoundError(f"Benchmark root '{benchmark_root}' does not exist")

    settings_dirs = []
    for current_dir, _, files in os.walk(benchmark_root):
        if "settings.yaml" in files:
            settings_dirs.append(current_dir)  

    for directory in tqdm(settings_dirs):
        
        settings_path = os.path.join(directory, "settings.yaml")
        parent_dir = os.path.dirname(settings_path)
        with open(settings_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for trial in range(n_trials):
            trial_dir = os.path.join(parent_dir, f"trials/trial-{trial}")
            train_data, adj_mat, val_data = generate_data_from_config(config)

            # check if the trial directory exists
            if not os.path.exists(trial_dir+"/artifacts"):
                os.makedirs(trial_dir+"/artifacts")
            
            # SCM parameters
            np.savez(
                os.path.join(trial_dir, "artifacts/scm_true.npz"),
                adjacency=adj_mat
            )

            # train datasets
            intervention_sets, datasets = train_data
            np.savez(
                os.path.join(trial_dir, "artifacts/train_datasets"),
                targets=np.array(intervention_sets),
                **{f"data_{i}": data for i, data in enumerate(datasets)}
            )

            # validation datasets
            val_targets, val_datasets = val_data
            np.savez(
                os.path.join(trial_dir, "artifacts/val_datasets"),
                targets=val_targets, 
                **{f"data_{i}": data for i, data in enumerate(val_datasets)}
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=str, help='root directory of benchmark data')
    ap.add_argument("--n_trials", type=int, default=10, help="number of repeats")
    ap.add_argument("--seed", type=int, default=2025, help="Set the value of seed for repeatability")
    
    args = ap.parse_args()

    benchmark_root = args.benchmark
    n_trials = args.n_trials
    seed = args.seed
    main(benchmark_root, n_trials, seed)
    
