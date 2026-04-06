import os, argparse, yaml

# Data generation parameters

default_params = {
    "n_samples_per_intervention" : 500,
    "out_degree" : 2,
    "min_noise_scale" : 0.2,
    "max_noise_scale" : 0.5,
    "contractive" : True,
    "val_num_targets_per_setting_min" : 2,
    "val_num_targets_per_setting_max" : 2,
    "cycles" : "random",
    "beta" : 1.0,
    "interventions" : -1,
    "soft-intervention": False,
    "missing-mech": "full", 
    "mlp" : False,
    "n_nodes" : 10,
    "missing-prob" : 0.3, 
}

exp_types = [
    "ablation-cycles",
]

exp_specific_params = {
    "ablation-cycles": {"cycles": [0, 2, 4, 6, 8]},
}

model_params = {
    "lip_const" : 0.9,
    "activation" : "tanh",
    "max_epochs" : 100,
    "batch_size" : 512,
    "lr" : 1e-1,
    "lc" : 1e-3,
    "missing-mech-sparsity-reg" : 1e-2,
    "ldag" : 5e-1,
    "lnc" : 1e-1,
    "adj_threshold" : 0.7,
    "mm_threshold" : 0.2
}

eval_params = {
    "metrics" : ["tl-shd", "x2r-shd", "r2r-shd"]
}

def main(benchmark_root):

    for abl_type in exp_types:
        abl_dir = os.path.join(benchmark_root, abl_type)
        if not os.path.exists(abl_dir):
            os.makedirs(abl_dir)

        cfg = {
            "name" : abl_type,
            "data" : {},
            "model" : {},
            "eval" : {}
        }

        for param, val in model_params.items():
            cfg["model"][param] = val
        
        for param, val in eval_params.items():
            cfg["eval"][param] = val

        settings = exp_specific_params[abl_type]
        
        for param, val in default_params.items():
            if param not in list(settings.keys())[0]:
                cfg["data"][param] = val
        
        setting, vals = list(settings.keys())[0], list(settings.values())[0]
        for val in vals: 
            setting_dir = os.path.join(abl_dir, setting+f"-{val}")
            cfg["setting"] = setting + f"-{val}"
            if not os.path.exists(setting_dir):
                os.makedirs(setting_dir)
            
            cfg["data"][setting] = val
        
            with open(os.path.join(setting_dir, "settings.yaml"), "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./", help="Root directory to write the config files")

    args = ap.parse_args()

    benchmark_root = args.outdir 

    main(benchmark_root)