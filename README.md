# MissNODAG: Differentiable Learning of Cyclic Causal Graphs from Incomplete Data

This repository contains the code base for MissNODAG a _differentiable framework for learning cyclic causal graphs and the missingness mechanism from partially observed data_. 

## Requirements

`uv` was used to maintain the project. The dependencies can be found in the `pyproject.toml` file. Instructions for installing `uv` can be found [here](https://docs.astral.sh/uv/getting-started/installation/#pypi). After clong the project, run the following command to install all the necessary python dependencies.  
```shell
uv sync
```

## Running MissNODAG

Testing code for running MissNODAG can be found at the `notebooks` folder. The folder contains the following two files. 

1. `testing-missnodag.ipynb` - contains instructions and code to test MissNODAG (and MissDAG) on synthetic data sets. 
2. `perturb-cite-seq-testing.ipynb` - contains instructions and code to test the MissNODAG on Perturb-CITE-seq dataset. The genes chosen for testing are provided in the `chosen_genes.csv` file. 

To set up the ablations studies conducted in the paper, follow the steps below: 

1. Create a data folder to store the generated data. 
    ```shell
    mkdir data
    ```

2. The code to generate the settings files for the ablation experiments are located in the `benchmarks/generate_settings` folder. For instance, in order to generate the settings file for ablation with respect to number of cycles, run the following command: 
    ```shell
    uv run python -m benchmarks.generate_settings.ablation_cycles --outdir data
    ```
    This creates a new folder `data/ablation-cycles` containing subfloders `data/ablation-cycles/cycles-{i}`, where is the number of cycles and varies between 0 and 8. Each subfolder contains the corresponding `settings.yaml` file for the experiment. 

2. Data for the ablation experiment can be generated as follows: 
    ```shell
    uv run python -m benchmarks.gen_exp_dataset --benchmark data/ablation-cycles
    ```
    This would then generate the data for all the settings files within `data/ablation-cycles`.

3. Run MissNODAG on the data (below we show the command for `cycles-2`):
    ```shell
    uv run python -m benchmarks.run_benchmark_per_setting --settings data/ablation-cycles/cycles-2 --root_out results --n_trials 10 --model missnodag
    ```
    Here we assume that the folder `results` is present. If not, it can be created using `mkdir results` command. Upon completion of the experiment, a results file with a `.parquet` extension, containing the error metrics, is created and stored in the `results` directory. 


## Citation

This project is an implementation of the following paper: 

Muralikrishnna G Sethuraman, Razieh Nabi, Faramarz Fekri. (2026). [MissNODAG: Differentiable Learning of Cyclic Causal Graphs from Incomplete Data](https://openreview.net/pdf?id=nNZXQ3Q0GP). [TMLR](https://openreview.net/forum?id=nNZXQ3Q0GP)


If you find this code useful, please consider citing: 

```bibtex
@article{
sethuraman2026missnodag,
title={Miss{NODAG}: Differentiable Learning of Cyclic Causal Graphs from Incomplete Data},
author={Muralikrishnna Guruswamy Sethuraman and Razieh Nabi and Faramarz Fekri},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=nNZXQ3Q0GP},
note={}
}
```




