# MissNODAG: Differentiable Learning of Cyclic Causal Graphs from Incomplete Data

This repository contains the code base for MissNODAG a differentiable framework for learning cyclic causal graphs and the missingness mechanism from partially observed data. 

## Requirements

Python (version > 3.9) is required to run the code in this library (our code was tested on Python 3.10.15). The testing codes are written using Jupyter notebook, please ensure Jupyter is installed prior to running them. To install the python dependencies, run the following command from the terminal. 
```shell
pip install -r requirements.txt
```

## Running MissNODAG

Testing code for running MissNODAG and the baselines can be found at the `notebooks` folder. The folder contains the following two files. 

1. `testing-missnodag.ipynb` - contains instructions and code to test MissNODAG (and MissDAG) on synthetic data sets. 
2. `testing-baseline.ipynb` - contains instructions and code to test the baselines on synthetic data sets. 

In order to test MissNODAG on gene perturbation data set, follow the instructions inside the notebooks within the folder `perturb_cite_seq` to download and process the data set. Once the data set is ready, run the code `perturb_cite_seq.py` inside `learning` to test MissNODAG on the downloaded data set. The genes chosen for testing are provided in the `chosen_genes.csv` file. 




