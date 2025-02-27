# TSA_BasisMerging
This repository contains the code and data for the case study of the paper "Time series aggregation using exact error quantification for optimization of energy systems".

## Installation
1. Clone the repository
```bash
git clone https://github.com/beltran99/TSA_BasisMerging
```
2. Create a conda environment
```bash
conda env create -f environment.yml
```
3. Once created, you can activate the environment as follows:
```bash
conda activate bases_merging
```

## Tutorial
The jupyter notebook [example.ipynb](example.ipynb) contains a basic tutorial where one can learn to:
1. Solve a full model of the optimal transport problem case study.
2. Identify the set of unique bases from the solution of the full model.
3. Create and solve an aggregated model.
4. Create and solve bases mergers.

## Experiments
- The python script [enumeration.py](src/enumeration.py) executes the exhaustive enumeration of all 4140 possible bases mergers given by the 8 bases found in the optimal transport problem case study and saves the results to the output file [merger_enumeration.csv](merger_enumeration.csv).
- The python script [com.py](src/com.py) computes the CoM of all 4140 possible bases mergers given by the 8 bases found in the optimal transport problem case study and prints the results to the standard console output.
- The python script [combinatorics.py](src/combinatorics.py) compares the number of possible set partitions for different set sizes and strategies, i.e., Bell Number, Greedy, and Greedy with adjacency.