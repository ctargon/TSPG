Code for the paper "Cellular State Transformations using Generative Adversarial Networks" https://arxiv.org/abs/1907.00118

# Transcriptome State Perturbation Generator

This repository contains the code for the Transcriptome State Perturbation Generator (TSPG). TSPG is an experimental tool for discovering biomarker genes using gene expression data and adversarial learning.

## Installation

All of TSPG's dependencies can be installed via Anaconda. On a shared system (such as a university research cluster), it is recommended that you install everything in an Anaconda environment:

```bash
# specific to Clemson's Palmetto cluster
module add anaconda3/5.1.0

conda create -n tspg python=3.6 tensorflow-gpu=1.13.1 matplotlib numpy pandas scikit-learn seaborn
```

You must then "activate" your environment in order to use it:
```bash
conda activate tspg

# use TSPG

conda deactivate
```

After that, simply clone this repository to use TSPG.
```bash
git clone https://github.com/ctargon/TSPG.git
cd TSPG

# run the example
example/run-example.sh
```

## Usage

TSPG consists of several phases, involving multiple scripts that are run in sequence. The easiest way to learn how to run these scripts, as well as the input / output data involved, is to run the example script as shown above. It demonstrates how to run TSPG on synthetic input data from `make-input-data.py`.

### Input Data

TSPG takes three primary inputs: (1) a gene expression matrix (GEM), (2) a list of sample labels, and (3) a list of gene sets. These inputs are described below.

The __gene expression matrix__ should be a plain-text file with rows being samples and columns being genes (features). Values in each row should be separated by tabs.
```
	Gene1	Gene2	Gene3	Gene4
Sample1	0.523	0.991	0.421	0.829
Sample2	8.891	7.673	3.333	9.103
Sample3	4.444	5.551	6.102	0.013
```

For large GEM files, it is recommended that you convert the GEM to numpy format using `convert.py` from the [GEMprep](https://github.com/SystemsGenetics/GEMprep) repo, as TSPG can load this binary format much more quickly than it does the plaintext format. The `convert.py` script can also transpose your GEM if it is arranged the wrong way:
```bash
python bin/convert.py GEM.txt GEM.npy --transpose
```

This example will create three files: `GEM.npy`, `GEM.rownames.txt`, and `GEM.colnames.txt`. The latter two files contain the row names and column names, respectively. Make sure that the rows are samples and the columns are genes!

The __label file__ should contain a label for each sample, corresponding to something such as a condition or phenotype state for the sample. This file should contain two columns, the first being the sample names and the second being the labels. Values in each row should be separated by tabs.
```
Sample1	Label1
Sample2	Label2
Sample3	Label3
Sample4	Label4
```

The __gene set list__ should contain the name and genes for a gene set on each line, similar to the GMT format. The gene names should be identical to those used in the GEM file. Values on each row should be separated by tabs.
```
GeneSet1	Gene1	Gene2	Gene3
GeneSet2	Gene2	Gene4	Gene5	Gene6
```
