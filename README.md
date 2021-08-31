# Transcriptome State Perturbation Generator

This repository contains the code for the Transcriptome State Perturbation Generator (TSPG). TSPG is an experimental tool for discovering biomarker genes using gene expression data and adversarial learning.

Publications based on TSPG:
- [Cellular State Transformations Using Deep Learning for Precision Medicine Applications](https://www.cell.com/patterns/fulltext/S2666-3899(20)30115-X)
- [Cellular State Transformations using Generative Adversarial Networks](https://arxiv.org/abs/1907.00118) (arXiv preprint)

## Installation

All of TSPG's dependencies can be installed via Anaconda. On a shared system (such as a university research cluster), it is recommended that you install everything in an Anaconda environment:

```bash
# specific to Clemson's Palmetto cluster
module load anaconda3/5.1.0-gcc/8.3.1

conda env create -f environment.yml
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

# run the example
cd TSPG
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
bin/convert.py GEM.txt GEM.npy --transpose
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

### Phase 1: Train Target Model

In this stage a basic MLP is trained to classify the input data and report the test accuracy of the model. Make sure that this target model achieves a high test accuracy or else the perturbation generator won't be very useful.

### Phase 2: Train Perturbation Generator

In this stage the generator is trained to perturb samples in the input data such that the target model classifies these samples as the target class. This stage is the most time-consuming, especially if the input data is large. Once the generator is trained it will report the "perturbation accuracy", which is the percentage of samples in the test set that were successfully perturbed (i.e. the target model classified them as the target class). A high perturbation accuracy means that the generator is effective at perturbing samples to "look like" the target class.

### Phase 3: Generate Sample Perturbations

In this stage the trained generator is used to perturb samples from a test set. Note that the data used to train the generator must be provided as the "training data" in this stage as TSPG still needs to refer to it. The output consists of a perturbation vector for each sample in the test set, which when added to the sample should cause it to be classified as the target class. Additionally, this stage will generate a dataframe of "mean perturbations", where each column is the difference between the means of the i-th class and target class. These perturbations can be used as a baseline to measure the effectiveness of the TSPG perturbations.

### Phase 4: Visualize Sample Perturbations

In this stage the perturbations from the previous stage are visualized in the form of (1) a t-SNE plot of all training samples, test samples, and perturbed test samples, and (2) a heatmap of each test sample which shows the original sample, perturbation, perturbed sample, and mean of the target class for comparison. Both of these visualizations should be examined to verify that the test samples were successfully perturbed to the target class. Note that the heatmap will order the genes (rows) according to the perturbation vector (the "P" column), so the top-most and bottom-most genes in that arrangement are the most strongly affected genes. These genes can be extracted from the perturbation vector via Excel for further analysis (functional enrichment, comparison with DGE analysis, etc).
