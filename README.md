# Transcriptome State Perturbation Generator

This repository contains the code for the Transcriptome State Perturbation Generator (TSPG). TSPG is an experimental tool for discovering biomarker genes using gene expression data and adversarial learning.

Publications based on TSPG:
- [Cellular State Transformations Using Deep Learning for Precision Medicine Applications](https://www.cell.com/patterns/fulltext/S2666-3899(20)30115-X)
- [Cellular State Transformations using Generative Adversarial Networks](https://arxiv.org/abs/1907.00118) (arXiv preprint)

Publications TSPG is based on:
- [Generating Adversarial Examples with Adversarial Networks](https://arxiv.org/pdf/1801.02610.pdf)

## Installation

TSPG is a collection of Python scripts as well as a Nextflow pipeline which wraps the Python scripts and provides some additional functionality. All of the Python dependencies can be installed in an Anaconda environment:
```bash
# load Anaconda module if needed
module load anaconda3/5.1.0-gcc/8.3.1

# create conda environment called "tspg"
conda env create -f environment.yml
```

To use the Python scripts directly, clone this repository and try the example:
```bash
# clone repository
git clone https://github.com/ctargon/TSPG.git
cd TSPG

# prepare example data
scripts/make-inputs.sh

mkdir -p input
mv example.* input

# run TSPG on example data
# (you may have to tweak this script for your environment)
scripts/run-tspg.sh
```

Alternatively, you can run TSPG as a Nextflow pipeline. It's a lot easier because you only need to install [Nextflow](https://nextflow.io/) and either the Anaconda environment or Docker/Singularity. After that, you can run the example data with a single command:
```bash
nextflow run ctargon/tspg -profile example,<conda|docker|singularity>,standard
```

If you use the Nextflow pipeline, you'll want to check out the [config file](https://github.com/ctargon/TSPG/blob/master/nextflow.config) to see the available params and profiles. Params can be supplied on the command-line, and custom profiles exist for different environments such as Palmetto's PBS scheduler, for example:
```bash
nextflow run ctargon/tspg \
    -profile conda,palmetto \
    --train_data "example.train.emx.txt" \
    --train_labels "example.train.labels.txt" \
    --perturb_data "example.perturb.emx.txt" \
    --perturb_labels "example.perturb.labels.txt" \
    --gmt_file "example.genesets.txt" \
    --target_class "class-00,class-01,class-02"
```

The Nextflow pipeline makes it easy to run TSPG multiple times on the same data with different target classes. However, the pipeline does not currently support input files in numpy format.

## Usage

TSPG consists of several phases, involving multiple scripts that are run in sequence.

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

The __labels file__ should contain a label for each sample, corresponding to something such as a condition or phenotype state for the sample. This file should contain two columns, the first being the sample names and the second being the labels. Values in each row should be separated by tabs.
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

It is recommended that you use the `split-train-perturb.py` script to split your GEM and labels file into train/perturb sets (similar to train/test sets in machine learning) in order to verify that TSPG can effectively perturb data that it did not see during training.

### Phase 1: Train Target Model

In this stage a basic MLP is trained to classify the input data and report the test accuracy of the model. Make sure that this target model achieves a high test accuracy or else the perturbation generator won't be very useful.

### Phase 2: Train Perturbation Generator

In this stage the generator is trained to perturb samples in the input data such that the target model classifies these samples as the target class. This stage is the most time-consuming, especially if the input data is large. Once the generator is trained it will report the "perturbation accuracy", which is the percentage of samples in the test set that were successfully perturbed (i.e. the target model classified them as the target class). A high perturbation accuracy means that the generator is effective at perturbing samples to "look like" the target class.

### Phase 3: Generate Sample Perturbations

In this stage the trained generator is used to perturb samples from a test set. Note that the data used to train the generator must be provided as the "training data" in this stage as TSPG still needs to refer to it. The output consists of a perturbation vector for each sample in the test set, which when added to the sample should cause it to be classified as the target class. Additionally, this stage will generate a dataframe of "mean perturbations", where each column is the difference between the means of the i-th class and target class. These perturbations can be used as a baseline to measure the effectiveness of the TSPG perturbations.

### Phase 4: Visualize Sample Perturbations

In this stage the perturbations from the previous stage are visualized in the form of (1) a t-SNE plot of all training samples, test samples, and perturbed test samples, and (2) a heatmap of each test sample which shows the original sample, perturbation, perturbed sample, and mean of the target class for comparison. Both of these visualizations should be examined to verify that the test samples were successfully perturbed to the target class. Note that the heatmap will order the genes (rows) according to the perturbation vector (the "P" column), so the top-most and bottom-most genes in that arrangement are the most strongly affected genes. These genes can be extracted from the perturbation vector via Excel for further analysis (functional enrichment, comparison with DGE analysis, etc).
