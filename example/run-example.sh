#!/bin/bash
# Example usage of TSPG on a synthetic dataset.

# remove old output data
OUTPUT_DIR="example/output"

rm -rf ${OUTPUT_DIR}

# create synthetic input data
python scripts/make-input-data.py \
	--n-samples 1000 \
	--n-genes 200 \
	--n-classes 10 \
	--n-sets 5

# train target model on a gene set
python scripts/train-target.py \
	--dataset    example.emx.txt \
	--labels     example.labels.txt \
	--gene-sets  example.genesets.txt \
	--set        gene-set-000 \
	--output-dir ${OUTPUT_DIR}

# train AdvGAN model on a gene set
python scripts/train-advgan.py \
	--dataset    example.emx.txt \
	--labels     example.labels.txt \
	--gene-sets  example.genesets.txt \
	--set        gene-set-000 \
	--target     0 \
	--output-dir ${OUTPUT_DIR}

# perform attack on AdvGAN
python scripts/attack.py \
	--dataset    example.emx.txt \
	--labels     example.labels.txt \
	--gene-sets  example.genesets.txt \
	--set        gene-set-000 \
	--target     0 \
	--output-dir ${OUTPUT_DIR}

# create t-SNE and heatmap visualizations of perturbed samples for a gene set
python scripts/visualize.py \
	--dataset    example.emx.txt \
	--labels     example.labels.txt \
	--gene-sets  example.genesets.txt \
	--set        gene-set-000 \
	--tsne \
	--heatmap \
	--target     0 \
	--output-dir ${OUTPUT_DIR}
