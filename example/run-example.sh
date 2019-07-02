#!/bin/bash
# Example usage of TSPG on a synthetic dataset.

# create synthetic input data
python scripts/make-input-data.py \
	--n-samples 1000 \
	--n-genes 200 \
	--n-classes 10 \
	--n-sets 1

# train target model on a gene set
python scripts/target_models.py \
	--dataset    example_data.txt \
	--labels     example_labels.txt \
	--gene-sets  example_genesets.txt

# train AdvGAN on a gene set
python scripts/train.py \
	--dataset    example_data.txt \
	--labels     example_labels.txt \
	--gene-sets  example_genesets.txt \
	--target     0

# perform attack on AdvGAN
python scripts/attack.py \
	--dataset    example_data.txt \
	--labels     example_labels.txt \
	--gene-sets  example_genesets.txt \
	--target     0
