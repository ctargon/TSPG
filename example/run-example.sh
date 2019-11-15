#!/bin/bash
# Example usage of TSPG on a synthetic dataset.

GENE_SET="gene-set-000"
TARGET_CLASS="0"
OUTPUT_DIR="example/output/${GENE_SET}"

# remove old output data
rm -rf ${OUTPUT_DIR}

# create synthetic input data
bin/make-input-data.py \
	--n-samples 1000 \
	--n-genes 200 \
	--n-classes 10 \
	--n-sets 5

# train target model on a gene set
bin/train-target.py \
	--dataset    example.emx.txt \
	--labels     example.labels.txt \
	--gene-sets  example.genesets.txt \
	--set        ${GENE_SET} \
	--output-dir ${OUTPUT_DIR}

# train AdvGAN model on a gene set
bin/train-advgan.py \
	--dataset    example.emx.txt \
	--labels     example.labels.txt \
	--gene-sets  example.genesets.txt \
	--set        ${GENE_SET} \
	--target     ${TARGET_CLASS} \
	--output-dir ${OUTPUT_DIR}

# generate perturbed samples using AdvGAN model
bin/perturb.py \
	--train-data   example.emx.txt \
	--train-labels example.labels.txt \
	--test-data    example.emx.txt \
	--test-labels  example.labels.txt \
	--gene-sets    example.genesets.txt \
	--set          ${GENE_SET} \
	--target       ${TARGET_CLASS} \
	--output-dir   ${OUTPUT_DIR}

# create t-SNE and heatmap visualizations of perturbed samples for a gene set
bin/visualize.py \
	--train-data   example.emx.txt \
	--train-labels example.labels.txt \
	--test-data    example.emx.txt \
	--test-labels  example.labels.txt \
	--gene-sets    example.genesets.txt \
	--set          ${GENE_SET} \
	--target       ${TARGET_CLASS} \
	--tsne \
	--heatmap \
	--output-dir   ${OUTPUT_DIR}
