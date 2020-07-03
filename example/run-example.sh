#!/bin/bash
# Example usage of TSPG on a synthetic dataset.

TRAIN_DATA="example.emx.txt"
TRAIN_LABELS="example.labels.txt"
PERTURB_DATA="example.emx.txt"
PERTURB_LABELS="example.labels.txt"
GMT_FILE="example.genesets.txt"
GENE_SET="gene-set-000"
TARGET_CLASS="class-00"
OUTPUT_DIR="example/output/${GENE_SET}"

# terminate script if any stage fails
set -e

# use conda environment
source activate tspg

# initialize output directory
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}

# create synthetic input data
bin/make-input-data.py \
	--n-samples 1000 \
	--n-genes   200 \
	--n-classes 10 \
	--n-sets    5 \
	--visualize

echo
echo "PHASE 1: TRAIN TARGET MODEL"
echo

# train target model on a gene set
bin/train-target.py \
	--dataset    ${TRAIN_DATA} \
	--labels     ${TRAIN_LABELS} \
	--gene-sets  ${GMT_FILE} \
	--set        ${GENE_SET} \
	--output-dir ${OUTPUT_DIR}

echo
echo "PHASE 2: TRAIN PERTURBATION GENERATOR"
echo

# train AdvGAN model on a gene set
bin/train-advgan.py \
	--dataset    ${TRAIN_DATA} \
	--labels     ${TRAIN_LABELS} \
	--gene-sets  ${GMT_FILE} \
	--set        ${GENE_SET} \
	--target     ${TARGET_CLASS} \
	--output-dir ${OUTPUT_DIR}

echo
echo "PHASE 3: GENERATE SAMPLE PERTURBATIONS"
echo

# generate perturbed samples using AdvGAN model
bin/perturb.py \
	--train-data      ${TRAIN_DATA} \
	--train-labels    ${TRAIN_LABELS} \
	--perturb-data    ${PERTURB_DATA} \
	--perturb-labels  ${PERTURB_LABELS} \
	--gene-sets       ${GMT_FILE} \
	--set             ${GENE_SET} \
	--target          ${TARGET_CLASS} \
	--output-dir      ${OUTPUT_DIR}

echo
echo "PHASE 4: VISUALIZE SAMPLE PERTURBATIONS"
echo

# create t-SNE and heatmap visualizations of perturbed samples for a gene set
bin/visualize.py \
	--train-data      ${TRAIN_DATA} \
	--train-labels    ${TRAIN_LABELS} \
	--perturb-data    ${PERTURB_DATA} \
	--perturb-labels  ${PERTURB_LABELS} \
	--gene-sets       ${GMT_FILE} \
	--set             ${GENE_SET} \
	--target          ${TARGET_CLASS} \
	--output-dir      ${OUTPUT_DIR} \
	--tsne \
	--heatmap
