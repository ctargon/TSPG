#!/bin/bash
#PBS -N tspg
#PBS -l select=1:ncpus=2:ngpus=1:mem=64gb:gpu_model=p100,walltime=72:00:00
#PBS -j oe

# This should be the directory where you cloned the TSPG repository
TSPG_DIR="${HOME}/TSPG"

# Change these variables to match your specific input files, gene set, and target class
INPUT_DIR="input"
TRAIN_DATA="${INPUT_DIR}/example.train.emx.txt"
TRAIN_LABELS="${INPUT_DIR}/example.train.labels.txt"
PERTURB_DATA="${INPUT_DIR}/example.perturb.emx.txt"
PERTURB_LABELS="${INPUT_DIR}/example.perturb.labels.txt"
GMT_FILE="${INPUT_DIR}/example.genesets.txt"
GENE_SET="gene-set-000"
TARGET_CLASS="class-00"
OUTPUT_DIR="output"

# use conda environment... look at TSPG documentation, and make a conda environment named 'tspg'
# ...or change this environment name here 
module purge
module load anaconda3/5.1.0

source activate tspg

# Copy input data temporarily to local scratch, which is faster
# Use your home and zfs directory for permanent storage only
cp -r ${INPUT_DIR} ${TMPDIR}

# remove old output data
# rm -rf ${OUTPUT_DIR}

echo
echo "PHASE 1: TRAIN TARGET MODEL"
echo

# train target model on a gene set
${TSPG_DIR}/bin/train-target.py \
    --dataset    ${TMPDIR}/${TRAIN_DATA} \
    --labels     ${TMPDIR}/${TRAIN_LABELS} \
    --gene-sets  ${TMPDIR}/${GMT_FILE} \
    --set        ${GENE_SET} \
    --output-dir ${TMPDIR}/${OUTPUT_DIR}

echo
echo "PHASE 2: TRAIN PERTURBATION GENERATOR"
echo

# train AdvGAN model on a gene set
${TSPG_DIR}/bin/train-advgan.py \
    --dataset    ${TMPDIR}/${TRAIN_DATA} \
    --labels     ${TMPDIR}/${TRAIN_LABELS} \
    --gene-sets  ${TMPDIR}/${GMT_FILE} \
    --set        ${GENE_SET} \
    --target     ${TARGET_CLASS} \
    --output-dir ${TMPDIR}/${OUTPUT_DIR}

echo
echo "PHASE 3: GENERATE SAMPLE PERTURBATIONS"
echo

# generate perturbed samples using AdvGAN model
${TSPG_DIR}/bin/perturb.py \
    --train-data      ${TMPDIR}/${TRAIN_DATA} \
    --train-labels    ${TMPDIR}/${TRAIN_LABELS} \
    --perturb-data    ${TMPDIR}/${PERTURB_DATA} \
    --perturb-labels  ${TMPDIR}/${PERTURB_LABELS} \
    --gene-sets       ${TMPDIR}/${GMT_FILE} \
    --set             ${GENE_SET} \
    --target          ${TARGET_CLASS} \
    --output-dir      ${TMPDIR}/${OUTPUT_DIR}


echo
echo "PHASE 4: VISUALIZE SAMPLE PERTURBATIONS"
echo

# create t-SNE and heatmap visualizations of perturbed samples for a gene set
${TSPG_DIR}/bin/visualize.py \
    --train-data      ${TMPDIR}/${TRAIN_DATA} \
    --train-labels    ${TMPDIR}/${TRAIN_LABELS} \
    --perturb-data    ${TMPDIR}/${PERTURB_DATA} \
    --perturb-labels  ${TMPDIR}/${PERTURB_LABELS} \
    --gene-sets       ${TMPDIR}/${GMT_FILE} \
    --set             ${GENE_SET} \
    --target          ${TARGET_CLASS} \
    --output-dir      ${TMPDIR}/${OUTPUT_DIR} \
    --tsne \
    --heatmap

cp -r ${TMPDIR}/${OUTPUT_DIR} .
