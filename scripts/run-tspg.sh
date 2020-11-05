#!/bin/bash
#PBS -N tspg
#PBS -l select=1:ncpus=8:ngpus=2:mem=64gb:gpu_model=p100,walltime=24:00:00
#PBS -j oe

module purge
module load anaconda3/5.1.0
module load gcc/4.8.1
module load openmpi/1.10.3

# This should be the directory where you cloned the TSPG repository
export TSPG_DIR="${HOME}/TSPG"

# This should be the explicit directory where all your inputs are... change to whatever
export INPUT_DIR="input"

# Copy input data temporarily to local scratch, which is faster
# Use your home and zfs directory for permanent storage only
mpirun sleep 20

cp -r ${INPUT_DIR} ${TMPDIR}

# Change these variables to match your specific input files, gene set, and target class
TRAIN_DATA="${TMPDIR}/${INPUT_DIR}/train.GEM.txt"
TRAIN_LABELS="${TMPDIR}/${INPUT_DIR}/train.labels.txt"
PERTURB_DATA="${TMPDIR}/${INPUT_DIR}/perturb.GEM.txt"
PERTURB_LABELS="${TMPDIR}/${INPUT_DIR}/perturb.labels.txt"
GMT_FILE="${TMPDIR}/${INPUT_DIR}/genesets.txt"
GENE_SET="gene-set-all"
TARGET_CLASS="normal"
OUTPUT_DIR="${TMPDIR}/output"

# use conda environment... look at TSPG documentation, and make a conda environment named 'tspg'
# ...or change this environment name here 
source activate tspg

# remove old output data
# rm -rf ${OUTPUT_DIR}
# mkdir -p ${OUTPUT_DIR}

echo
echo "PHASE 1: TRAIN TARGET MODEL"
echo

# train target model on a gene set
${TSPG_DIR}/bin/train-target.py \
    --dataset    ${TRAIN_DATA} \
    --labels     ${TRAIN_LABELS} \
    --gene-sets  ${GMT_FILE} \
    --set        ${GENE_SET} \
    --output-dir ${OUTPUT_DIR}

echo
echo "PHASE 2: TRAIN PERTURBATION GENERATOR"
echo

# train AdvGAN model on a gene set
${TSPG_DIR}/bin/train-advgan.py \
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
${TSPG_DIR}/bin/perturb.py \
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
${TSPG_DIR}/bin/visualize.py \
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

cp -r ${OUTPUT_DIR} ${PWD}
