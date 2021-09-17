#!/bin/bash
# Create synthetic input data for TSPG.

module purge
module load anaconda3/5.1.0-gcc

source activate tspg

bin/make-inputs.py \
    --n-samples 1000 \
    --n-genes   100 \
    --n-classes 5 \
    --n-sets    10 \
    --tsne      example.tsne.png
