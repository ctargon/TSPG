#!/bin/bash
# Create synthetic input data for TSPG.

module purge
module load anaconda3/5.1.0

bin/make-inputs.py \
    --n-samples 1000 \
    --n-genes   200 \
    --n-classes 10 \
    --n-sets    5 \
    --tsne      example.tsne.png
