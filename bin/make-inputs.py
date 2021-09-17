#!/usr/bin/env python3

import argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sklearn.datasets
import sklearn.manifold
import sklearn.model_selection
import sys

import utils



if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Create a synthetic classification dataset')
    parser.add_argument('--n-samples', help='number of samples', type=int, default=100)
    parser.add_argument('--n-genes', help='number of genes', type=int, default=20)
    parser.add_argument('--n-classes', help='number of classes', type=int, default=2)
    parser.add_argument('--train-size', help='training set proportion', type=float, default=0.8)
    parser.add_argument('--n-sets', help='number of gene sets', type=int, default=10)
    parser.add_argument('--train-data', help='train dataset filename', default='example.train.emx.txt')
    parser.add_argument('--train-labels', help='train labels filename', default='example.train.labels.txt')
    parser.add_argument('--perturb-data', help='perturb dataset filename', default='example.perturb.emx.txt')
    parser.add_argument('--perturb-labels', help='perturb labels filename', default='example.perturb.labels.txt')
    parser.add_argument('--gene-sets', help='name of gene sets file', default='example.genesets.txt')
    parser.add_argument('--tsne', help='create t-SNE plot of dataset')

    args = parser.parse_args()

    # create synthetic dataset
    x, y = sklearn.datasets.make_blobs(
        args.n_samples,
        args.n_genes,
        centers=args.n_classes)

    # initialize class names
    classes = ['class-%02d' % i for i in range(args.n_classes)]
    y = [classes[y_i] for y_i in y]

    # initialize gene names, sample names
    x_samples = ['sample-%08d' % i for i in range(args.n_samples)]
    x_genes = ['gene-%06d' % i for i in range(args.n_genes)]

    # initialize dataframes
    x = pd.DataFrame(x, index=x_samples, columns=x_genes)
    y = pd.DataFrame(y, index=x_samples)

    # create synthetic gene sets
    gene_sets = []

    for i in range(args.n_sets):
        n_genes = random.randint(5, min(max(10, args.n_genes // 10), args.n_genes))
        genes = random.sample(x_genes, n_genes)

        gene_sets.append(['gene-set-%03d' % i] + genes)

    # create t-sne visualization if specified
    if args.tsne:
        # compute t-SNE embedding
        x_tsne = sklearn.manifold.TSNE().fit_transform(x)

        # plot t-SNE embedding with class labels
        plt.axis('off')

        for c in classes:
            indices = (y[0] == c)
            plt.scatter(x_tsne[indices, 0], x_tsne[indices, 1], label=c, edgecolors='w')

        plt.subplots_adjust(right=0.70)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig(args.tsne)
        plt.close()

    # split dataset into train/perturb sets
    x_train, x_perturb, y_train, y_perturb = sklearn.model_selection.train_test_split(x, y, test_size=1 - args.train_size)

    # save datasets to file
    utils.save_dataframe(args.train_data, x_train)
    utils.save_dataframe(args.perturb_data, x_perturb)

    # save labels to file
    y_train.to_csv(args.train_labels, sep='\t', header=None)
    y_perturb.to_csv(args.perturb_labels, sep='\t', header=None)

    # save gene sets to file
    f = open(args.gene_sets, 'w')
    f.write('\n'.join(['\t'.join(gene_set) for gene_set in gene_sets]))
