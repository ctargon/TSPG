#!/usr/bin/env python3

import argparse
import cycler
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sklearn.decomposition
import sklearn.manifold
import sklearn.preprocessing
import sys

import utils



def plot_tsne(
    x, y,
    x_pert, y_pert,
    target,
    classes,
    class_indices,
    n_pca=None,
    output_dir='.'):

    # merge original and perturbed data (if provided)
    x_merged = np.vstack([x, x_pert])

    # validate n_pca argument
    if n_pca != None:
        n_pca = min(n_pca, *x_merged.shape)

    # compute PCA embedding of merged data
    x_pca = sklearn.decomposition.PCA(n_components=n_pca).fit_transform(x_merged)

    # compute t-SNE embedding of merged data
    x_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(x_pca)

    # separate embedded data back into original and perturbed
    x      = x_tsne[:len(x)]
    x_pert = x_tsne[len(x):]

    # initialize figure
    plt.axis('off')

    # plot original data
    plt.gca().set_prop_cycle(cycler.cycler(color=cm.get_cmap('tab10').colors))

    for k in class_indices:
        indices = (y == k)
        if np.any(indices):
            plt.scatter(
                x[indices, 0],
                x[indices, 1],
                label=classes[k],
                edgecolors='w')

    # plot perturbed data
    plt.gca().set_prop_cycle(cycler.cycler(color=cm.get_cmap('tab10').colors))

    for k in class_indices:
        indices = (y_pert == k)
        if np.any(indices):
            plt.scatter(
                x_pert[indices, 0],
                x_pert[indices, 1],
                label='%s (perturbed)' % (classes[k]),
                alpha=0.25,
                edgecolors='w')

    # plot legend
    plt.subplots_adjust(right=0.70)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig('%s/%s.tsne.png' % (output_dir, classes[target]))
    plt.close()



def plot_heatmap(
    data,
    sample_name,
    source_class,
    target_class,
    output_dir='.'):

    # initialize figure
    fig, axes = plt.subplots(1, len(data.columns))

    # plot heatmap of each column
    for i in range(len(data.columns)):
        column = data.columns[i]
        ax = axes[i]

        # expand column into a matrix with 1:10 aspect ratio
        x = np.expand_dims(data[column], -1)
        x = np.repeat(x, max(1, len(x) // 10), axis=1)

        # plot heatmap
        im = ax.imshow(x, cmap='seismic')
        im.set_clim(-1, 1)

        ax.set_title(column)
        ax.set_xticks([])
        ax.set_xticklabels([])

        # display row names if there aren't too many
        if i == 0 and len(data.index) < 30:
            ax.set_yticks(np.arange(len(data.index)))
            ax.set_yticklabels(data.index)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

    # plot colobar
    cbar = axes[-1].figure.colorbar(im, ax=axes[-1], shrink=0.5)
    cbar.ax.set_ylabel('Expression Level', rotation=-90, va='bottom')

    plt.tight_layout()
    plt.savefig('%s/%s.%s.%s.png' % (output_dir, source_class, target_class, sample_name))
    plt.close()



if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', help='training data (samples x genes)', required=True)
    parser.add_argument('--train-labels', help='training labels', required=True)
    parser.add_argument('--perturb-data', help='perturb data (samples x genes)', required=True)
    parser.add_argument('--perturb-labels', help='perturb labels', required=True)
    parser.add_argument('--gene-sets', help='list of curated gene sets')
    parser.add_argument('--set', help='specific gene set to run')
    parser.add_argument('--tsne', help='plot t-SNE of samples', action='store_true')
    parser.add_argument("--tsne-npca", help="number of principal components to take before t-SNE", type=int)
    parser.add_argument('--heatmap', help='plot heatmaps of sample perturbations', action='store_true')
    parser.add_argument('--heatmap-nsamples', help='plot a random sampling of heatmaps', type=int)
    parser.add_argument('--target', help='target class', required=True)
    parser.add_argument('--output-dir', help='output directory', default='.')

    args = parser.parse_args()

    # load input data
    print('loading train/perturb data...')

    df_train = utils.load_dataframe(args.train_data)
    df_perturb = utils.load_dataframe(args.perturb_data)

    y_train, classes = utils.load_labels(args.train_labels)
    y_perturb, _ = utils.load_labels(args.perturb_labels, classes)

    print('loaded train data (%s genes, %s samples)' % (df_train.shape[1], df_train.shape[0]))
    print('loaded perturb data (%s genes, %s samples)' % (df_perturb.shape[1], df_perturb.shape[0]))

    # impute missing values
    min_value = df_train.min().min()

    df_train.fillna(value=min_value, inplace=True)
    df_perturb.fillna(value=min_value, inplace=True)

    # sort labels to match data if needed
    if (df_train.index != y_train.index).any():
        print('warning: train data and labels are not ordered the same, re-ordering labels')
        y_train = y_train.loc[df_train.index]

    if (df_perturb.index != y_perturb.index).any():
        print('warning: perturb data and labels are not ordered the same, re-ordering labels')
        y_perturb = y_perturb.loc[df_perturb.index]

    # sanitize class names
    classes = [utils.sanitize(c) for c in classes]

    # determine target class
    try:
        args.target = classes.index(args.target)
        print('target class is: %s' % (classes[args.target]))
    except ValueError:
        print('error: class %s not found in dataset' % (args.target))
        sys.exit(1)

    # load gene sets file if it was provided
    if args.gene_sets != None:
        print('loading gene sets...')

        gene_sets = utils.load_gene_sets(args.gene_sets)
        gene_sets = utils.filter_gene_sets(gene_sets, df_perturb.columns)

        print('loaded %d gene sets' % (len(gene_sets)))
    else:
        gene_sets = {'all_genes': set(df_genes)}

    # create visualizations for each gene set
    try:
        name = args.set
        genes = gene_sets[name]
    except:
        print('error: gene set is not the subset file provided')
        sys.exit(1)

    # extract train/perturb data
    x_train = df_train[genes]
    x_perturb = df_perturb[genes]

    # normalize perturb data (using the train data)
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_perturb = scaler.transform(x_perturb)

    # load sample perturbations (genes x samples)
    p = utils.load_dataframe('%s/%s.perturbations.samples.txt' % (args.output_dir, classes[args.target]))
    p = p.to_numpy()

    # transpose sample perturbations to (samples x genes)
    p = p.T

    # compute perturbed samples
    x_perturbed = np.clip(x_perturb + p, 0, 1)

    # plot t-SNE visualization if specified
    if args.tsne:
        print('creating tsne plot...')

        # select classes to include in plot
        class_indices = list(range(len(classes)))

        plot_tsne(
            x_train, y_train,
            x_perturbed, y_perturb,
            args.target,
            classes,
            class_indices,
            n_pca=args.tsne_npca,
            output_dir=args.output_dir)

    # plot heatmaps if specified
    if args.heatmap:
        print('creating heatmaps...')

        # compute mean of target class
        mu_target = np.mean(x_train[y_train == args.target], axis=0)

        # select random subset of heatmaps if specified
        if args.heatmap_nsamples != None:
            samples = random.sample(list(df_perturb.index), k=args.heatmap_nsamples)
        else:
            samples = df_perturb.index

        # plot perturbation heatmaps for each sample
        for i, sample_name in enumerate(samples):
            print('  %s' % (sample_name))

            # extract original sample, perturbation, perturbed sample, and target mean
            data = pd.DataFrame({
                'X': x_perturb[i],
                'P': p[i],
                'X + P': x_perturbed[i],
                'mu_T': mu_target
            }, index=genes)
            data = data.sort_values('P', ascending=False)

            # plot heatmap for each extracted column
            sample_name = utils.sanitize(sample_name)
            source_class = classes[y_perturb[i]]
            target_class = classes[args.target]

            plot_heatmap(
                data,
                sample_name,
                source_class,
                target_class,
                output_dir=args.output_dir)
