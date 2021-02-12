#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.manifold
import sklearn.preprocessing

import utils



def plot_tsne(x, y, classes, class_indices, n_pca=None, x_pert=None, y_pert=None, target=-1, output_dir='.'):
    # merge original and perturbed data (if provided)
    if target != -1:
        x_merged = np.vstack([x, x_pert])
    else:
        x_merged = x

    # compute PCA embedding of merged data
    x_pca = sklearn.decomposition.PCA(n_components=n_pca, copy=False).fit_transform(x_merged)

    # compute t-SNE embedding of merged data
    x_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(x_pca)

    # separate embedded data back into original and perturbed
    x      = x_tsne[:len(x)]
    x_pert = x_tsne[len(x):]

    # plot t-SNE embedding by class
    plt.axis('off')

    for k in class_indices:
        indices = (y == k)
        if indices.sum() > 0:
            plt.scatter(x[indices, 0], x[indices, 1], label=classes[k], alpha=0.75)

    if target != -1:
        for k in class_indices:
            indices = (y_pert == k)
            if indices.sum() > 0:
                label = '%s (perturbed)' % (classes[k])
                plt.scatter(x_pert[indices, 0], x_pert[indices, 1], label=label, alpha=0.25)

    plt.subplots_adjust(right=0.70)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('%s/tsne.png' % (output_dir))
    plt.close()



def plot_heatmap(df, sample_name, source_class, target_class, output_dir='.'):
    fig, ax = plt.subplots(1, len(df.columns))

    for i in range(len(df.columns)):
        # get the vector, then tile it some so it is visible if very long
        column = np.expand_dims(df[df.columns[i]], -1)
        column = np.tile(column, (1, max(1, int(column.shape[0] / 10))))

        # plot tiled vector as heatmap image
        im = ax[i].imshow(column, cmap='seismic')
        im.set_clim(-1, 1)

        ax[i].set_title(df.columns[i])
        ax[i].set_xticks([])
        ax[i].set_xticklabels([])

        # display row names if there aren't too many
        if i == 0 and len(df.index) < 30:
            ax[i].set_yticks(np.arange(len(df.index)))
            ax[i].set_yticklabels(df.index)
        else:
            ax[i].set_yticks([])
            ax[i].set_yticklabels([])

    # insert colobar and shrink it some
    cbar = ax[-1].figure.colorbar(im, ax=ax[-1], shrink=0.5)
    cbar.ax.set_ylabel('Expression Level', rotation=-90, va='bottom')

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
    parser.add_argument('--target', help='target class')
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

    # sanitize class names
    classes = [utils.sanitize(c) for c in classes]

    # determine target class
    try:
        if args.target == None:
            args.target = -1
        else:
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
        gene_sets = {'all_genes': df_genes}

    # create visualizations for each gene set
    name = args.set

    try:
        genes = gene_sets[name]
    except:
        print('gene set is not the subset file provided')
        sys.exit(1)

    # extract train/perturb data
    x_train = df_train[genes]
    x_perturb = df_perturb[genes]

    # normalize perturb data (using the train data)
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_perturb = scaler.transform(x_perturb)

    # load perturbed samples
    if args.target != -1:
        df_pert = utils.load_dataframe('%s/%s.perturbations.samples.txt' % (args.output_dir, classes[args.target]))
        p = df_pert.values.T
    else:
        df_pert = pd.DataFrame()

    # plot t-SNE visualization if specified
    if args.tsne:
        # select classes to include in plot
        class_indices = list(range(len(classes)))

        plot_tsne(
            x_train,
            y_train,
            classes,
            class_indices,
            n_pca=args.tsne_npca,
            x_pert=np.clip(x_perturb + p, 0, 1),
            y_pert=y_perturb,
            target=args.target,
            output_dir=args.output_dir)

    # plot heatmaps if specified
    if args.heatmap:
        # compute mean of target class
        mu_target = x_train[y_train == args.target].mean(axis=0)

        # plot heatmap of each perturbed sample
        for i, sample_name in enumerate(df_pert.columns):
            # extract original sample, perturbation, and perturbed sample
            df = pd.DataFrame({
                'X': x_perturb[i],
                'P': p[i],
                'X + P': np.clip(x_perturb[i] + p[i], 0, 1),
                'mu_T': mu_target
            })
            df = df.sort_values('P', ascending=False)

            # create heatmap of original and perturbed samples
            sample_name = utils.sanitize(sample_name)
            source_class = classes[y_perturb[i]]
            target_class = classes[args.target]

            plot_heatmap(df, sample_name, source_class, target_class, output_dir=args.output_dir)
