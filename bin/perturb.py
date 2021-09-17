#!/usr/bin/env python3

import argparse
import math
import numpy as np
import pandas as pd
import os
import sklearn.model_selection
import sklearn.preprocessing
import sys
from tensorflow import keras

import advgan
import utils



def perturb_classes(x, y, classes, target):
    # decode labels from one-hot
    y = np.argmax(y, axis=1)

    # compute mean differences
    perturbations = []

    for k in range(len(classes)):
        # get mean of source and target class
        mu_source = np.mean(x[y == k], axis=0)
        mu_target = np.mean(x[y == target], axis=0)

        # compute difference between source and target mean
        perturbations.append(mu_target - mu_source)

    return np.vstack(perturbations)



def perturb_samples(x, y, classes, target, output_dir='.'):
    # load pre-trained advgan model
    model = advgan.AdvGAN(
        n_inputs=x.shape[1],
        n_classes=len(classes),
        target=target,
        preload=True,
        output_dir=output_dir)

    # compute perturbations
    y_real = model.predict_target(x)
    score, x_fake, p, y_fake = model.score(x, y)

    print('perturbation accuracy: %0.3f' % (score))

    return p



if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', help='training data (samples x genes)', required=True)
    parser.add_argument('--train-labels', help='training labels', required=True)
    parser.add_argument('--perturb-data', help='perturb data (samples x genes)', required=True)
    parser.add_argument('--perturb-labels', help='perturb labels', required=True)
    parser.add_argument('--gene-sets', help='list of curated gene sets')
    parser.add_argument('--set', help='specific gene set to run')
    parser.add_argument('--target', help='target class', required=True)
    parser.add_argument('--perturb-classes', help='compute class perturbations', action='store_true')
    parser.add_argument('--perturb-samples', help='compute sample perturbations', action='store_true', default=True)
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
        gene_sets = {'all_genes': set(df_perturb.columns)}

    # select gene set
    try:
        name = args.set
        genes = gene_sets[name]
    except:
        print('error: gene set is not the subset file provided')
        sys.exit(1)

    # extract train/perturb data
    x_train = df_train[genes]
    x_perturb = df_perturb[genes]

    y_train = keras.utils.to_categorical(y_train, num_classes=len(classes))
    y_perturb = keras.utils.to_categorical(y_perturb, num_classes=len(classes))

    # normalize perturb data (using the train data)
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_perturb = scaler.transform(x_perturb)

    # compute class perturbations if specified
    if args.perturb_classes:
        # compute class perturbations (samples x classes)
        p_classes = perturb_classes(
            x_train,
            y_train,
            classes,
            args.target)

        # transpose perturbations to (classes x samples)
        p_classes = p_classes.T

        # save mean peturbations to dataframe
        p_classes = pd.DataFrame(
            data=p_classes,
            index=genes,
            columns=classes)

        utils.save_dataframe(
            '%s/%s.perturbations.classes.txt' % (args.output_dir, classes[args.target]),
            p_classes)

    # compute sample perturbations if specified
    if args.perturb_samples:
        # compute sample perturbations (samples x genes)
        p_samples = perturb_samples(
            x_perturb,
            y_perturb,
            classes,
            args.target,
            output_dir=args.output_dir)

        # transpose perturbations to (genes x samples)
        p_samples = p_samples.T

        # save sample perturbations to dataframe
        p_samples = pd.DataFrame(
            data=p_samples,
            index=genes,
            columns=df_perturb.index)

        utils.save_dataframe(
            '%s/%s.perturbations.samples.txt' % (args.output_dir, classes[args.target]),
            p_samples)
