#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.model_selection
import sklearn.preprocessing
import sys
from tensorflow import keras

import advgan
import target_model
import utils



if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='input dataset (samples x genes)', required=True)
    parser.add_argument('--labels', help='list of sample labels', required=True)
    parser.add_argument('--gene-sets', help='list of curated gene sets')
    parser.add_argument('--target', help='target class', required=True)
    parser.add_argument('--target-cov', help='covariance matrix for target distribution', choices=['diagonal', 'full'], default='full')
    parser.add_argument('--set', help='gene set to run')
    parser.add_argument('--output-dir', help='Output directory', default='.')
    parser.add_argument('--test-size', help='proportional test set size', type=float, default=0.2)
    parser.add_argument('--epochs', help='number of training epochs', type=int, default=150)
    parser.add_argument('--batch-size', help='minibatch size', type=int, default=128)

    args = parser.parse_args()

    # load input data
    print('loading input dataset...')

    df = utils.load_dataframe(args.dataset)
    df_samples = df.index
    df_genes = df.columns

    labels, classes = utils.load_labels(args.labels)

    print('loaded input dataset (%s genes, %s samples)' % (df.shape[1], df.shape[0]))

    # impute missing values
    df.fillna(value=df.min().min(), inplace=True)

    # sort labels to match data if needed
    if (df.index != labels.index).any():
        print('warning: data and labels are not ordered the same, re-ordering labels')
        labels = labels.loc[df.index]

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
        gene_sets = utils.filter_gene_sets(gene_sets, df_genes)

        print('loaded %d gene sets' % (len(gene_sets)))
    else:
        gene_sets = {'all_genes': set(df_genes)}

    # select gene set
    try:
        name = args.set
        genes = gene_sets[name]
    except:
        print('error: gene set is not the subset file provided')
        sys.exit(1)

    # print warning about covariance matrix if gene set is large
    if len(genes) > 10000 and args.target_cov == 'full':
        print('warning: gene set is very large, consider using --target-cov=diagonal')

    # extract dataset
    X = df[genes]
    y = keras.utils.to_categorical(labels, num_classes=len(classes))

    # create train/test sets
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=args.test_size)

    # normalize dataset
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # adjust batch size if necessary
    if args.batch_size > len(x_train):
        print('warning: reducing batch size to train set size, consider reducing further')
        args.batch_size = len(x_train)

    # compute target distribution params
    target_data = x_train[np.argmax(y_train, axis=1) == args.target]
    target_mu = np.mean(target_data, axis=0)

    if args.target_cov == 'full':
        target_cov = np.cov(target_data, rowvar=False)
    elif args.target_cov == 'diagonal':
        target_cov = np.std(target_data, axis=0)

    # create advgan model
    model = advgan.AdvGAN(
        n_inputs=len(genes),
        n_classes=len(classes),
        target=args.target,
        target_mu=target_mu,
        target_cov=target_cov,
        output_dir=args.output_dir)

    model.compile()

    # train model
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size)

    # plot the training history
    for label, y in history.history.items():
        y_min = min(y_i for y_i in y if y_i > 0)
        y = np.maximum(y, y_min)
        plt.plot(y, label=label)

    plt.yscale('log')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('%s/%d_history.png' % (args.output_dir, args.target))
    plt.close()

    # evaluate model
    y_real = model.predict_target(x_test)
    loss, accuracy = model.target_model.evaluate(x_test, y_test, verbose=False)

    score, x_fake, p, y_fake = model.score(x_test, y_test)

    print('original labels:      ', np.argmax(y_test[:32], axis=1))
    print('original predictions: ', np.argmax(y_real[:32], axis=1))
    print('perturbed predictions:', np.argmax(y_fake[:32], axis=1))
    print()
    print('target model accuracy: %0.3f' % (accuracy))
    print('perturbation accuracy: %0.3f' % (score))

    # save model
    model.save()
