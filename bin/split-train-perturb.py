#!/usr/bin/env python3

import argparse
import pandas as pd
import sklearn.model_selection



if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Split a dataset into train/perturb sets')
    parser.add_argument('--dataset', help='input dataset (samples x genes)', required=True)
    parser.add_argument('--labels', help='list of sample labels', required=True)
    parser.add_argument('--train-size', help='training set proportion', type=float, default=0.8)

    args = parser.parse_args()

    # load input dataset
    x = pd.read_csv(args.dataset, sep='\t', index_col=0)
    y = pd.read_csv(args.labels, sep='\t', index_col=0, header=None)

    # get filename prefix
    prefix = args.dataset.split('.')[0]

    # split dataset into train/perturb sets
    x_train, x_perturb, y_train, y_perturb = sklearn.model_selection.train_test_split(x, y, test_size=1 - args.train_size)

    # save train/perturb data to file
    x_train.to_csv('%s.train.emx.txt' % (prefix), sep='\t')
    y_train.to_csv('%s.train.labels.txt' % (prefix), sep='\t', header=None)
    x_perturb.to_csv('%s.perturb.emx.txt' % (prefix), sep='\t')
    y_perturb.to_csv('%s.perturb.labels.txt' % (prefix), sep='\t', header=None)
