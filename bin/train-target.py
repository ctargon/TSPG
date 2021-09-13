#!/usr/bin/env python3

import argparse
import sklearn.model_selection
import sklearn.preprocessing

import target_model
import utils



if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='input dataset (samples x genes)', required=True)
    parser.add_argument('--labels', help='list of sample labels', required=True)
    parser.add_argument('--gene-sets', help='list of curated gene sets')
    parser.add_argument('--set', help='specific gene set to run')
    parser.add_argument('--output-dir', help='output directory', default='.')
    parser.add_argument('--test-size', help='proportional test set size', type=float, default=0.2)
    parser.add_argument('--epochs', help='number of training epochs', type=int, default=50)
    parser.add_argument('--batch-size', help='minibatch size', type=int, default=32)

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

    # extract dataset
    X = df[genes]
    y = utils.onehot_encode(labels, classes)

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

    # create target model
    model = target_model.TargetModel(
        n_inputs=x_train.shape[1],
        n_classes=len(classes),
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir)

    # train model
    model.fit(x_train, y_train)

    # evaluate model
    score = model.score(x_test, y_test)

    print('test accuracy: %0.3f' % (score))

    # save model
    model.save()