#!/usr/bin/env python3

import argparse
import math
import numpy as np
import pandas as pd
import os
import sklearn.model_selection
import sklearn.preprocessing
import sys
import tensorflow as tf

import generator
import utils
from target_models import Target_A as target_model



def get_class_mean(x, y, k):
    return x[np.argmax(y, axis=1) == k].mean(axis=0)



def perturb_mean_diff(x, y, target, classes):
    perturbations = []

    for k in range(len(classes)):
        # get mean of source and target class
        mu_source = get_class_mean(x, y, k)
        mu_target = get_class_mean(x, y, target)

        # compute difference between source and target mean
        perturbations.append(mu_target - mu_source)

    return np.vstack(perturbations).T



def perturb_advgan(x, y, target=-1, batch_size=32, output_dir='.'):
    x_pl = tf.placeholder(tf.float32, [None, x.shape[-1]])
    y_pl = tf.placeholder(tf.float32, [None, y.shape[-1]])
    is_training = tf.placeholder(tf.bool, [])
    is_training_target = tf.placeholder(tf.bool, [])

    if target != -1:
        is_targeted = True
    else:
        is_targeted = False

    # generate pertubation, add to original, clip to valid expression level
    p_pl, logit_perturb = generator.generator(x_pl, is_training)
    x_perturbed = x_pl + p_pl
    x_perturbed = tf.clip_by_value(x_perturbed, 0, 1)

    # instantiate target model, create graphs for original and perturbed data
    f = target_model(n_input=x.shape[-1], n_classes=y.shape[-1])
    f_real_logits, f_real_probs = f.Model(x_pl, is_training_target)
    f_fake_logits, f_fake_probs = f.Model(x_perturbed, is_training_target)

    # get variables
    t_vars = tf.trainable_variables()
    f_vars = [var for var in t_vars if 'Model_A' in var.name]
    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

    sess = tf.Session()

    # load checkpoints
    f_saver = tf.train.Saver(f_vars)
    g_saver = tf.train.Saver(g_vars)
    f_saver.restore(sess, tf.train.latest_checkpoint('%s/target_model/' % (output_dir)))
    g_saver.restore(sess, tf.train.latest_checkpoint('%s/generator/' % (output_dir)))

    # calculate accuracy of target model on perturbed data
    correct_prediction = tf.equal(tf.argmax(f_fake_probs, 1), tf.argmax(y_pl, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    # generate perturbed samples from original samples
    n_batches = math.ceil(len(x) / batch_size)
    scores = []
    perturbations = []

    for i in range(n_batches):
        batch_x, batch_y = utils.next_batch(x, y, batch_size, i)

        if is_targeted:
            targets = np.full((batch_y.shape[0],), target)
            batch_y_pert = np.eye(y_pl.shape[-1])[targets]

        score, _, batch_x_pert, batch_p = sess.run([accuracy, f_fake_probs, x_perturbed, p_pl], feed_dict={
            x_pl: batch_x,
            y_pl: batch_y_pert,
            is_training: False,
            is_training_target: False
        })
        scores.append(score)
        perturbations.append(batch_p)

    print('perturbation accuracy: %0.3f' % (sum(scores) / len(scores)))

    # perform post-processing of perturbations
    p = np.vstack(perturbations)
    p = np.clip(x + p, 0, 1) - x
    p = p.T

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
    parser.add_argument('--target', help='target class')
    parser.add_argument('--output-dir', help='Output directory', default='.')

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
        gene_sets = {'all_genes': df_perturb.columns}

    # select gene set
    try:
        name = args.set
        genes = gene_sets[name]
    except:
        print('gene set is not the subset file provided')
        sys.exit(1)

    # extract train/perturb data
    x_train = df_train[genes]
    x_perturb = df_perturb[genes]

    y_train = utils.onehot_encode(y_train, classes)
    y_perturb = utils.onehot_encode(y_perturb, classes)

    genes = x_train.columns

    # normalize perturb data (using the train data)
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_perturb = scaler.transform(x_perturb)

    # perturb each class mean to the target class
    mu_perturbed = perturb_mean_diff(x_train, y_train, args.target, classes)

    # save mean peturbations to dataframe
    df_perturbed = pd.DataFrame(
        data=mu_perturbed,
        index=genes,
        columns=classes
    )

    utils.save_dataframe('%s/%s.perturbations.means.txt' % (args.output_dir, classes[args.target]), df_perturbed)

    # perturb all samples to target class
    perturbations = perturb_advgan(x_perturb, y_perturb, args.target, output_dir=args.output_dir)

    # save sample perturbations to dataframe
    df_perturbed = pd.DataFrame(
        data=perturbations,
        index=genes,
        columns=df_perturb.index
    )

    utils.save_dataframe('%s/%s.perturbations.samples.txt' % (args.output_dir, classes[args.target]), df_perturbed)
