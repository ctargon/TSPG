#!/usr/bin/env python3

'''
    AdvGAN architecture

    ref: https://arxiv.org/pdf/1801.02610.pdf
'''
import argparse
import numpy as np
import os
import sklearn.model_selection
import sklearn.preprocessing
import sys
import tensorflow as tf

import discriminator
import generator
import utils
from target_models import Target_A as Target



# loss function to encourage misclassification after perturbation from carlini&wagner
def adv_loss(preds, labels, is_targeted):
    real = tf.reduce_sum(labels * preds, 1)
    other = tf.reduce_max((1 - labels) * preds - (labels * 10000), 1)
    if is_targeted:
        return tf.reduce_sum(tf.maximum(0.0, other - real))
    return tf.reduce_sum(tf.maximum(0.0, real - other))



# loss function to influence the perturbation to be as close to 0 as possible
def perturb_loss(preds, epsilon=1e-8):
    zeros = tf.zeros((tf.shape(preds)[0]))
    reshape = tf.reshape(preds, (tf.shape(preds)[0], -1))
    norm = tf.norm(reshape, axis=1)
    norm = norm + epsilon
    return tf.reduce_mean(norm)



# function that defines ops, graphs, and training procedure for AdvGAN framework
def AdvGAN(x_train, y_train, x_test, y_test, t_mu, t_cov, target=-1, epochs=50, batch_size=32, output_dir='.'):
    tf.reset_default_graph()

    # placeholder definitions
    x_pl = tf.placeholder(tf.float32, [None, x_train.shape[-1]])
    y_pl = tf.placeholder(tf.float32, [None, y_train.shape[-1]])
    is_training = tf.placeholder(tf.bool, [])
    target_is_training = tf.placeholder(tf.bool, [])

    #---------------------------------------------------------------------------
    # MODEL DEFINITIONS
    #---------------------------------------------------------------------------

    if target != -1:
        is_targeted = True
    else:
        is_targeted = False

    # gather target model
    f = Target(n_input=x_train.shape[-1], n_classes=y_train.shape[-1])

    # generate perturbation, add to original input image(s)
    perturb, logit_perturb = generator.generator(x_pl, is_training)
    x_perturbed = perturb + x_pl
    x_perturbed = tf.clip_by_value(x_perturbed, 0, 1)

    # pass real and perturbed image to discriminator and the target model
    d_real_logits, d_real_probs = discriminator.discriminator(x_pl, is_training)
    d_fake_logits, d_fake_probs = discriminator.discriminator(x_perturbed, is_training)

    # pass real and perturbed images to the model we are trying to fool
    f_real_logits, f_real_probs = f.Model(x_pl, target_is_training)
    f_fake_logits, f_fake_probs = f.Model(x_perturbed, target_is_training)

    # generate labels for discriminator (optionally smooth labels for stability)
    smooth = 0.0
    d_labels_real = tf.ones_like(d_real_probs) * (1 - smooth)
    d_labels_fake = tf.zeros_like(d_fake_probs)

    #---------------------------------------------------------------------------
    # LOSS DEFINITIONS
    #---------------------------------------------------------------------------

    # discriminator loss
    d_loss_real = tf.losses.mean_squared_error(predictions=d_real_probs, labels=d_labels_real)
    d_loss_fake = tf.losses.mean_squared_error(predictions=d_fake_probs, labels=d_labels_fake)
    d_loss = d_loss_real + d_loss_fake

    # generator loss
    g_loss_fake = tf.losses.mean_squared_error(predictions=d_fake_probs, labels=tf.ones_like(d_fake_probs))

    # perturbation loss (minimize overall perturbation)
    l_perturb = perturb_loss(perturb, 1.0)

    # adversarial loss (encourage misclassification)
    l_adv = adv_loss(f_fake_probs, y_pl, is_targeted)

    # loss minimizing L1 distance between target class average and perturbed vector
    # this is used to encourage realism of sample
    target_normal = tf.placeholder(tf.float32, [None, x_train.shape[-1]])
    l_tar_dist = tf.reduce_mean(tf.norm(target_normal - x_perturbed, axis=1, ord=1))

    # weights for generator loss function
    alpha = 1.0
    beta = 1.0
    g_loss = l_adv + alpha*g_loss_fake + l_tar_dist + beta*l_perturb

    # gather variables for training/restoring
    t_vars = tf.trainable_variables()
    f_vars = [var for var in t_vars if 'Model_A' in var.name]
    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

    # define optimizers for discriminator and generator
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        d_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(d_loss, var_list=d_vars)
        g_opt = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(g_loss, var_list=g_vars)

    # create saver objects for the target model, generator, and discriminator
    saver = tf.train.Saver(f_vars)
    g_saver = tf.train.Saver(g_vars)
    d_saver = tf.train.Saver(d_vars)

    #---------------------------------------------------------------------------
    # TRAINING
    #---------------------------------------------------------------------------

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # load the pretrained target model
    try:
        saver.restore(sess, tf.train.latest_checkpoint('%s/target_model' % (output_dir)))
    except:
        print('error: target model not found')
        sys.exit(1)

    # intialize directory for target class
    os.makedirs('%s/%s' % (output_dir, str(target)), exist_ok=True)

    # train the advgan model
    n_batches = len(x_train) // batch_size

    for epoch in range(epochs):
        # shuffle training data
        x_train, y_train = utils.shuffle(x_train, y_train)

        loss_D = 0.0
        loss_G_fake = 0.0
        loss_perturb = 0.0
        loss_adv = 0.0
        loss_target_norm = 0.0

        target_normal_np = np.random.multivariate_normal(t_mu, t_cov, (batch_size))
        target_normal_np = np.clip(target_normal_np, 0, 1)

        for i in range(n_batches):
            # extract batch
            batch_x, batch_y = utils.next_batch(x_train, y_train, batch_size, i)

            # if targeted, create one hot vectors of the target
            if is_targeted:
                targets = np.full((batch_y.shape[0],), target)
                batch_y = np.eye(y_train.shape[-1])[targets]

            # train the discriminator n times
            for _ in range(1):
                _, loss_D_batch = sess.run([d_opt, d_loss], feed_dict={
                    x_pl: batch_x,
                    target_normal: target_normal_np,
                    is_training: True
                })

            # train the generator n times
            for _ in range(1):
                _, loss_G_fake_batch, loss_adv_batch, loss_perturb_batch, loss_target_batch = \
                    sess.run([g_opt, g_loss_fake, l_adv, l_perturb, l_tar_dist], feed_dict={
                        x_pl: batch_x,
                        y_pl: batch_y,
                        target_normal: target_normal_np,
                        is_training: True,
                        target_is_training: False
                    })

            loss_D += loss_D_batch
            loss_G_fake += loss_G_fake_batch
            loss_perturb += loss_perturb_batch
            loss_adv += loss_adv_batch
            loss_target_norm += loss_target_batch

        loss_D /= n_batches
        loss_G_fake /= n_batches
        loss_perturb /= n_batches
        loss_adv /= n_batches
        loss_target_norm /= n_batches

        print('epoch %d:' % (epoch + 1))
        print('  loss_D: %8.3f, loss_G_fake: %8.3f, loss_perturb: %8.3f, loss_adv: %8.3f, loss_target_norm: %8.3f' % (
            loss_D,
            loss_G_fake,
            loss_perturb,
            loss_adv,
            loss_target_norm
        ))
        print()

        if epoch % 10 == 0:
            g_saver.save(sess, '%s/%s/generator/generator.ckpt' % (output_dir, str(target)))
            d_saver.save(sess, '%s/%s/discriminator/discriminator.ckpt' % (output_dir, str(target)))

    # compute perturbation accuracy
    _, pert, y_fake, y_real = sess.run([perturb, x_perturbed, f_fake_probs, f_real_probs], feed_dict={
        x_pl: x_test,
        is_training: False,
        target_is_training: False
    })

    score = sum(np.argmax(y_fake, axis=1) == target) / len(y_fake)

    print('original labels:       ', np.argmax(y_test[:32], axis=1))
    print('original predictions:  ', np.argmax(y_real[:32], axis=1))
    print('perturbed predictions: ', np.argmax(y_fake[:32], axis=1))
    print()
    print('perturbation accuracy: %0.3f' % (score))

    g_saver.save(sess, '%s/%s/generator/generator.ckpt' % (output_dir, str(target)))
    d_saver.save(sess, '%s/%s/discriminator/discriminator.ckpt' % (output_dir, str(target)))



if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='input dataset (samples x genes)', required=True)
    parser.add_argument('--labels', help='list of sample labels', required=True)
    parser.add_argument('--gene-sets', help='list of curated gene sets')
    parser.add_argument('--target', help='target class')
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
        gene_sets = utils.filter_gene_sets(gene_sets, df_genes)

        print('loaded %d gene sets' % (len(gene_sets)))
    else:
        gene_sets = {'all_genes': df_genes}

    # select gene set
    try:
        name = args.set
        genes = gene_sets[name]
    except:
        print('gene set is not the subset file provided')
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
        print('info: reducing batch size to train set size, consider reducing further')
        args.batch_size = len(x_train)

    # get mu and sigma of target class feature vectors
    target_data = x_train[np.argmax(y_train, axis=1) == args.target]
    target_mu = np.mean(target_data, axis=0)
    target_cov = np.cov(target_data, rowvar=False)

    # train advgan model
    AdvGAN(
        x_train, y_train,
        x_test, y_test,
        target_mu, target_cov,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target=args.target,
        output_dir=args.output_dir)
