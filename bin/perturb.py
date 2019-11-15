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



def perturb_mean_diff(x, y, classes, source, target, output_dir="."):
	# get mean of source and target class
	mu_source = get_class_mean(x, y, source)
	mu_target = get_class_mean(x, y, target)

	# save first perturbed sample to dataframe
	df_pert = pd.DataFrame(
		data=np.vstack([mu_source, mu_target - mu_source, mu_target, mu_target]).T,
		index=genes,
		columns=["X", "P", "X_adv", "mu_T"]
	)

	source_class = utils.clean_label(classes[source])
	target_class = utils.clean_label(classes[target])

	utils.save_dataframe("%s/%s_to_%s.txt" % (output_dir, source_class, target_class), df_pert)



def perturb_source_target(x, y, classes, source, target, mu_target, output_dir="."):
	print("attempting to perturb %s samples to %s..." % (classes[source], classes[target]))

	# extract samples in source class
	source_indices = np.where(np.argmax(y, axis=1) == source)
	x_source = x[source_indices]
	y_source = y[source_indices]

	if len(source_indices[0]) == 0:
		print("there are no test samples with class %s" % (classes[source]))
		return

	x_pl = tf.placeholder(tf.float32, [None, x_source.shape[-1]])
	y_pl = tf.placeholder(tf.float32, [None, y_source.shape[-1]])
	is_training = tf.placeholder(tf.bool, [])
	is_training_target = tf.placeholder(tf.bool, [])

	if target != -1:
		is_targeted = True
	else:
		is_targeted = False

	# generate pertubation, add to original, clip to valid expression level
	perturb, logit_perturb = generator.generator(x_pl, is_training)
	x_perturbed = perturb + x_pl
	x_perturbed = tf.clip_by_value(x_perturbed, 0, 1)

	# instantiate target model, create graphs for original and perturbed data
	f = target_model(n_input=x.shape[-1], n_classes=y.shape[-1])
	f_real_logits, f_real_probs = f.Model(x_pl, is_training_target)
	f_fake_logits, f_fake_probs = f.Model(x_perturbed, is_training_target)

	# get variables
	t_vars = tf.trainable_variables()
	f_vars = [var for var in t_vars if "Model_A" in var.name]
	g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")

	sess = tf.Session()

	# load checkpoints
	f_saver = tf.train.Saver(f_vars)
	g_saver = tf.train.Saver(g_vars)
	f_saver.restore(sess, tf.train.latest_checkpoint("%s/target_model/" % (output_dir)))
	g_saver.restore(sess, tf.train.latest_checkpoint("%s/generator/" % (output_dir)))

	if is_targeted:
		targets = np.full((y_source.shape[0],), target)
		batch_y = np.eye(y_pl.shape[-1])[targets]

	# generate perturbed samples
	x_pert, p = sess.run([x_perturbed, perturb], feed_dict={
		x_pl: x_source,
		y_pl: batch_y,
		is_training: False,
		is_training_target: False
	})

	# save first perturbed sample to dataframe
	df_pert = pd.DataFrame(
		data=np.vstack([x_source[0], p[0], x_pert[0], mu_target]).T,
		index=genes,
		columns=["X", "P", "X_adv", "mu_T"]
	)

	source_class = utils.clean_label(classes[source])
	target_class = utils.clean_label(classes[target])

	utils.save_dataframe("%s/%s_to_%s.npy" % (output_dir, source_class, target_class), df_pert)



def perturb(x, y, classes, target=-1, batch_size=64, output_dir="."):
	tf.reset_default_graph()

	x_pl = tf.placeholder(tf.float32, [None, x.shape[-1]])
	y_pl = tf.placeholder(tf.float32, [None, y.shape[-1]])
	is_training = tf.placeholder(tf.bool, [])
	is_training_target = tf.placeholder(tf.bool, [])

	if target != -1:
		is_targeted = True
	else:
		is_targeted = False

	# generate pertubation, add to original, clip to valid expression level
	perturb, logit_perturb = generator.generator(x_pl, is_training)
	x_perturbed = perturb + x_pl
	x_perturbed = tf.clip_by_value(x_perturbed, 0, 1)

	# instantiate target model, create graphs for original and perturbed data
	f = target_model(n_input=x.shape[-1], n_classes=y.shape[-1])
	f_real_logits, f_real_probs = f.Model(x_pl, is_training_target)
	f_fake_logits, f_fake_probs = f.Model(x_perturbed, is_training_target)

	# get variables
	t_vars = tf.trainable_variables()
	f_vars = [var for var in t_vars if "Model_A" in var.name]
	g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")

	sess = tf.Session()

	# load checkpoints
	f_saver = tf.train.Saver(f_vars)
	g_saver = tf.train.Saver(g_vars)
	f_saver.restore(sess, tf.train.latest_checkpoint("%s/target_model/" % (output_dir)))
	g_saver.restore(sess, tf.train.latest_checkpoint("%s/generator/" % (output_dir)))

	# calculate accuracy of target model on perturbed data
	correct_prediction = tf.equal(tf.argmax(f_fake_probs, 1), tf.argmax(y_pl, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	scores = []
	x_pert = []
	n_batches = math.ceil(len(x) / batch_size)

	for i in range(n_batches):
		batch_x, batch_y_og = utils.next_batch(x, y, batch_size, i)

		if is_targeted:
			targets = np.full((batch_y_og.shape[0],), target)
			batch_y = np.eye(y_pl.shape[-1])[targets]

		score, fake_l, x_p, p = sess.run([accuracy, f_fake_probs, x_perturbed, perturb], feed_dict={
			x_pl: batch_x,
			y_pl: batch_y,
			is_training: False,
			is_training_target: False
		})
		scores.append(score)
		x_pert.append(x_p)

	# print a sample original, perturbation, and original + perturbation
	np.set_printoptions(precision=4, suppress=True)

	print("original class is: %s" % (classes[np.argmax(batch_y_og, axis=1)[0]]))
	print(batch_x[0])
	print(p[0])
	print(x_p[0])

	np.save("%s/perturbed_%s.npy" % (output_dir, target), np.vstack(x_pert))

	print("test accuracy: %0.3f" % (sum(scores) / len(scores)))



if __name__ == "__main__":
	# parse command-line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--train-data", help="training data (samples x genes)", required=True)
	parser.add_argument("--train-labels", help="training labels", required=True)
	parser.add_argument("--test-data", help="test data (samples x genes)", required=True)
	parser.add_argument("--test-labels", help="test labels", required=True)
	parser.add_argument("--gene-sets", help="list of curated gene sets")
	parser.add_argument("--set", help="specific gene set to run")
	parser.add_argument("--target", help="target class", type=int, default=-1)
	parser.add_argument("--output-dir", help="Output directory", default=".")

	args = parser.parse_args()

	# load input data
	print("loading train/test data...")

	df_train = utils.load_dataframe(args.train_data)
	df_test = utils.load_dataframe(args.test_data)

	y_train, classes = utils.load_labels(args.train_labels)
	y_test, _ = utils.load_labels(args.test_labels, classes)

	print("loaded train data (%s genes, %s samples)" % (df_train.shape[1], df_train.shape[0]))
	print("loaded test data (%s genes, %s samples)" % (df_test.shape[1], df_test.shape[0]))

	# impute missing values
	min_value = df_train.min().min()

	df_train.fillna(value=min_value, inplace=True)
	df_test.fillna(value=min_value, inplace=True)

	# print target class if specified
	if args.target != -1:
		print("target class is: %s" % (classes[args.target]))

	# load gene sets file if it was provided
	if args.gene_sets != None:
		print("loading gene sets...")

		gene_sets = utils.load_gene_sets(args.gene_sets)
		gene_sets = utils.filter_gene_sets(gene_sets, df_test.columns)

		print("loaded %d gene sets" % (len(gene_sets)))
	else:
		gene_sets = {"all_genes": df_test.columns}

	# select gene set
	try:
		name = args.set
		genes = gene_sets[name]
	except:
		print("gene set is not the subset file provided")
		sys.exit(1)

	# extract train/test data
	x_train = df_train[genes]
	x_test = df_test[genes]

	y_train = utils.onehot_encode(y_train, classes)
	y_test = utils.onehot_encode(y_test, classes)

	# normalize test data (using the train data)
	scaler = sklearn.preprocessing.MinMaxScaler()
	scaler.fit(x_train)

	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	# get mu and sigma of target class
	mu_target = get_class_mean(x_train, y_train, args.target)

	# perturb all samples to target class
	perturb(x_test, y_test, classes, args.target, output_dir=args.output_dir)

	# perturb samples from each individual class to target
	for i in range(len(classes)):
		perturb_source_target(x_test, y_test, classes, i, args.target, mu_target, output_dir=args.output_dir)
