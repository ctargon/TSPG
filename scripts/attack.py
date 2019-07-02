import argparse
import numpy as np
import os
import sklearn.model_selection
import sklearn.preprocessing
import sys
import tensorflow as tf

import generator
import utils
from target_models import Target_A as target_model



def cleanse_label(label):
	label = label.replace(" ", "_")
	label = label.replace("-", "")
	label = label.replace("(", "")
	label = label.replace(")", "")
	return label



def attack_source_target(x, y, classes, source, target, target_mu):
	source_indices = np.where(np.argmax(y, axis=1) == source)
	x_source = x[source_indices]
	y_source = y[source_indices]

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
	f_saver.restore(sess, tf.train.latest_checkpoint("./weights/target_model/Model_A/"))
	g_saver.restore(sess, tf.train.latest_checkpoint("./weights/generator/"))

	if is_targeted:
		targets = np.full((y_source.shape[0],), target)
		batch_y = np.eye(y_pl.shape[-1])[targets]

	x_pert, p = sess.run([x_perturbed, perturb], feed_dict={
		x_pl: x_source,
		y_pl: batch_y,
		is_training: False,
		is_training_target: False
	})

	print("source class is: %s" % (classes[source]))
	print("X:")
	print(x_source[0])
	print("P:")
	print(p[0])
	print("X_adv:")
	print(x_pert[0])
	print("target_mu:")
	print(target_mu)

	# save the results in X, P, X_adv, target_mu order
	results = np.vstack([x_source[0], p[0], x_pert[0], target_mu])

	source_class = cleanse_label(classes[source])
	target_class = cleanse_label(classes[target])

	np.save("%s_to_%s.npy" % (source_class, target_class), results)



def attack(x_train, y_train, target=-1, batch_size=64):
	x_pl = tf.placeholder(tf.float32, [None, x_train.shape[-1]])
	y_pl = tf.placeholder(tf.float32, [None, y_train.shape[-1]])
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
	f = target_model(n_input=x_train.shape[-1], n_classes=y_train.shape[-1])
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
	f_saver.restore(sess, tf.train.latest_checkpoint("./weights/target_model/Model_A/"))
	g_saver.restore(sess, tf.train.latest_checkpoint("./weights/generator/"))

	# calculate accuracy of target model on perturbed data
	correct_prediction = tf.equal(tf.argmax(f_fake_probs, 1), tf.argmax(y_pl, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	scores = []
	x_pert = []
	n_batches = int(len(x_train) / batch_size)

	for i in range(n_batches):
		batch_x, batch_y_og = utils.next_batch(x_train, y_train, batch_size, i)

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

	np.save("perturbed_%s.npy" % (target), np.vstack(x_pert))

	print("test accuracy: %0.3f" % (sum(scores) / len(scores)))



if __name__ == "__main__":
	# parse command-line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", help="input dataset (samples x genes)", required=True)
	parser.add_argument("--labels", help="list of sample labels", required=True)
	parser.add_argument("--gene-sets", help="list of curated gene sets")
	parser.add_argument("--target", help="target class", type=int, default=-1)

	args = parser.parse_args()

	# load input data
	print("loading input dataset...")

	df = utils.load_dataframe(args.dataset)
	df_samples = df.index
	df_genes = df.columns

	labels, classes = utils.load_labels(args.labels)

	print("loaded input dataset (%s genes, %s samples)" % (df.shape[1], df.shape[0]))

	# print target class if specified
	if args.target != -1:
		print("target class is: %s" % (classes[args.target]))

	# load gene sets file if it was provided
	if args.gene_sets != None:
		print("loading gene sets...")

		gene_sets = utils.load_gene_sets(args.gene_sets)

		print("loaded %d gene sets" % (len(gene_sets)))

		# remove genes which do not exist in the dataset
		genes = list(set(sum([genes for (name, genes) in gene_sets], [])))
		missing_genes = [g for g in genes if g not in df_genes]

		gene_sets = [(name, [g for g in genes if g in df_genes]) for (name, genes) in gene_sets]

		print("%d / %d genes from gene sets were not found in the input dataset" % (len(missing_genes), len(genes)))
	else:
		gene_sets = []

	# perform attack for each gene set
	for name, genes in gene_sets:
		# extract dataset
		x = df[genes]
		y = utils.onehot_encode(labels, range(len(classes)))

		# normalize dataset
		x = sklearn.preprocessing.MinMaxScaler().fit_transform(x)

		# get mu and sigma of target class feature vectors
		target_data = x[np.argmax(y, axis=1) == args.target]
		target_mu = np.mean(target_data, axis=0)

		# perform attack
		attack(x, y, target=args.target)

		# perform source-to-target attack for each source class
		for i in range(len(classes)):
			attack_source_target(x, y, classes, i, args.target, target_mu)
