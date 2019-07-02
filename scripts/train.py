"""
	AdvGAN architecture

	ref: https://arxiv.org/pdf/1801.02610.pdf
"""
import argparse
import numpy as np
import os
import sklearn.model_selection
import sklearn.preprocessing
import sys
import tensorflow as tf

import utils
from generator import generator
from discriminator import discriminator
from target_models import Target_A as target_model



# loss function to encourage misclassification after perturbation from carlini&wagner
def adv_loss(preds, labels, is_targeted):
	real = tf.reduce_sum(labels * preds, 1)
	other = tf.reduce_max((1 - labels) * preds - (labels * 10000), 1)
	if is_targeted:
		return tf.reduce_sum(tf.maximum(0.0, other - real))
	return tf.reduce_sum(tf.maximum(0.0, real - other))



# loss function to influence the perturbation to be as close to 0 as possible
def perturb_loss(preds, thresh=0.3, epsilon=1e-8):
	zeros = tf.zeros((tf.shape(preds)[0]))
	#return tf.reduce_mean(tf.maximum(zeros, tf.norm(tf.reshape(preds, (tf.shape(preds)[0], -1)), axis=1) - thresh))
	reshape = tf.reshape(preds, (tf.shape(preds)[0], -1))
	norm = tf.norm(reshape, axis=1)
	norm = norm + epsilon
	return tf.reduce_mean(norm)



# function that defines ops, graphs, and training procedure for AdvGAN framework
def AdvGAN(x_train, y_train, x_test, y_test, t_mu, t_cov, epochs=50, batch_size=32, target=-1):
	# placeholder definitions
	x_pl = tf.placeholder(tf.float32, [None, x_train.shape[-1]])
	t = tf.placeholder(tf.float32, [None, y_train.shape[-1]])
	is_training = tf.placeholder(tf.bool, [])
	target_is_training = tf.placeholder(tf.bool, [])

	#-----------------------------------------------------------------------------------
	# MODEL DEFINITIONS
	if target != -1:
		is_targeted = True
	else:
		is_targeted = False

	# gather target model
	f = target_model(n_input=x_train.shape[-1], n_classes=y_train.shape[-1])

	thresh = 0.3

	# generate perturbation, add to original input image(s)
	perturb, logit_perturb = generator(x_pl, is_training)#generator(x_pl, is_training)#tf.clip_by_value(generator(x_pl, is_training), -thresh, thresh)
	x_perturbed = perturb + x_pl
	x_perturbed = tf.clip_by_value(x_perturbed, 0, 1)

	# pass real and perturbed image to discriminator and the target model
	d_real_logits, d_real_probs = discriminator(x_pl, is_training)
	d_fake_logits, d_fake_probs = discriminator(x_perturbed, is_training)
	
	# pass real and perturbed images to the model we are trying to fool
	f_real_logits, f_real_probs = f.Model(x_pl, target_is_training)
	f_fake_logits, f_fake_probs = f.Model(x_perturbed, target_is_training)
	
	# generate labels for discriminator (optionally smooth labels for stability)
	smooth = 0.0
	d_labels_real = tf.ones_like(d_real_probs) * (1 - smooth)
	d_labels_fake = tf.zeros_like(d_fake_probs)

	#-----------------------------------------------------------------------------------
	# LOSS DEFINITIONS
	# discriminator loss
	d_loss_real = tf.losses.mean_squared_error(predictions=d_real_probs, labels=d_labels_real)
	d_loss_fake = tf.losses.mean_squared_error(predictions=d_fake_probs, labels=d_labels_fake)
	d_loss = d_loss_real + d_loss_fake

	# generator loss
	g_loss_fake = tf.losses.mean_squared_error(predictions=d_fake_probs, labels=tf.ones_like(d_fake_probs))

	# perturbation loss (minimize overall perturbation)
	l_perturb = perturb_loss(perturb, 1.0)

	# adversarial loss (encourage misclassification)
	l_adv = adv_loss(f_fake_probs, t, is_targeted)

	# loss minimizing L1 distance between target class average and perturbed vector
	# this is used to encourage realism of sample
	target_normal = tf.placeholder(tf.float32, [None, x_train.shape[-1]])
	l_tar_dist = tf.reduce_mean(tf.norm(target_normal - x_perturbed, axis=1, ord=1))

	# weights for generator loss function
	alpha = 1.0
	beta = 1.0
	g_loss = l_adv + alpha*g_loss_fake + l_tar_dist + beta*l_perturb

	# ----------------------------------------------------------------------------------
	# gather variables for training/restoring
	t_vars = tf.trainable_variables()
	f_vars = [var for var in t_vars if "Model_A" in var.name]
	d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")
	g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")

	# define optimizers for discriminator and generator
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		d_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(d_loss, var_list=d_vars)
		g_opt = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(g_loss, var_list=g_vars)

	# create saver objects for the target model, generator, and discriminator
	saver = tf.train.Saver(f_vars)
	g_saver = tf.train.Saver(g_vars)
	d_saver = tf.train.Saver(d_vars)

	init  = tf.global_variables_initializer()

	sess  = tf.Session()
	sess.run(init)

	# load the pretrained target model
	try:
		saver.restore(sess, tf.train.latest_checkpoint("./weights/target_model/Model_A/"))
	except:
		print("make sure to train the target model first...")
		sys.exit(1)

	total_batches = int(len(y_train) / batch_size)

	for epoch in range(0, epochs):
		# shuffle training data
		x_train, y_train = utils.shuffle(x_train, y_train)

		loss_D_sum = 0.0
		loss_G_fake_sum = 0.0
		loss_perturb_sum = 0.0
		loss_adv_sum = 0.0
		loss_target_norm = 0.0

		target_normal_np = np.random.multivariate_normal(t_mu, t_cov, (batch_size))
		target_normal_np = np.clip(target_normal_np, 0, 1)
	
		for i in range(total_batches):
			# extract batch
			batch_x, batch_y = utils.next_batch(x_train, y_train, batch_size, i)

			# if targeted, create one hot vectors of the target
			if is_targeted:
				targets = np.full((batch_y.shape[0],), target)
				batch_y = np.eye(y_train.shape[-1])[targets]

			# train the discriminator first n times
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
						t: batch_y, 
						target_normal: target_normal_np,
						is_training: True, 
						target_is_training: False
					})

			loss_D_sum += loss_D_batch
			loss_G_fake_sum += loss_G_fake_batch
			loss_perturb_sum += loss_perturb_batch
			loss_adv_sum += loss_adv_batch
			loss_target_norm += loss_target_batch

		print("epoch %d:\n\
				loss_D: %.3f, loss_G_fake: %.3f\n\
				loss_perturb: %.3f, loss_adv: %.3f\n\
				loss_tar_norm: %.3f\n" % (
			epoch + 1,
			loss_D_sum / total_batches,
			loss_G_fake_sum / total_batches,
			loss_perturb_sum / total_batches,
			loss_adv_sum / total_batches,
			loss_target_norm / total_batches
		))

		if epoch % 10 == 0:
			g_saver.save(sess, "weights/generator/gen.ckpt")
			d_saver.save(sess, "weights/discriminator/disc.ckpt")

	# quick sample to see some outputs
	rawpert, pert, fake_l, real_l = sess.run([perturb, x_perturbed, f_fake_probs, f_real_probs], feed_dict={
		x_pl: x_test[:32],
		is_training: False,
		target_is_training: False
	})

	print("Original Labels:\n" + str(np.argmax(y_test[:32], axis=1)))
	print("Original Target Model Classification:\n" + str(np.argmax(real_l, axis=1)))
	print("Perturbed Target Model Classification:\n" + str(np.argmax(fake_l, axis=1)))

	# evaluate the test set
	correct_prediction = tf.equal(tf.argmax(f_fake_probs, 1), tf.argmax(t, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	scores = []
	total_batches_test = int(len(y_test) / batch_size)

	for i in range(total_batches_test):
		batch_x, batch_y = utils.next_batch(x_test, y_test, batch_size, i)
		score, x_pert = sess.run([accuracy, x_perturbed], feed_dict={
			x_pl: batch_x,
			t: batch_y,
			is_training: False,
			target_is_training: False
		})
		scores.append(score)

	print("test accuracy: %0.3f" % (sum(scores) / len(scores)))

	print("finished training, saving weights")
	g_saver.save(sess, "weights/generator/gen.ckpt")
	d_saver.save(sess, "weights/discriminator/disc.ckpt")



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
		print("targeting class \'%s\'" % (classes[args.target]))

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

	# train a model for each gene set
	for name, genes in gene_sets:
		# extract dataset
		X = df[genes]
		y = utils.onehot_encode(labels, classes)

		# create train/test sets
		x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)

		# normalize dataset
		Scaler = sklearn.preprocessing.MinMaxScaler
		x_train = Scaler().fit_transform(x_train)
		x_test = Scaler().fit_transform(x_test)

		# get mu and sigma of target class feature vectors
		target_data = x_train[np.argmax(y_train, axis=1) == args.target]
		target_mu = np.mean(target_data, axis=0)
		target_cov = np.cov(target_data, rowvar=False)

		AdvGAN(x_train, y_train, x_test, y_test, target_mu, target_cov, epochs=150, batch_size=128, target=args.target)
