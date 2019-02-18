'''
	AdvGAN architecture

	ref: https://arxiv.org/pdf/1801.02610.pdf
'''


import tensorflow as tf
from sklearn import preprocessing
import numpy as np
import os, sys, argparse

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

from models.generator import generator
from models.discriminator import discriminator
from models.target_models import Target_A as target_model
from utils.dataset import DataContainer as DC
from utils.utils import load_data, read_subset_file


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
def AdvGAN(dataset, epochs=50, batch_size=32, target=-1):
	# placeholder definitions
	x_pl = tf.placeholder(tf.float32, [None, dataset.train.data.shape[-1]]) # data placeholder
	t = tf.placeholder(tf.float32, [None, dataset.train.labels.shape[-1]]) # target placeholder
	is_training = tf.placeholder(tf.bool, [])
	target_is_training = tf.placeholder(tf.bool, [])

	#-----------------------------------------------------------------------------------
	# MODEL DEFINITIONS
	is_targeted = False
	if target in range(0, dataset.train.labels.shape[-1]):
		print('target is: ' + str(dataset.label_names_ordered[target]))
		is_targeted = True

	# gather target model
	f = target_model(n_input=dataset.train.data.shape[-1], n_classes=dataset.train.labels.shape[-1])

	thresh = 0.3

	# generate perturbation, add to original input image(s)
	perturb, logit_perturb = generator(x_pl, is_training)#tf.clip_by_value(generator(x_pl, is_training), -thresh, thresh)
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

	# weights for generator loss function
	alpha = 1.0
	beta = 1.0
	g_loss = l_adv + alpha*g_loss_fake + beta*l_perturb 

	# ----------------------------------------------------------------------------------
	# gather variables for training/restoring
	t_vars = tf.trainable_variables()
	f_vars = [var for var in t_vars if 'Model_A' in var.name]
	d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
	g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

	# define optimizers for discriminator and generator
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		d_opt = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(d_loss, var_list=d_vars)
		g_opt = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(g_loss, var_list=g_vars)

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

	total_batches = int(dataset.train.num_examples / batch_size)

	for epoch in range(0, epochs):

		dataset.shuffle()
		loss_D_sum = 0.0
		loss_G_fake_sum = 0.0
		loss_perturb_sum = 0.0
		loss_adv_sum = 0.0

		for i in range(total_batches):

			batch_x, batch_y = dataset.train.next_batch(batch_size, i)

			# if targeted, create one hot vectors of the target
			if is_targeted:
				targets = np.full((batch_y.shape[0],), target)
				batch_y = np.eye(dataset.train.labels.shape[-1])[targets]

			# train the discriminator first n times
			for _ in range(1):
				_, loss_D_batch = sess.run([d_opt, d_loss], feed_dict={x_pl: batch_x, \
																	   is_training: True})

			# train the generator n times
			for _ in range(1):
				_, loss_G_fake_batch, loss_adv_batch, loss_perturb_batch= \
									sess.run([g_opt, g_loss_fake, l_adv, l_perturb], \
												feed_dict={x_pl: batch_x, \
														   t: batch_y, \
														   is_training: True, \
														   target_is_training: False})
			loss_D_sum += loss_D_batch
			loss_G_fake_sum += loss_G_fake_batch
			loss_perturb_sum += loss_perturb_batch
			loss_adv_sum += loss_adv_batch

		print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f, \
				\nloss_perturb: %.3f, loss_adv: %.3f, \n" %
				(epoch + 1, loss_D_sum/total_batches, loss_G_fake_sum/total_batches,
				loss_perturb_sum/total_batches, loss_adv_sum/total_batches))

		if epoch % 10 == 0:
			g_saver.save(sess, "weights/generator/gen.ckpt")
			d_saver.save(sess, "weights/discriminator/disc.ckpt")

	# quick sample to see some outputs
	rawpert, pert, fake_l, real_l = sess.run([perturb, x_perturbed, f_fake_probs, f_real_probs], \
												feed_dict={x_pl: dataset.test.data[:32], \
														   is_training: False, \
														   target_is_training: False})
	print('LA: ' + str(np.argmax(dataset.test.labels[:32], axis=1)))
	print('OG: ' + str(np.argmax(real_l, axis=1)))
	print('PB: ' + str(np.argmax(fake_l, axis=1)))


	# evaluate the test set
	correct_prediction = tf.equal(tf.argmax(f_fake_probs, 1), tf.argmax(t, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	accs = []
	total_batches_test = int(dataset.test.num_examples / batch_size)
	for i in range(total_batches_test):
		batch_x, batch_y = dataset.test.next_batch(batch_size, i)
		acc, x_pert = sess.run([accuracy, x_perturbed], feed_dict={x_pl: batch_x, \
																   t: batch_y, \
																   is_training: False, \
																   target_is_training: False})
		accs.append(acc)

	print('accuracy of test set: {}'.format(sum(accs) / len(accs)))

	print('finished training, saving weights')
	g_saver.save(sess, "weights/generator/gen.ckpt")
	d_saver.save(sess, "weights/discriminator/disc.ckpt")





def attack(dataset, batch_size=64, thresh=0.3, target=-1):
	x_pl = tf.placeholder(tf.float32, [None, dataset.test.data.shape[-1]]) # image placeholder
	t = tf.placeholder(tf.float32, [None, dataset.test.labels.shape[-1]]) # target placeholder
	is_training = tf.placeholder(tf.bool, [])
	is_training_target = tf.placeholder(tf.bool, [])

	is_targeted = False
	if target in range(0, dataset.test.labels.shape[-1]):
		print('target is: ' + str(dataset.label_names_ordered[target]))
		is_targeted = True

	perturb, logit_perturb = generator(x_pl, is_training)#tf.clip_by_value(generator(x_pl, is_training), -thresh, thresh)
	x_perturbed = perturb + x_pl
	x_perturbed = tf.clip_by_value(x_perturbed, 0, 1)

	f = target_model(n_input=dataset.train.data.shape[-1], n_classes=dataset.train.labels.shape[-1])
	f_real_logits, f_real_probs = f.Model(x_pl, is_training_target)
	f_fake_logits, f_fake_probs = f.Model(x_perturbed, is_training_target)

	t_vars = tf.trainable_variables()
	f_vars = [var for var in t_vars if 'Model_A' in var.name]
	g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

	sess = tf.Session()

	f_saver = tf.train.Saver(f_vars)
	g_saver = tf.train.Saver(g_vars)
	f_saver.restore(sess, tf.train.latest_checkpoint("./weights/target_model/Model_A/"))
	g_saver.restore(sess, tf.train.latest_checkpoint("./weights/generator/"))

	rawpert, pert, fake_l, real_l = sess.run([perturb, x_perturbed, f_fake_probs, f_real_probs], \
												feed_dict={x_pl: dataset.test.data[:32], \
														   is_training: False, \
														   is_training_target: False})
	print('LA: ' + str(np.argmax(dataset.test.labels[:32], axis=1)))
	print('OG: ' + str(np.argmax(real_l, axis=1)))
	print('PB: ' + str(np.argmax(fake_l, axis=1)))

	correct_prediction = tf.equal(tf.argmax(f_fake_probs, 1), tf.argmax(t, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	accs = []
	perts = []
	total_batches_test = int(dataset.test.num_examples / batch_size)
	for i in range(total_batches_test):
		batch_x, batch_y = dataset.train.next_batch(batch_size, i)

		if is_targeted:
			targets = np.full((batch_y.shape[0],), target)
			batch_y = np.eye(t.shape[-1])[targets]

		acc, fake_l, x_pert, p = sess.run([accuracy, f_fake_probs, x_perturbed, perturb], feed_dict={x_pl: batch_x, \
																						 t: batch_y, \
																						 is_training: False, \
																						 is_training_target: False})
		accs.append(acc)
		perts.append(x_pert)

	np.set_printoptions(precision=4, suppress=True)
	print(batch_x[0])
	print(p[0])
	print(x_pert[0])
	perts = np.vstack(perts)
	np.save('./data/perturbed_' + str(target) + '.npy', perts)

	print('accuracy of test set: {}'.format(sum(accs) / len(accs)))



if __name__ == '__main__':
	#Parse Arguments
	parser = argparse.ArgumentParser(description='Run classification on specified dataset, \
		subset of genes, or a random set')
	parser.add_argument('--dataset', help='dataset to be used', type=str, required=True)
	parser.add_argument('--gene_list', help='list of genes in dataset (same order as dataset)', \
		type=str, required=True)
	parser.add_argument('--class_counts', help='json file containing number of samples per class', \
		type=str, required=True)
	parser.add_argument('--subset_list', help='gmt/gct file containing subsets', type=str, required=False)
	parser.add_argument('--set', help='specific subset to run', type=str, required=False)
	parser.add_argument('--target', help='target class', type=str, required=False)

	args = parser.parse_args()

	# load the data
	print('loading genetic data...')
	gtex_gct_flt = np.load(args.dataset)
	total_gene_list = np.load(args.gene_list)
	data = load_data(args.class_counts, gtex_gct_flt)

	# if subset is passed, filter out the genes that are not in the total gene list
	# and redefine the subsets with valid genes
	if args.subset_list:
		subsets = read_subset_file(args.subset_list)

		tot_genes = []
		missing_genes = []

		print('checking for valid genes...')
		for s in subsets:
			genes = []
			for g in subsets[s]:
				if g not in tot_genes:
					tot_genes.append(g)
				if g in total_gene_list:
					genes.append(g)
				else:
					if g not in missing_genes:
						missing_genes.append(g)
			subsets[s] = genes
					#print('missing gene ' + str(g))
		print('missing ' + str(len(missing_genes)) + '/' + str(len(tot_genes)) + ' genes' + ' or ' \
			 + str(int((float(len(missing_genes)) / len(tot_genes)) * 100.0)) + '% of genes')


	if args.subset_list:
		# dataset using only certain genes
		dataset = DC(data, total_gene_list, subsets[args.set.upper()])
	else:
		# dataset using every gene
		dataset = DC(data, total_gene_list)

	# preprocess data
	scaler = preprocessing.MinMaxScaler() #preprocessing.MaxAbsScaler()
	dataset.train.data = scaler.fit_transform(dataset.train.data)
	dataset.test.data = scaler.fit_transform(dataset.test.data)

	# AdvGAN(dataset, batch_size=32, epochs=20, target=22)
	attack(dataset, target=22)



