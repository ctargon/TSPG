import tensorflow as tf
from sklearn import preprocessing
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

from models.generator import generator
from models.target_models import Target_A as target_model
from utils.utils import parse_and_load_data

def cleanse_label(label):
	label = label.replace(" ", "_")
	label = label.replace("-", "")
	label = label.replace("(", "")
	label = label.replace(")", "")
	return label


def targeted_specific_attack(dataset, start, target, mu_T):
	s_idxs = np.where(start == np.argmax(dataset.train.labels, axis=1))
	data = dataset.train.data[s_idxs]
	labels = dataset.train.labels[s_idxs]

	x_pl = tf.placeholder(tf.float32, [None, data.shape[-1]]) # sample placeholder
	t = tf.placeholder(tf.float32, [None, labels.shape[-1]]) # target placeholder
	is_training = tf.placeholder(tf.bool, [])
	is_training_target = tf.placeholder(tf.bool, [])

	is_targeted = False
	if target in range(0, labels.shape[-1]):
		print("target is: " + str(dataset.label_names_ordered[target]))
		is_targeted = True

	# generate pertubation, add to original, clip to valid expression level
	perturb, logit_perturb = generator(x_pl, is_training)
	x_perturbed = perturb + x_pl
	x_perturbed = tf.clip_by_value(x_perturbed, 0, 1)

	# isntantiate target model, create graphs for original and perturbed data
	f = target_model(n_input=dataset.train.data.shape[-1], n_classes=dataset.train.labels.shape[-1])
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

	start_tissue = str(dataset.label_names_ordered[start])
	target_tissue = str(dataset.label_names_ordered[target])

	if is_targeted:
		targets = np.full((labels.shape[0],), target)
		batch_y = np.eye(t.shape[-1])[targets]

	x_pert, p = sess.run([x_perturbed, perturb], feed_dict={x_pl: data, \
															t: batch_y, \
															is_training: False, \
															is_training_target: False})

	print("original sample is: " + str(dataset.label_names_ordered[start]))
	print(data[0])
	print("perturbation:")
	print(p[0])
	print("X_adv:")
	print(x_pert[0])
	print("mu_target:")
	print(mu_T)

	# save the results in X, P, X_adv, mu_T order
	results = np.vstack([data[0], p[0], x_pert[0], mu_T])

	start_tissue = cleanse_label(start_tissue)
	target_tissue = cleanse_label(target_tissue)

	np.save("./data/heatmap/" + start_tissue + "_to_" + target_tissue + ".npy", results)



def attack(dataset, batch_size=64, thresh=0.3, target=-1):
	x_pl = tf.placeholder(tf.float32, [None, dataset.test.data.shape[-1]]) # sample placeholder
	t = tf.placeholder(tf.float32, [None, dataset.test.labels.shape[-1]]) # target placeholder
	is_training = tf.placeholder(tf.bool, [])
	is_training_target = tf.placeholder(tf.bool, [])

	is_targeted = False
	if target in range(0, dataset.test.labels.shape[-1]):
		print("target is: " + str(dataset.label_names_ordered[target]))
		is_targeted = True

	# generate pertubation, add to original, clip to valid expression level
	perturb, logit_perturb = generator(x_pl, is_training)
	x_perturbed = perturb + x_pl
	x_perturbed = tf.clip_by_value(x_perturbed, 0, 1)

	# isntantiate target model, create graphs for original and perturbed data
	f = target_model(n_input=dataset.train.data.shape[-1], n_classes=dataset.train.labels.shape[-1])
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
	correct_prediction = tf.equal(tf.argmax(f_fake_probs, 1), tf.argmax(t, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	accs = []
	perts = []
	total_batches_test = int(dataset.test.num_examples / batch_size)
	for i in range(total_batches_test):
		batch_x, batch_y_og = dataset.train.next_batch(batch_size, i)

		if is_targeted:
			targets = np.full((batch_y_og.shape[0],), target)
			batch_y = np.eye(t.shape[-1])[targets]

		acc, fake_l, x_pert, p = sess.run([accuracy, f_fake_probs, x_perturbed, perturb], feed_dict={x_pl: batch_x, \
																						 t: batch_y, \
																						 is_training: False, \
																						 is_training_target: False})
		accs.append(acc)
		perts.append(x_pert)

	# print a sample original, perturbation, and original + perturbation
	np.set_printoptions(precision=4, suppress=True)
	print(str(np.argmax(batch_y_og[0])))
	print("original sample is: " + str(dataset.label_names_ordered[np.argmax(batch_y_og, axis=1)[0]]))
	print(batch_x[0])
	print(p[0])
	print(x_pert[0])
	perts = np.vstack(perts)
	np.save("./data/perturbed/perturbed_" + str(target) + ".npy", perts)

	print("accuracy of test set: {}".format(sum(accs) / len(accs)))



if __name__ == "__main__":
	dataset, target = parse_and_load_data()

	# preprocess data
	scaler = preprocessing.MinMaxScaler() #preprocessing.MaxAbsScaler()
	dataset.train.data = scaler.fit_transform(dataset.train.data)
	dataset.test.data = scaler.fit_transform(dataset.test.data)

	# get mu and sigma of target class feature vectors
	t_idxs = np.where(target == np.argmax(dataset.train.labels, axis=1))
	target_data = dataset.train.data[t_idxs]
	target_mu = np.mean(target_data, axis=0)
	target_cov = np.cov(target_data, rowvar=False)

	attack(dataset, target=target)
	
	# for i in range(dataset.num_classes):
	# 	targeted_specific_attack(dataset, i, target, target_mu)
