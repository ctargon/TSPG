"""
	Target model definitions that adverserial examples are attempting to "fool"

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



class Target:
	def __init__(self, lr=0.001, epochs=50, n_input=28, n_classes=10, batch_size=256,\
					output_dir="."):
		self.lr = lr
		self.epochs = epochs
		self.n_input = n_input
		self.n_classes = n_classes
		self.batch_size = batch_size
		self.output_dir = output_dir

		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

	def train_model(self, x_train, y_train, model_name):
		tf.reset_default_graph()

		# define placeholders for input data
		x = tf.placeholder(tf.float32, [None, self.n_input])
		y = tf.placeholder(tf.float32, [None, self.n_classes])
		training = tf.placeholder(tf.bool, [])

		# define compute graph
		logits, _ = self.Model(x, training)

		# define cost
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

		# optimizer
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

		saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_name))

		# Initializing the variables
		init = tf.global_variables_initializer()

		sess = tf.Session()
		sess.run(init)

		for epoch in range(self.epochs):
			avg_cost = 0.
			n_batches = int(len(x_train) / self.batch_size)

			for i in range(n_batches):
				batch_x, batch_y = utils.next_batch(x_train, y_train, self.batch_size, i)
				
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, training: True})

				avg_cost += c / n_batches

			print("Epoch: %04d, cost=%f" % (epoch + 1, avg_cost))

		path = "%s/target_model" % (self.output_dir)
		os.makedirs(path, exist_ok=True)

		saver.save(sess, "%s/%s.ckpt" % (path, model_name))
		sess.close() 

	def inference_model(self, x_test, y_test, model_name):
		tf.reset_default_graph()

		x = tf.placeholder(tf.float32, [None, self.n_input])
		y = tf.placeholder(tf.float32, [None, self.n_classes])
		training = tf.placeholder(tf.bool, shape=())
		
		logits, probs = self.Model(x, training)

		saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_name))
		sess = tf.Session()
		saver.restore(sess, "%s/target_model/%s.ckpt" % (self.output_dir, model_name))

		# Test model
		correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

		# Calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		scores = []

		n_batches = int(len(x_test) / self.batch_size)
		for i in range(n_batches):
			batch_x, batch_y = utils.next_batch(x_test, y_test, self.batch_size, i)
			score, prob = sess.run([accuracy, probs], {x: batch_x, y: batch_y, training: False})
			scores.append(score)

		print("accuracy of test set: %0.3f" % (sum(scores) / len(scores)))
		sess.close()



class Target_A(Target):
	def __init__(self, lr=0.001, epochs=50, n_input=28, n_classes=10, batch_size=256, \
					output_dir="."):
		Target.__init__(self, lr, epochs, n_input, n_classes, batch_size, output_dir)

	# USAGE:
	# 		- encoder network for vae
	# PARAMS:
	#	x: input data sample
	#	h_hidden: LIST of num. neurons per hidden layer
	def Model(self, x, training):
		with tf.variable_scope("Model_A", reuse=tf.AUTO_REUSE):
			fc1 = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
			fc2 = tf.layers.dense(inputs=fc1, units=512, activation=tf.nn.relu)
			fc3 = tf.layers.dense(inputs=fc2, units=128, activation=tf.nn.relu)

			logits = tf.layers.dense(inputs=fc3, units=self.n_classes, activation=None)

			probs = tf.nn.softmax(logits)

			return logits, probs

	def train(self, x, y):
		return self.train_model(x, y, "Model_A")

	def inference(self, x, y):
		return self.inference_model(x, y, "Model_A")



class Target_B(Target):
	def __init__(self, lr=0.001, epochs=50, n_input=28, n_classes=10, batch_size=256, \
					output_dir="."):
		Target.__init__(self, lr, epochs, n_input, n_classes, batch_size, output_dir)

	# USAGE:
	# 		- encoder network for vae
	# PARAMS:
	#	x: input data sample
	#	h_hidden: LIST of num. neurons per hidden layer
	def Model(self, x, training):
		with tf.variable_scope("Model_B", reuse=tf.AUTO_REUSE):
			fc1 = tf.layers.dense(inputs=x, units=1024, activation=None)
			fc1_bn = tf.nn.relu(tf.layers.batch_normalization(fc1, training=training))

			fc2 = tf.layers.dense(inputs=fc1_bn, units=512, activation=None)
			fc2_bn = tf.nn.relu(tf.layers.batch_normalization(fc2, training=training))
			
			fc3 = tf.layers.dense(inputs=fc2_bn, units=128, activation=None)
			fc3_bn = tf.nn.relu(tf.layers.batch_normalization(fc3, training=training))

			logits = tf.layers.dense(inputs=fc3_bn, units=self.n_classes, activation=None)

			probs = tf.nn.softmax(logits)

			return logits, probs

	def train(self, x, y):
		return self.train_model(x, y, "Model_B")

	def inference(self, x, y):
		return self.inference_model(x, y, "Model_B")



class Target_C(Target):
	def __init__(self, lr=0.001, epochs=50, n_input=28, n_classes=10, batch_size=256, \
					output_dir="."):
		Target.__init__(self, lr, epochs, n_input, n_classes, batch_size, output_dir)

	# USAGE:
	# 		- encoder network for vae
	# PARAMS:
	#	x: input data sample
	#	h_hidden: LIST of num. neurons per hidden layer
	def Model(self, x, training):
		with tf.variable_scope("Model_C", reuse=tf.AUTO_REUSE):
			fc1 = tf.layers.dense(inputs=x, units=1024, activation=None)
			fc1_bn = tf.nn.relu(tf.layers.batch_normalization(fc1, training=training))

			fc2 = tf.layers.dense(inputs=fc1_bn, units=1024, activation=None)
			fc2_bn = tf.nn.relu(tf.layers.batch_normalization(fc2, training=training))

			fc3 = tf.layers.dense(inputs=fc2_bn, units=512, activation=None)
			fc3_bn = tf.nn.relu(tf.layers.batch_normalization(fc2, training=training))
			
			fc4 = tf.layers.dense(inputs=fc3_bn, units=512, activation=None)
			fc4_bn = tf.nn.relu(tf.layers.batch_normalization(fc2, training=training))

			fc5 = tf.layers.dense(inputs=fc4_bn, units=128, activation=None)
			fc5_bn = tf.nn.relu(tf.layers.batch_normalization(fc3, training=training))

			logits = tf.layers.dense(inputs=fc5_bn, units=self.n_classes, activation=None)

			probs = tf.nn.softmax(logits)

			return logits, probs

	def train(self, x, y):
		return self.train_model(x, y, "Model_C")

	def inference(self, x, y):
		return self.inference_model(x, y, "Model_C")



if __name__ == "__main__":
	# parse command-line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", help="input dataset (samples x genes)", required=True)
	parser.add_argument("--labels", help="list of sample labels", required=True)
	parser.add_argument("--gene-sets", help="list of curated gene sets")
	parser.add_argument("--output-dir", help="Output directory", default=".")

	args = parser.parse_args()

	# load input data
	print("loading input dataset...")

	df = utils.load_dataframe(args.dataset)
	df_samples = df.index
	df_genes = df.columns

	labels, classes = utils.load_labels(args.labels)

	print("loaded input dataset (%s genes, %s samples)" % (df.shape[1], df.shape[0]))

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
		# initialize output directory
		output_dir = "%s/%s" % (args.output_dir, name)

		# extract dataset
		X = df[genes]
		y = utils.onehot_encode(labels, classes)

		# create train/test sets
		x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)

		# normalize dataset
		Scaler = sklearn.preprocessing.MinMaxScaler
		x_train = Scaler().fit_transform(x_train)
		x_test = Scaler().fit_transform(x_test)

		clf = Target_A(n_input=x_train.shape[1], n_classes=len(classes), epochs=30, batch_size=128, output_dir=output_dir)
		clf.train(x_train, y_train)
		clf.inference(x_test, y_test)
