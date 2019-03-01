'''
	Target model definitions that adverserial examples are attempting to 'fool'

	ref: https://arxiv.org/pdf/1801.02610.pdf
'''

import tensorflow as tf
import numpy as np
from sklearn import preprocessing

import sys, os, argparse

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

from utils.dataset import DataContainer as DC
from utils.utils import load_data, read_subset_file

class Target:
	def __init__(self, lr=0.001, epochs=50, n_input=28, n_classes=10, batch_size=256,\
					restore=0):
		self.lr = lr
		self.epochs = epochs
		self.n_input = n_input
		self.n_classes = n_classes
		self.batch_size = batch_size
		self.restore = restore

	def train_model(self, dataset, model_name):
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

		for epoch in range(1, self.epochs + 1):
			avg_cost = 0.
			total_batch = int(dataset.train.num_examples/self.batch_size)

			for i in range(total_batch):
				batch_x, batch_y = dataset.train.next_batch(self.batch_size, i)
				
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, training: True})

				avg_cost += c / total_batch

			print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost))

		# Test model
		correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

		# Calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		accs = []

		total_test_batch = int(dataset.test.num_examples / self.batch_size)
		for i in range(total_test_batch):
			batch_x, batch_y = dataset.test.next_batch(self.batch_size, i)
			accs.append(accuracy.eval({x: batch_x, y: batch_y, training: False}, session=sess))

		print('accuracy of test set: {}'.format(sum(accs) / len(accs)))

		path = "./weights/target_model/" + str(model_name)
		if (not os.path.isdir(path)):
			os.makedirs(path)

		saver.save(sess, path + "/" + str(model_name) + ".ckpt")
		sess.close() 


	def inference_model(self, dataset, model_name):
		tf.reset_default_graph()

		x = tf.placeholder(tf.float32, [None, self.n_input])
		y = tf.placeholder(tf.float32, [None, self.n_classes])
		training = tf.placeholder(tf.bool, shape=())
		
		logits, probs = self.Model(x, training)

		saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_name))
		sess = tf.Session()
		saver.restore(sess, "./weights/target_model/" + str(model_name) + "/" + str(model_name) + ".ckpt")

		# Test model
		correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

		# Calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		accs = []

		total_test_batch = int(dataset.test.num_examples / self.batch_size)
		for i in range(total_test_batch):
			batch_x, batch_y = dataset.test.next_batch(self.batch_size, i)
			acc, prob = sess.run([accuracy, probs], {x: batch_x, y: batch_y, training: False})
			accs.append(acc)

		print('accuracy of test set: {}'.format(sum(accs) / len(accs)))
		sess.close()



class Target_A(Target):
	def __init__(self, lr=0.001, epochs=50, n_input=28, n_classes=10, batch_size=256,\
					restore=0):
		Target.__init__(self, lr, epochs, n_input, n_classes, batch_size, restore)
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


	# USAGE:
	# 		- encoder network for vae
	# PARAMS:
	#	x: input data sample
	#	h_hidden: LIST of num. neurons per hidden layer
	def Model(self, x, training):
		with tf.variable_scope('Model_A', reuse=tf.AUTO_REUSE):
			fc1 = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
			fc2 = tf.layers.dense(inputs=fc1, units=512, activation=tf.nn.relu)
			fc3 = tf.layers.dense(inputs=fc2, units=128, activation=tf.nn.relu)

			logits = tf.layers.dense(inputs=fc3, units=self.n_classes, activation=None)

			probs = tf.nn.softmax(logits)

			return logits, probs

	def train(self, dataset):
		return self.train_model(dataset, "Model_A")

	def inference(self, dataset):
		return self.inference_model(dataset, "Model_A")



class Target_B(Target):
	def __init__(self, lr=0.001, epochs=50, n_input=28, n_classes=10, batch_size=256,\
					restore=0):
		Target.__init__(self, lr, epochs, n_input, n_classes, batch_size, restore)
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


	# USAGE:
	# 		- encoder network for vae
	# PARAMS:
	#	x: input data sample
	#	h_hidden: LIST of num. neurons per hidden layer
	def Model(self, x, training):
		with tf.variable_scope('Model_B', reuse=tf.AUTO_REUSE):
			fc1 = tf.layers.dense(inputs=x, units=1024, activation=None)
			fc1_bn = tf.nn.relu(tf.layers.batch_normalization(fc1, training=training))

			fc2 = tf.layers.dense(inputs=fc1_bn, units=512, activation=None)
			fc2_bn = tf.nn.relu(tf.layers.batch_normalization(fc2, training=training))
			
			fc3 = tf.layers.dense(inputs=fc2_bn, units=128, activation=None)
			fc3_bn = tf.nn.relu(tf.layers.batch_normalization(fc3, training=training))

			logits = tf.layers.dense(inputs=fc3_bn, units=self.n_classes, activation=None)

			probs = tf.nn.softmax(logits)

			return logits, probs

	def train(self, dataset):
		return self.train_model(dataset, "Model_B")

	def inference(self, dataset):
		return self.inference_model(dataset, "Model_B")


class Target_C(Target):
	def __init__(self, lr=0.001, epochs=50, n_input=28, n_classes=10, batch_size=256,\
					restore=0):
		Target.__init__(self, lr, epochs, n_input, n_classes, batch_size, restore)
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


	# USAGE:
	# 		- encoder network for vae
	# PARAMS:
	#	x: input data sample
	#	h_hidden: LIST of num. neurons per hidden layer
	def Model(self, x, training):
		with tf.variable_scope('Model_C', reuse=tf.AUTO_REUSE):
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

	def train(self, dataset):
		return self.train_model(dataset, "Model_C")

	def inference(self, dataset):
		return self.inference_model(dataset, "Model_C")


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

	mlp = Target_C(n_input=dataset.train.data.shape[1], n_classes=len(data), batch_size=128, epochs=30)
	mlp.train(dataset)
	mlp.inference(dataset)


