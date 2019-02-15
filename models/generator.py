'''
	Generator definition for AdvGAN

	ref: https://arxiv.org/pdf/1801.02610.pdf
'''

import tensorflow as tf

# helper function for convolution -> instance norm -> relu
def DenseInstNormRelu(x, units):
	dense = tf.layers.dense(inputs=x, units=units, activation=None)

	InstNorm = tf.contrib.layers.instance_norm(dense)

	return tf.nn.relu(InstNorm)

# helper function for residual block of 2 convolutions with same num filters
# in the same style as ConvInstNormRelu
def ResBlock(x, training, units):
	dense1 = tf.layers.dense(inputs=x, units=units, activation=None)
	bn_1 = tf.layers.batch_normalization(dense1, training=training)
	r_1 = tf.nn.relu(bn_1)

	dense2 = tf.layers.dense(inputs=r_1, units=units, activation=None)
	bn_2 = tf.layers.batch_normalization(dense2, training=training)

	return x + bn_2


def generator(x, training):
	with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
		# define first three dense + inst + relu layers
		d1 = DenseInstNormRelu(x, units=512)
		d2 = DenseInstNormRelu(d1, units=256)
		d3 = DenseInstNormRelu(d2, units=128)

		# define residual blocks
		rb1 = ResBlock(d3, training, units=128)
		rb2 = ResBlock(rb1, training, units=128)
		rb3 = ResBlock(rb2, training, units=128)
		#rb4 = ResBlock(rb3, training, filters=32)

		# upsample using conv transpose
		u1 = DenseInstNormRelu(rb3, units=256)
		u2 = DenseInstNormRelu(u1, units=512)

		# final layer block
		out = tf.layers.dense(u2, units=x.get_shape()[-1], activation=None)

		# out = tf.contrib.layers.instance_norm(out)

		return tf.nn.tanh(out)




