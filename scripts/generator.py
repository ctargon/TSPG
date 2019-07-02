"""
	Generator definition for AdvGAN

	ref: https://arxiv.org/pdf/1801.02610.pdf
"""
import tensorflow as tf



# helper function for convolution -> instance norm -> relu
def DenseBatchNormRelu(x, units, training):
	dense = tf.layers.dense(inputs=x, units=units, activation=None)

	#InstNorm = tf.contrib.layers.instance_norm(dense)
	bn = tf.layers.batch_normalization(dense, training=training)

	return tf.nn.relu(bn)



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
	with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
		# define first three dense + inst + relu layers
		d1 = DenseBatchNormRelu(x, units=512, training=training)
		d2 = DenseBatchNormRelu(d1, units=256, training=training)
		d3 = DenseBatchNormRelu(d2, units=128, training=training)

		# define residual blocks
		rb1 = ResBlock(d3, training, units=128)
		rb2 = ResBlock(rb1, training, units=128)
		rb3 = ResBlock(rb2, training, units=128)
		#rb4 = ResBlock(rb3, training, filters=32)

		# upsample using conv transpose
		u1 = DenseBatchNormRelu(rb3, units=256, training=training)
		u2 = DenseBatchNormRelu(u1, units=512, training=training)

		# final layer block
		out = tf.layers.dense(u2, units=x.get_shape()[-1], activation=None)

		return tf.nn.tanh(out), out



def generator_unet(x, training):
	with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
		# define first three dense + inst + relu layers
		d1 = DenseBatchNormRelu(x, units=512, training=training)
		d2 = DenseBatchNormRelu(d1, units=256, training=training)
		d3 = DenseBatchNormRelu(d2, units=128, training=training)

		# define residual blocks
		rb1 = ResBlock(d3, training, units=128)
		rb2 = ResBlock(rb1, training, units=128)
		rb3 = ResBlock(rb2, training, units=128)
		#rb4 = ResBlock(rb3, training, filters=32)
		skip1 = d3 + rb3

		# upsample using conv transpose
		u1 = DenseBatchNormRelu(skip1, units=256, training=training)
		skip2 = d2 + u1
		u2 = DenseBatchNormRelu(skip2, units=512, training=training)
		skip3 = d1 + u2

		# final layer block
		out = tf.layers.dense(skip3, units=x.get_shape()[-1], activation=None)

		return tf.nn.tanh(out), out
