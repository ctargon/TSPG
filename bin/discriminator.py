'''
    Discriminator definition for AdvGAN

    ref: https://arxiv.org/pdf/1801.02610.pdf
'''
import tensorflow as tf



def discriminator(x, training):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        dense1 = tf.layers.dense(inputs=x, units=512, activation=None)
        bn_1 = tf.layers.batch_normalization(dense1, training=training)
        # in_1 = tf.contrib.layers.instance_norm(dense1)
        lr_1 = tf.nn.leaky_relu(bn_1, alpha=0.2)

        dense2 = tf.layers.dense(inputs=lr_1, units=256, activation=None)
        bn_2 = tf.layers.batch_normalization(dense2, training=training)
        # in_2 = tf.contrib.layers.instance_norm(dense2)
        lr_2 = tf.nn.leaky_relu(bn_2, alpha=0.2)

        dense3 = tf.layers.dense(inputs=lr_2, units=128, activation=None)
        bn_3 = tf.layers.batch_normalization(dense3, training=training)
        # in_3= tf.contrib.layers.instance_norm(dense3)
        lr_3 = tf.nn.leaky_relu(bn_3, alpha=0.2)

        logits = tf.layers.dense(lr_3, 1)

        probs = tf.nn.sigmoid(logits)

        return logits, probs
