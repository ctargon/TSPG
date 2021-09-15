import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

import target_model
import utils



def discriminator(n_inputs, hidden_layer_sizes=[512, 256, 128]):
    # define input layer
    x_input = keras.Input(shape=n_inputs)
    x = x_input

    # define hidden layers
    for units in hidden_layer_sizes:
        x = keras.layers.Dense(units=units)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)

    # define output layer
    y_output = keras.layers.Dense(units=1, activation='sigmoid')(x)

    # define model
    return keras.models.Model(x_input, y_output)



def DenseBatchNormRelu(x, units):

    x = keras.layers.Dense(units=units)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return x



def ResBlock(x_input, units):

    x = x_input
    x = keras.layers.Dense(units=units)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(units=units)(x)
    x = keras.layers.BatchNormalization()(x)

    return x_input + x



def generator(n_inputs):
    # define input layer
    x_input = keras.Input(shape=n_inputs)
    x = x_input

    # define encoding layers
    for units in [512, 256, 128]:
        x = DenseBatchNormRelu(x, units=units)

    # define residual blocks
    for units in [128, 128, 128]:
        x = ResBlock(x, units=units)

    # define decoding layers
    for units in [256, 512]:
        x = DenseBatchNormRelu(x, units=units)

    # define output layer
    p_output = keras.layers.Dense(units=n_inputs, activation='tanh')(x)

    # define model
    return keras.models.Model(x_input, p_output)



def advgan_loss(alpha=1.0, beta=1.0, lmbda=1.0):
    # define helper functions
    def adv_loss(labels, preds):
        real = tf.reduce_sum(labels * preds, 1)
        other = tf.reduce_max((1 - labels) * preds - (labels * 10000), 1)
        return tf.reduce_sum(tf.maximum(0.0, other - real))

    def perturb_loss(preds, epsilon=1e-8):
        preds = tf.reshape(preds, (tf.shape(preds)[0], -1))
        norm = tf.norm(preds, axis=1)
        return tf.reduce_mean(norm + epsilon)

    # define loss function
    def loss_fn(labels, preds):
        # unpack preds
        target_normal, p, x_fake, d_fake, f_fake = preds

        # define adversarial loss (encourage misclassification)
        l_adv = adv_loss(labels, f_fake)

        # define GAN loss (fool the discriminator)
        g_loss_fake = keras.losses.mean_squared_error(tf.ones_like(d_fake), d_fake)

        # define perturbation loss (encourage minimal perturbation)
        l_perturb = perturb_loss(p, epsilon=1.0)

        # define target distribution loss (encourage realism of perturbed sample)
        l_target = keras.losses.mean_absolute_error(x_fake, target_normal)

        # define generator loss
        return l_adv + alpha*g_loss_fake + beta*l_perturb + lmbda*l_target

    return loss_fn



class AdvGAN(keras.Model):

    def __init__(self,
                 n_inputs,
                 n_classes,
                 target=-1,
                 target_mu=None,
                 target_cov=None,
                 preload=False,
                 output_dir='.'):

        super(AdvGAN, self).__init__()

        # save attributes
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.target = target
        self.output_dir = output_dir

        # load pre-trained target model
        self.target_model = target_model.load(output_dir)

        # define target distribution
        if target_mu is not None:
            target_mu = target_mu.astype(np.float32)
            target_cov = target_cov.astype(np.float32)

            if len(target_cov.shape) == 2:
                self.target_dist = tfp.distributions.MultivariateNormalTriL(
                    loc=target_mu,
                    scale_tril=tf.linalg.cholesky(target_cov))

            elif len(target_cov.shape) == 1:
                self.target_dist = tfp.distributions.MultivariateNormalDiag(
                    loc=target_mu,
                    scale_diag=target_cov)

        # load models if specified
        if preload:
            self.load()

        # otherwise initialize models
        else:
            self.generator = generator(n_inputs)
            self.discriminator = discriminator(n_inputs)

        # define loss trackers
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")

    @property
    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker]

    def compile(self):
        super(AdvGAN, self).compile()
        self.g_optimizer = keras.optimizers.Adam(learning_rate=0.0002)
        self.d_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        self.g_loss_fn = advgan_loss()
        self.d_loss_fn = keras.losses.MeanSquaredError()

    def train_step(self, data):
        # unpack arguments
        x, y = data

        # determine batch size
        batch_size = tf.shape(x)[0]

        # generate samples from target distribution
        target_normal = self.target_dist.sample(batch_size)
        target_normal = tf.clip_by_value(target_normal, 0, 1)

        # change labels to target class
        indices = tf.fill((tf.shape(y)[0], 1), self.target)
        y = tf.one_hot(indices, depth=tf.shape(y)[-1], axis=-1)

        # generated perturbed samples
        p = self.generator(x, training=False)
        x_fake = tf.clip_by_value(x + p, 0, 1)

        # prepare training data for discriminator
        d_inputs = tf.concat([x, x_fake], axis=0)
        d_labels = tf.concat([
            tf.ones((batch_size, 1)),
            tf.zeros((batch_size, 1))
        ], axis=0)

        # train the discriminator
        with tf.GradientTape() as tape:
            d_preds = self.discriminator(d_inputs)
            d_loss = self.d_loss_fn(d_labels, d_preds)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # train the generator
        with tf.GradientTape() as tape:
            # compute perturbed samples
            p = self.generator(x)
            x_fake = tf.clip_by_value(x + p, 0, 1)

            # feed perturbed samples to discriminator
            d_fake = self.discriminator(x_fake, training=False)

            # feed perturbed samples to target model
            f_fake = self.target_model(x_fake, training=False)

            # compute loss
            labels = tf.zeros((batch_size, 1))
            preds = target_normal, p, x_fake, d_fake, f_fake

            g_loss = self.g_loss_fn(labels, preds)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )

        # monitor loss
        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)

        return {
            'g_loss': self.g_loss_tracker.result(),
            'd_loss': self.d_loss_tracker.result()
        }

    def perturb(self, x):
        # compute perturbations, perturbed samples
        p = self.generator(x, training=False)
        x_fake = np.clip(x + p, 0, 1)

        # apply clip to perturbations
        p = x_fake - x

        return x_fake, p

    def predict_target(self, x):
        return self.target_model(x, training=False)

    def score(self, x, y):
        # compute perturbed samples
        x_fake, p = self.perturb(x)

        # feed perturbed samples to target model
        y_fake = self.predict_target(x_fake)

        # compute perturbation accuracy
        score = np.mean(np.argmax(y_fake, axis=1) == self.target)

        return score, x_fake, p, y_fake

    def save(self):
        # initialize output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # save generator, discriminator
        self.generator.save('%s/generator_%d.h5' % (self.output_dir, self.target))
        self.discriminator.save('%s/discriminator_%d.h5' % (self.output_dir, self.target))

    def load(self):
        self.generator = keras.models.load_model('%s/generator_%d.h5' % (self.output_dir, self.target))
        self.discriminator = keras.models.load_model('%s/discriminator_%d.h5' % (self.output_dir, self.target))
