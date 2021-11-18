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



def dense_batchnorm_relu(x, units):
    x = keras.layers.Dense(units=units)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return x



def res_block(x_input, units):
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
        x = dense_batchnorm_relu(x, units=units)

    # define residual blocks
    for units in [128, 128, 128]:
        x = res_block(x, units=units)

    # define decoding layers
    for units in [256, 512]:
        x = dense_batchnorm_relu(x, units=units)

    # define output layer
    p_output = keras.layers.Dense(units=n_inputs, activation='tanh')(x)

    # define model
    return keras.models.Model(x_input, p_output)



def adversarial_loss(labels, preds, kappa=0.0):
    fake = tf.reduce_max((1 - labels) * preds - (labels * 10000), axis=-1)
    real = tf.reduce_sum(labels * preds, axis=-1)
    return tf.reduce_sum(tf.maximum(fake - real, kappa))



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
        losses = [
            'loss_d',
            'loss_adv',
            'loss_gan',
            'loss_norm',
            'loss_td',
            'loss_g'
        ]

        self.trackers = {name: keras.metrics.Mean(name=name) for name in losses}

    @property
    def metrics(self):
        return self.trackers.values()

    def compile(self):
        super(AdvGAN, self).compile()
        self.d_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        self.g_optimizer = keras.optimizers.Adam(learning_rate=0.0002)

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

        d_labels_real = tf.ones((batch_size, 1))
        d_labels_fake = tf.zeros((batch_size, 1))
        d_labels = tf.concat([d_labels_real, d_labels_fake], axis=0)

        # train the discriminator
        with tf.GradientTape() as tape:
            d_preds = self.discriminator(d_inputs, training=True)
            d_loss = keras.losses.mean_squared_error(d_labels, d_preds)

        trainable_weights = self.discriminator.trainable_weights
        grads = tape.gradient(d_loss, trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, trainable_weights))

        # train the generator
        with tf.GradientTape() as tape:
            # compute perturbed samples
            p = self.generator(x, training=True)
            x_fake = tf.clip_by_value(x + p, 0, 1)

            # feed perturbed samples to discriminator
            d_preds = self.discriminator(x_fake, training=False)

            # feed perturbed samples to target model
            f_preds = self.target_model(x_fake, training=False)

            # compute adversarial loss (encourage misclassification)
            l_adv = adversarial_loss(y, f_preds)

            # compute GAN loss (fool the discriminator)
            l_gan = keras.losses.mean_squared_error(d_labels_real, d_preds)

            # compute perturbation loss (encourage minimal perturbation)
            epsilon = 1.0
            l_norm = tf.reduce_mean(tf.norm(p, ord=2) + epsilon)

            # compute target distribution loss (encourage realism of perturbed samples)
            l_td = keras.losses.mean_absolute_error(x_fake, target_normal)

            # define generator loss
            alpha_1 = 1.0
            alpha_2 = 1.0
            alpha_3 = 1.0
            alpha_4 = 1.0
            g_loss = alpha_1*l_adv + alpha_2*l_gan + alpha_3*l_norm + alpha_4*l_td

        trainable_weights = self.generator.trainable_weights
        grads = tape.gradient(g_loss, trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, trainable_weights))

        # update metrics
        metrics = {
            'loss_d': d_loss,
            'loss_adv': l_adv,
            'loss_gan': l_gan,
            'loss_norm': l_norm,
            'loss_td': l_td,
            'loss_g': g_loss
        }

        for name, value in metrics.items():
            self.trackers[name].update_state(value)

        return {m.name: m.result() for m in self.metrics}

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
        self.generator.save('%s/%d_generator.h5' % (self.output_dir, self.target))
        self.discriminator.save('%s/%d_discriminator.h5' % (self.output_dir, self.target))

    def load(self):
        self.generator = keras.models.load_model('%s/%d_generator.h5' % (self.output_dir, self.target))
        self.discriminator = keras.models.load_model('%s/%d_discriminator.h5' % (self.output_dir, self.target))
