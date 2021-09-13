import logging
import numpy as np
import os
import sklearn.metrics
import tensorflow as tf



tf.get_logger().setLevel(logging.ERROR)



class TargetModel:

    def __init__(self,
                 n_inputs,
                 n_classes,
                 hidden_layer_sizes=[512, 256, 128],
                 l1=0,
                 l2=0,
                 dropout=0.0,
                 batch_norm=False,
                 lr=0.001,
                 epochs=50,
                 batch_size=32,
                 output_dir='.'):

        # save attributes
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = output_dir

        # initialize model
        x_input = tf.keras.Input(shape=n_inputs)

        x = x_input
        for units in hidden_layer_sizes:
            x = tf.keras.layers.Dense(
                units=units,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2),
                bias_regularizer=tf.keras.regularizers.l1_l2(l1, l2)
            )(x)

            if batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.ReLU()(x)

            if dropout:
                x = tf.keras.layers.Dropout(dropout)(x)

        y_output = tf.keras.layers.Dense(units=n_classes, activation='softmax')(x)

        self.model = tf.keras.models.Model(x_input, y_output)

        # define optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # compile model
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, x_train, y_train):

        # train model
        history = self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.1
        )

        # return training history
        return history

    def predict(self, x_test):
        return self.model.predict(x_test)

    def score(self, x_test, y_test):
        # compute predictions
        y_pred = self.predict(x_test)

        # convert ground truth and predictions from one-hot to indices
        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        # compute accuracy
        score = sklearn.metrics.accuracy_score(y_test, y_pred)

        return score

    def save(self):
        # initialize output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # save model
        self.model.save('%s/target_model' % (self.output_dir))

    def load(self):
        self.model = tf.keras.models.load_model('%s/target_model' % (self.output_dir))
