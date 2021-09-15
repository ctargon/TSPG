import numpy as np
import os
from tensorflow import keras



def create(
    n_inputs,
    n_classes,
    hidden_layer_sizes=[512, 256, 128],
    batch_norm=False):

    # define input layer
    x_input = keras.Input(shape=n_inputs)

    # define hidden layers
    x = x_input
    for units in hidden_layer_sizes:
        x = keras.layers.Dense(units=units)(x)
        if batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

    # define output layer
    y_output = keras.layers.Dense(units=n_classes, activation='softmax')(x)

    # define model
    model = keras.models.Model(x_input, y_output)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model



def save(model, output_dir):
    # initialize output directory
    os.makedirs(output_dir, exist_ok=True)

    # save model
    model.save('%s/target_model.h5' % (output_dir))



def load(output_dir):
    return keras.models.load_model('%s/target_model.h5' % (output_dir))
