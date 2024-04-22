import tensorflow as tf
from tensorflow.keras import layers, models

def create_functional_model(input_shape, drop_rate, gauss_std, conv_params):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for filters, kernel_size in conv_params:
        x = layers.GaussianNoise(gauss_std)(x)
        x = layers.Conv1D(filters, kernel_size, activation='tanh')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(drop_rate)(x, training=True)
    x = layers.Flatten()(x)
    x = layers.Dense(2, activation='tanh')(x)
    x = layers.Dropout(drop_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model