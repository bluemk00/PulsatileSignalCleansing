import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from model import SelfAttentionuNet_1D, custom_loss

def load_data(config):
    Train_X = np.load(config['paths']['train_x'])
    Train_Y = np.load(config['paths']['train_y'])
    Test_X = np.load(config['paths']['test_x'])
    Test_Y = np.load(config['paths']['test_y'])
    return Train_X, Train_Y, Test_X, Test_Y
 
def train(config):
    Train_X, Train_Y, Test_X, Test_Y = load_data(config)
    input_shape = (1200, 1)
    model = SelfAttentionuNet_1D(input_shape, dropout=config['model']['dropout'], batchnorm=config['model']['batchnorm'])
    model.compile(optimizer='adam', loss=custom_loss, metrics=['mse', 'mae'])

    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=config['paths']['checkpoint'],
        save_weights_only=False,
        save_best_only=True,
        save_freq='epoch', 
        verbose=1
    )

    csv_logger = callbacks.CSVLogger(config['paths']['log'], append=True, separator=',')

    model.fit(Train_X, Train_Y,
              epochs=config['train']['total_epochs'],
              validation_data=(Test_X, Test_Y),
              batch_size=config['train']['batch_size'],
              shuffle=config['train']['shuffle'],
              callbacks=[model_checkpoint_callback, csv_logger])

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    os.environ["CUDA_DEVICE_ORDER"] = config['gpu']['cuda_device_order']
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']['cuda_visible_devices']

    gpu_config = tf.compat.v1.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.gpu_options.per_process_gpu_memory_fraction = config['gpu']['per_process_gpu_memory_fraction']
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=gpu_config))

    train(config)