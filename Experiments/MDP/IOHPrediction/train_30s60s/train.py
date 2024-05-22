import yaml
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from models import create_functional_model
import tensorflow as tf
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_data(data_path, x_file, sec):
    X_input = np.load(os.path.join(data_path, x_file))
    if sec == '30s':
        X_input = X_input[:, 15000:]
    else:
        X_input = X_input[:, 12000:]
    return X_input

def main():
    #YAML load
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Hypoer parameters
    batch_size = config['hyperparameters']['batch_size']
    epochs = config['hyperparameters']['epochs']
    drop_rate = config['hyperparameters']['drop_rate']
    gauss_std = config['hyperparameters']['gauss_std']
    learning_rate = config['hyperparameters']['learning_rate']
    conv_params = config['hyperparameters']['conv_params']

    #Set data
    db = config['data']['db']
    sec = config['data']['sec']
    data_path = config['data'][db.lower()]['paths']['data_path']
    x_file = config['data'][db.lower()]['paths']['x_file']
    y_file = config['data'][db.lower()]['paths']['y_file']

    X_input = load_data(data_path, x_file, sec)
    Y_input = np.load(os.path.join(data_path, y_file))

    #Save log
    log_path = config['model']['log_path']
    checkpoint_dir = config['model']['checkpoint_dir'].format(sec=sec)
    checkpoint_prefix = config['model']['checkpoint_prefix']

    csv_logger = CSVLogger(os.path.join(log_path, f"{db}_train_{sec}.csv"), append=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{db}_train_{sec}", f"{checkpoint_prefix}_epoch_{{epoch:02d}}_loss_{{loss:.5f}}_auc_{{auc:.5f}}_valloss_{{val_loss:.5f}}_valauc_{{val_auc:.5f}}.hdf5")
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=False, save_best_only=True, save_freq='epoch', verbose=1)

    #Create model and compile
    input_shape = (X_input.shape[1], 1)
    model = create_functional_model(input_shape, drop_rate, gauss_std, conv_params)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])

    #Train Model
    model.fit(X_input, Y_input, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, csv_logger])

if __name__ == '__main__':
    main()