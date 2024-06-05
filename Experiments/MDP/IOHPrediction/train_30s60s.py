import tensorflow as tf
from tensorflow.keras import layers, models
import yaml
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

def create_functional_model(input_shape, drop_rate, gauss_std, conv_params):
    """Create a functional model using Keras Functional API."""
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

def load_data(data_path, x_file, sec):
    """Load data from given path and file name, and trim based on 'sec'."""
    X_input = np.load(os.path.join(data_path, x_file))
    if sec == '30s':
        X_input = X_input[:, 15000:]
    else:
        X_input = X_input[:, 12000:]
    return X_input

def print_training_overview(config, X_input, Y_input):
    """Print an overview of the training configuration and data."""
    print()
    print(f"{config['sec']}-IOH Model Training Configuration:")
    print(f"  Batch size: {config['hyperparameters']['batch_size']}")
    print(f"  Epochs: {config['hyperparameters']['epochs']}")
    print(f"  Dropout rate: {config['hyperparameters']['drop_rate']}")
    print(f"  Gaussian noise std: {config['hyperparameters']['gauss_std']}")
    print(f"  Learning rate: {config['hyperparameters']['learning_rate']}")
    print(f"  Convolution parameters: {config['hyperparameters']['conv_params']}")
    print("\nData Overview:")
    print(f"  Input shape: {X_input.shape}")
    print(f"  Output shape: {Y_input.shape}")
    print(f"  Data type: {X_input.dtype}")
    print()

def main():
    # Load configuration from YAML file
    with open('train_config_30s60s.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set GPU configuration
    os.environ["CUDA_DEVICE_ORDER"] = config['gpu']['cuda_device_order']
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']['cuda_visible_devices']

    gpu_config = tf.compat.v1.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.gpu_options.per_process_gpu_memory_fraction = config['gpu']['per_process_gpu_memory_fraction']
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=gpu_config))

    # Hyperparameters
    batch_size = config['hyperparameters']['batch_size']
    epochs = config['hyperparameters']['epochs']
    drop_rate = config['hyperparameters']['drop_rate']
    gauss_std = config['hyperparameters']['gauss_std']
    learning_rate = config['hyperparameters']['learning_rate']
    conv_params = config['hyperparameters']['conv_params']

    # Data configuration
    db = config['data']['db']
    sec = config['sec']
    data_path = config['data'][db.lower()]['paths']['data_path']
    x_file = config['data'][db.lower()]['paths']['x_file']
    y_file = config['data'][db.lower()]['paths']['y_file']

    # Load input data
    X_input = load_data(data_path, x_file, sec)
    Y_input = np.load(os.path.join(data_path, y_file))

    # Print training overview
    print_training_overview(config, X_input, Y_input)

    # Log and checkpoint configuration
    log_path = config['model']['log_path']
    os.makedirs(log_path, exist_ok=True)
    checkpoint_dir = config['model']['checkpoint_dir'].format(sec=sec)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = config['model']['checkpoint_prefix']

    csv_logger = CSVLogger(f"{log_path}train_{sec}IOHmodel.csv", append=True)
    checkpoint_path = f"{checkpoint_dir}{checkpoint_prefix}_epoch_{{epoch:02d}}_loss_{{loss:.5f}}_auc_{{auc:.5f}}_valloss_{{val_loss:.5f}}_valauc_{{val_auc:.5f}}.hdf5"
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=False, save_best_only=True, save_freq='epoch', verbose=1)

    # Create and compile the model
    input_shape = (X_input.shape[1], 1)
    model = create_functional_model(input_shape, drop_rate, gauss_std, conv_params)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])

    # Train the model
    model.fit(X_input, Y_input, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, csv_logger])

if __name__ == "__main__":
    main()
