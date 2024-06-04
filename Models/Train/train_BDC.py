import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

# Ensure yaml is installed
try:
    import yaml
except ImportError:
    print("yaml is not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml

# Append custom library paths
sys.path.append("../utils/")
from BDC_utils import *

# Load configuration from YAML file
with open('config_BDC.yaml', 'r') as file:
    config = yaml.safe_load(file)


if __name__ == "__main__":

    # Set up GPU environment
    os.environ["CUDA_DEVICE_ORDER"] = config['gpu']['cuda_device_order']
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']['cuda_visible_devices']

    gpu_config = tf.compat.v1.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.gpu_options.per_process_gpu_memory_fraction = config['gpu']['per_process_gpu_memory_fraction']
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=gpu_config))
    tf.compat.v1.enable_eager_execution()

    signal_type = config['model']['signal_type']
    batch_size = config['hyperparameters']['batchsize']
    epochs = config['hyperparameters']['epochs']

    # Output directory for models
    if signal_type == 'ABP':
        outdir = config['output']['directory']
    elif signal_type == 'PPG':
        outdir = config['output']['directory'].replace('ABP', 'PPG')

    os.makedirs(outdir, exist_ok=True)
    checkpoint_prefix = os.path.join(outdir, "ckpt")

    ################
    # Loading Data #
    ################

    # Load and normalize data based on model type
    if signal_type == 'ABP':
        TrSet = np.load(config['mimic3']['paths']['train'])
        ValSet = np.load(config['mimic3']['paths']['valid'])
        refer_mean, refer_std = 80, 25  # mean, std of MIMIC III ABP clean data
        refer_min, refer_max = 20.0, 220.0
    elif signal_type == 'PPG':
        TrSet = np.load(config['mimic3']['paths']['train'].replace('ABP', 'PPG'))
        ValSet = np.load(config['mimic3']['paths']['valid'].replace('ABP', 'PPG'))
        refer_mean, refer_std = 0.448, 0.146  # mean, std of MIMIC III PPG clean data
        refer_min, refer_max = 0.0, 1.0
    else:
        raise ValueError("Invalid signal type. Must be 'ABP' or 'PPG'.")

    print('\n    *********************************************************************************\n')
    print(f'        Train set total {TrSet.shape[0]} size')
    print(f'        Valid set total {ValSet.shape[0]} size\n')

    # Print the SaveFolder location
    print(f'        Model weights will be saved to: {outdir}\n')
    print('    *********************************************************************************\n')

    #######################
    # Generating Missings #
    #######################

    TrDataFrame = tf.signal.frame(TrSet.astype('float32'), 1, 1).numpy()
    ValDataFrame = tf.signal.frame(ValSet.astype('float32'), 1, 1).numpy()
    np.random.shuffle(TrDataFrame)
    np.random.shuffle(ValDataFrame)

    # Create masks for missing values
    TrMask = np.ones_like(TrDataFrame)
    ValMask = np.ones_like(ValDataFrame)

    # Introduce random missing values
    TrMask[:, :-500, :] = np.random.choice([0, 1], size=TrDataFrame[:, :-500, :].shape, p=[0.1, 0.9])
    ValMask[:, :-500, :] = np.random.choice([0, 1], size=ValDataFrame[:, :-500, :].shape, p=[0.1, 0.9])
    TrMask[:, -500:, :] = 0
    ValMask[:, -500:, :] = 0

    # Normalize training data and introduce missing values
    Tr_X = TrDataFrame.copy()
    random_values = np.random.normal(loc=refer_mean, scale=refer_std, size=TrDataFrame.shape)
    Tr_X[TrMask == 0] = random_values[TrMask == 0]
    if signal_type == 'ABP':
        Tr_X = (Tr_X - refer_min) / (refer_max - refer_min)
    Tr_X = np.clip(Tr_X, 0.0, 1.0)

    Tr_Y = TrDataFrame.copy()
    if signal_type == 'ABP':
        Tr_Y = (Tr_Y - refer_min) / (refer_max - refer_min)
    Tr_Y = np.clip(Tr_Y, 0.0, 1.0)

    # Normalize validation data and introduce missing values
    Val_X = ValDataFrame.copy()
    random_values = np.random.normal(loc=refer_mean, scale=refer_std, size=ValDataFrame.shape)
    Val_X[ValMask == 0] = random_values[ValMask == 0]
    if signal_type == 'ABP':
        Val_X = (Val_X - refer_min) / (refer_max - refer_min)
    Val_X = np.clip(Val_X, 0.0, 1.0)

    Val_Y = ValDataFrame.copy()
    if signal_type == 'ABP':
        Val_Y = (Val_Y - refer_min) / (refer_max - refer_min)
    Val_Y = np.clip(Val_Y, 0.0, 1.0)

    # Clean up to save memory
    del random_values
    del TrDataFrame
    del ValDataFrame

    ###################
    # Compiling Model #
    ###################

    # Set up model checkpoint and CSV logger
    checkpoint_filepath = outdir + 'BDC_epoch{epoch:04}_valloss{val_loss:.7f}_loss{loss:.7f}.hdf5'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min')
    csv_logger = CSVLogger(outdir + 'training_log.csv', append=True)

    # Define the optimizer
    adam = tf.keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)

    # Model parameters
    Length = Tr_X.shape[1]
    OrigDim = Tr_X.shape[-1]
    NumLayers = config['model']['num_layers']
    EmbedDim = config['model']['embed_dim']
    NumHead = config['model']['num_heads']
    DimFC = config['model']['dim_fc']

    # Build the model
    InpLayer = Input((Length, 1))
    InpMask = Input((Length, 1))
    EmbedOut = BDCEncoder(num_layers=NumLayers, d_model=EmbedDim, num_heads=NumHead, dff=DimFC, rate=0.2, t_len=Length)(InpLayer, True, InpMask)
    Projection = Conv1D(filters=OrigDim, kernel_size=11, strides=1, padding='same', dilation_rate=1)(EmbedOut)

    BDCModel = Model([InpLayer, InpMask], Projection)
    BDCModel.compile(loss='mse', optimizer=adam)

    ############
    # Training #
    ############

    # Train the model
    BDCModel.fit([Tr_X, TrMask], Tr_Y, validation_data=([Val_X, ValMask], Val_Y), batch_size=batch_size, 
                verbose=1, shuffle=True, epochs=epochs, callbacks=[model_checkpoint_callback, csv_logger])

