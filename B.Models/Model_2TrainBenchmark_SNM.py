import sys
import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

# Append the necessary library paths
sys.path.append("./Model_Utils/")
from SNM_GRUD import *
from SNM_Interpolate import *
from SNM_SupNotMIWAE import *


if __name__ == "__main__":
    # Argument parser for command line options
    parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
    parser.add_argument('--signal_type', '-s', type=str, required=True, choices=['ABP', 'PPG'], help="Signal type (ABP or PPG)")
    parser.add_argument('--batch_size', '-b', type=int, default=1500, help="Batch size (default: 1500)")
    parser.add_argument('--epochs', '-e', type=int, default=10000, help="Number of epochs (default: 10000)")
    parser.add_argument('--gpu', '-g', type=str, default="0", help="GPU device id (default: 0)")

    args = parser.parse_args()

    # Set up GPU environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.98
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))     
    tf.compat.v1.enable_eager_execution()

    signal_type = args.signal_type
    batch_size = args.batch_size
    epochs = args.epochs

    # Output directory for models
    outdir = f'./Model_TrainedWeights/{signal_type}Cleansing/SNM/'
    os.makedirs(outdir, exist_ok=True)
    checkpoint_prefix = os.path.join(outdir, "ckpt")

    ################
    # Loading Data #
    ################

    # Load and normalize data based on model type
    if signal_type == 'ABP':
        TrSet = np.load('../A.Data/Data_1ModelTrain/MIMIC3_ABP/MIMIC_ART_TrSet.npy')
        ValSet = np.load('../A.Data/Data_1ModelTrain/MIMIC3_ABP/MIMIC_ART_ValSet.npy')
        refer_mean, refer_std = 80, 25 # mean, std of MIMIC III ABP clean data
        refer_min, refer_max = 20.0, 220.0
    elif signal_type == 'PPG':
        TrSet = np.load('../A.Data/Data_1ModelTrain/MIMIC3_PPG/MIMIC_PPG_TrSet.npy')
        ValSet = np.load('../A.Data/Data_1ModelTrain/MIMIC3_PPG/MIMIC_PPG_ValSet.npy')
        refer_mean, refer_std = 0.448, 0.146 # mean, std of MIMIC III PPG clean data
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

    TrDataFrame = tf.signal.frame(TrSet.astype('float32'), 50, 50).numpy()
    ValDataFrame = tf.signal.frame(ValSet.astype('float32'), 50, 50).numpy()
    np.random.shuffle(TrDataFrame)
    np.random.shuffle(ValDataFrame)

    # Create masks for missing values
    TrMask = np.ones_like(TrDataFrame)
    ValMask = np.ones_like(ValDataFrame)

    # Introduce random missing values
    TrMask[:, :-10, :] = np.random.choice([0, 1], size=TrDataFrame[:, :-10, :].shape, p=[0.1, 0.9])
    ValMask[:, :-10, :] = np.random.choice([0, 1], size=ValDataFrame[:, :-10, :].shape, p=[0.1, 0.9])
    TrMask[:, -10:, :] = 0
    ValMask[:, -10:, :] = 0

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

    data_dim = TrDataFrame.shape[-1]
    time_length = TrDataFrame.shape[1]
    tr_sig_nb = len(TrDataFrame)

    # Clean up to save memory
    del random_values
    del TrDataFrame
    del ValDataFrame

    ###################
    # Compiling Model #
    ###################

    checkpoint_filepath = outdir + 'SNM_epoch{epoch:04}_valloss{val_loss:.7f}_loss{loss:.7f}.hdf5'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)

    # CSV Logger
    csv_logger = CSVLogger(outdir + 'training_log.csv', append=True)

    learning_rate = 0.0005
    decay = 1e-6
    adam = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)

    Length = Tr_X.shape[1]
    OrigDim = Tr_X.shape[-1]
    InpLayer = tf.keras.layers.Input((Length, OrigDim))
    InpMask  = tf.keras.layers.Input((Length, OrigDim))
    ImputeOut = SupNotMIWAE(n_train_latents=10, n_train_samples=1)([InpLayer, InpMask])
    SNMModel = Model([InpLayer, InpMask], ImputeOut)
    SNMModel.compile(loss='mse', optimizer=adam)

    ############
    # Training #
    ############

    SNMModel.fit([Tr_X, TrMask], tf.convert_to_tensor(Tr_Y), validation_data=([Val_X, ValMask], tf.convert_to_tensor(Val_Y)), 
                 batch_size=batch_size, verbose=1, shuffle=True, epochs=epochs, callbacks=[model_checkpoint_callback, csv_logger])
