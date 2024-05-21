import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

# Set up environment variables for GPU usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure GPU memory usage
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.98
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
tf.compat.v1.enable_eager_execution()

# Append custom library paths
sys.path.append("../../Benchmarks/")
from BDC_utils import *

# Reference mean and standard deviation for normalization
refer_mean = 80  # Mean of MIMIC III PPG clean data
refer_std = 25   # Std of MIMIC III PPG clean data

def main():
    batch_size = 30

    # Output directory for models
    outdir = '../../TrainedModels/ABPCleansing/BDC/'
    os.makedirs(outdir, exist_ok=True)
    checkpoint_prefix = os.path.join(outdir, "ckpt")

    ################
    # Loading Data #
    ################

    # Load training and validation datasets
    TrSet = np.load('../../TrainDataSet/MIMIC_ABP/MIMIC_ART_TrSet.npy')
    ValSet = np.load('../../TrainDataSet/MIMIC_ABP/MIMIC_ART_ValSet.npy')

    #######################
    # Generating Missings #
    #######################

    # Create data frames and shuffle them
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
    Tr_X = (Tr_X - 20.0) / (220.0 - 20.0)
    Tr_X = np.clip(Tr_X, 0.0, 1.0)

    Tr_Y = (TrDataFrame - 20.0) / (220.0 - 20.0)
    Tr_Y = np.clip(Tr_Y, 0.0, 1.0)

    # Normalize validation data and introduce missing values
    Val_X = ValDataFrame.copy()
    random_values = np.random.normal(loc=refer_mean, scale=refer_std, size=ValDataFrame.shape)
    Val_X[ValMask == 0] = random_values[ValMask == 0]
    Val_X = (Val_X - 20.0) / (220.0 - 20.0)
    Val_X = np.clip(Val_X, 0.0, 1.0)

    Val_Y = (ValDataFrame - 20.0) / (220.0 - 20.0)
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
    NumLayers = 2
    EmbedDim = 64
    NumHead = 1
    DimFC = 200

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
    BDCModel.fit([Tr_X, TrMask], Tr_Y, validation_data=([Val_X, ValMask], Val_Y), batch_size=batch_size, verbose=1, shuffle=True, epochs=10000, callbacks=[model_checkpoint_callback, csv_logger])

if __name__ == "__main__":
    main()
