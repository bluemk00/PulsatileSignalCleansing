import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Permute, Reshape, LayerNormalization, LSTM, Bidirectional, GaussianNoise
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Append the necessary library paths
sys.path.append("../lib/")
sys.path.append("./Model_Utils/")
from artifact_augmentation import *
from MAIN_ModelStructure import *

# Argument parser for command line options
parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
parser.add_argument('--modelname', '-m', type=str, required=True, choices=['DI','DI-D','DI-A','DA','DA-D','DA-A'], 
                                         help="Model name (DI, DI-D, DI-A, DA, DA-D, or DA-A")
parser.add_argument('--batchsize', '-b', type=int, default=1500, help="Batch size (default: 1500)")
parser.add_argument('--epochs', '-e', type=int, default=10000, help="Number of epochs (default: 10000)")
parser.add_argument('--gpu', '-g', type=str, default="0", help="GPU device id (default: 0)")

args = parser.parse_args()

# Set up GPU environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Clear any existing TensorFlow sessions
tf.keras.backend.clear_session()

# TensorFlow configuration for GPU memory management
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.98
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# Set gaussian_noise and dropout_rate based on model type
if args.modelname == 'DI':
    modeltype, outptype, gaussian_noise, dropout_rate = 'DI', 2, 0.05, 0.1
elif args.modelname == 'DI-D':
    modeltype, outptype, gaussian_noise, dropout_rate = 'DI', 1, 0.05, 0.1
elif args.modelname == 'DI-A':
    modeltype, outptype, gaussian_noise, dropout_rate = 'DI', 0, 0.05, 0.1
elif args.modelname == 'DA':
    modeltype, outptype, gaussian_noise, dropout_rate = 'DA', 2, 0.05, 0.1
elif args.modelname == 'DA-D':
    modeltype, outptype, gaussian_noise, dropout_rate = 'DA', 1, 0.05, 0.1
elif args.modelname == 'DA-A':
    modeltype, outptype, gaussian_noise, dropout_rate = 'DA', 0, 0.05, 0.1
else:
    raise ValueError("Invalid model name. Must be one of: 'DI', 'DI-D', 'DI-A', 'DA', 'DA-D', or 'DA-A'.")


if __name__ == "__main__":
    BatchSize = args.batchsize
    Epochs = args.epochs

    # Load and normalize data based on model type
    if modeltype == 'DI':
        TrSet = np.load('../A.Data/Data_1ModelTrain/MIMIC3_ABP/MIMIC_ART_TrSet.npy')
        ValSet = np.load('../A.Data/Data_1ModelTrain/MIMIC3_ABP/MIMIC_ART_ValSet.npy')
        TrSet = (TrSet - 20.0) / (220.0 - 20.0)
        ValSet = (ValSet - 20.0) / (220.0 - 20.0)
    elif modeltype == 'DA':
        TrSet = np.load('../A.Data/Data_1ModelTrain/MIMIC3_PPG/MIMIC_PPG_TrSet.npy')
        ValSet = np.load('../A.Data/Data_1ModelTrain/MIMIC3_PPG/MIMIC_PPG_ValSet.npy')

    # Use MirroredStrategy for distributed training
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )

    # Data Loader
    TrainSet = DataBatch(TrSet, BatchSize, outptype=outptype)
    ValidSet = DataBatch(ValSet, BatchSize, outptype=outptype)

    # Build model
    model_structure = ModelStructure(modeltype, outptype, gn=gaussian_noise, dr=dropout_rate)
    AEModel, SaveFolder, SaveFilePath = model_structure.build_model()

    print('\n    *********************************************************************************\n')
    print(f'        Train set total {TrSet.shape[0]} size')
    print(f'        Valid set total {ValSet.shape[0]} size\n')

    # Print the SaveFolder location
    print(f'        Model weights will be saved to: {SaveFolder}\n')

    print('    *********************************************************************************\n')
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        SaveFolder + SaveFilePath, 
        monitor='val_loss', 
        verbose=0, 
        save_best_only=True, 
        mode='auto', 
        period=1
    )
    earlystopper = EarlyStopping(
        monitor='val_loss', 
        patience=Epochs, 
        verbose=1, 
        restore_best_weights=True
    )
    history = LossHistory(SaveFolder + 'training_loss.csv')

    # Train the model
    AEModel.fit(
        TrainSet, 
        validation_data=ValidSet, 
        verbose=1, 
        epochs=Epochs, 
        callbacks=[history, earlystopper, checkpoint]
    )
