import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Permute, Reshape, LayerNormalization, LSTM, Bidirectional, GaussianNoise
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Append the necessary library paths
sys.path.append("../../../lib/")
sys.path.append("../../Ours/")

from artifact_augmentation import *
from ModelStructure import *

# Set up GPU environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Clear any existing TensorFlow sessions
tf.keras.backend.clear_session()

# TensorFlow configuration for GPU memory management
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.98
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# Define model type and output type
modeltype, outptype = 'DA', 2
gaussian_noise, dropout_rate = 0.05, 0.1

'''
Model type and output type options:
'DI', 2 : DI
'DI', 1 : DI-D
'DI', 0 : DI-A
'DA', 2 : DA
'DA', 1 : DA-D
'DA', 0 : DA-A
'''

if __name__ == "__main__":
    BatchSize = 1500

    # Load and normalize data based on model type
    if modeltype == 'DI':
        TrSet = np.load('../../TrainDataSet/MIMIC_ART_TrSet.npy')
        ValSet = np.load('../../TrainDataSet/MIMIC_ART_ValSet.npy')
        TrSet = (TrSet - 20.0) / (220.0 - 20.0)
        ValSet = (ValSet - 20.0) / (220.0 - 20.0)
    elif modeltype == 'DA':
        TrSet = np.load('../../TrainDataSet/MIMIC_PPG_TrSet.npy')
        ValSet = np.load('../../TrainDataSet/MIMIC_PPG_ValSet.npy')

    print('************************************************')
    print(f'    Train set total {TrSet.shape[0]} size   ')
    print(f'    Valid set total {ValSet.shape[0]} size   ')
    print('************************************************')

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
        patience=10000, 
        verbose=1, 
        restore_best_weights=True
    )
    history = LossHistory(SaveFolder + 'training_loss.csv')

    # Train the model
    AEModel.fit(
        TrainSet, 
        validation_data=ValidSet, 
        verbose=1, 
        epochs=10000, 
        callbacks=[history, earlystopper, checkpoint]
    )
