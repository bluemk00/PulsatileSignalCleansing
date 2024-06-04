import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Permute, Reshape, LayerNormalization, LSTM, Bidirectional, GaussianNoise
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Ensure yaml is installed
try:
    import yaml
except ImportError:
    print("yaml is not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml

# Append the necessary library paths
sys.path.append("../../lib/")
sys.path.append("../utils/")
from artifact_augmentation import *
from MAIN_ModelStructure import *

# Load configuration from YAML file
with open('config_ours.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set up GPU environment
os.environ["CUDA_DEVICE_ORDER"] = config['gpu']['cuda_device_order']
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']['cuda_visible_devices']

# Clear any existing TensorFlow sessions
tf.keras.backend.clear_session()

# TensorFlow configuration for GPU memory management
gpu_config = tf.compat.v1.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.per_process_gpu_memory_fraction = config['gpu']['per_process_gpu_memory_fraction']
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=gpu_config))

# Extract configuration parameters
batch_size = config['hyperparameters']['batchsize']
epochs = config['hyperparameters']['epochs']
model_config = config['model']
model_type = model_config['modeltype']
outp_type = model_config['outptype']
gaussian_noise = model_config['gaussian_noise']
dropout_rate = model_config['dropout_rate']
mimic3_paths = model_config['mimic3']['paths']
refer_min = model_config['refer_min']
refer_max = model_config['refer_max']


if __name__ == "__main__":

    # Load and normalize data based on model type
    if model_type == 'DI':
        TrSet = np.load(mimic3_paths['train'])
        ValSet = np.load(mimic3_paths['valid'])
    elif model_type == 'DA':
        TrSet = np.load(mimic3_paths['train'].replace('ABP', 'PPG'))
        ValSet = np.load(mimic3_paths['valid'].replace('ABP', 'PPG'))
    TrSet = (TrSet - refer_min) / (refer_max - refer_min)
    ValSet = (ValSet - refer_min) / (refer_max - refer_min)

    # Use MirroredStrategy for distributed training
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )

    # Data Loader
    TrainSet = DataBatch(TrSet, batch_size, outptype=outp_type)
    ValidSet = DataBatch(ValSet, batch_size, outptype=outp_type)

    # Build model
    model_structure = ModelStructure(model_type, outp_type, gn=gaussian_noise, dr=dropout_rate)
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
        patience=epochs, 
        verbose=1, 
        restore_best_weights=True
    )
    history = LossHistory(SaveFolder + 'training_loss.csv')

    # Train the model
    AEModel.fit(
        TrainSet, 
        validation_data=ValidSet, 
        verbose=1, 
        epochs=epochs, 
        callbacks=[history, earlystopper, checkpoint]
    )
