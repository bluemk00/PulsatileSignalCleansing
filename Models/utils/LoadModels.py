import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D
from tensorflow.keras import Model

# Configure GPU settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.98
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))     
tf.compat.v1.enable_eager_execution()

# Append necessary paths
sys.path.append('../../lib/')

# Import custom functions and models
from functions import *
from cumm_pred_dtaidtw import *
from GPVAE import *
from BDC_utils import *
from SNM_GRUD import *
from SNM_Interpolate import *
from SNM_SupNotMIWAE import *
from MAIN_ModelStructure import build_model_structure

# Load models and their weights
def load_models_ABP(model_path):
    
    # Build and load DI models
    DI = build_model_structure(2)
    DI.build(input_shape=(None, 3000))  # Build the model with the correct input shape
    DI.load_weights(model_path+'DI.hdf5')

    DI_D = build_model_structure(1)
    DI_D.build(input_shape=(None, 3000))  # Build the model with the correct input shape
    DI_D.load_weights(model_path+'DI_D.hdf5')

    DI_A = build_model_structure(0)
    DI_A.build(input_shape=(None, 3000))  # Build the model with the correct input shape
    DI_A.load_weights(model_path+'DI_A.hdf5')

    # Build and load HIVAE model
    HIVAE = HI_VAE(latent_dim=10, data_dim=50, time_length=60, encoder_sizes=[100, 80, 60], encoder=JointEncoderGRU,
                    decoder_sizes=[60,80,100], decoder=GaussianDecoder, M=1, K=1, beta=0.1)
    _ = tf.compat.v1.train.get_or_create_global_step()
    trainable_vars = HIVAE.get_trainable_vars()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    HIVAE.encoder.net.load_weights(model_path+'HIVAE_encoder.hdf5')
    HIVAE.decoder.net.load_weights(model_path+'HIVAE_decoder.hdf5')

    # Build and load GPVAE model
    GPVAE = GP_VAE(latent_dim=10, data_dim=50, time_length=60, encoder_sizes=[100, 80, 60], encoder=BandedJointEncoderGRU,
                    decoder_sizes=[60,80,100], decoder=GaussianDecoder, M=1, K=1, beta=0.1)
    _ = tf.compat.v1.train.get_or_create_global_step()
    trainable_vars = GPVAE.get_trainable_vars()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    GPVAE.encoder.net.load_weights(model_path+'GPVAE_encoder.hdf5')
    GPVAE.decoder.net.load_weights(model_path+'GPVAE_decoder.hdf5')

    # Build and load SNM model
    InpLayer = tf.keras.layers.Input((60, 50))
    InpMask  = tf.keras.layers.Input((60, 50))
    ImputeOut = SupNotMIWAE(n_train_latents=10, n_train_samples=1)([InpLayer, InpMask])
    SNM = Model([InpLayer,InpMask ], ImputeOut)
    SNM.compile(loss='mse', optimizer='adam')
    SNM.load_weights(model_path+'SNM.hdf5')

    # Build and load BDC model
    Length = 3000
    OrigDim = 1
    NumLayers = 2
    EmbedDim = 64
    NumHead = 1
    DimFC = 200

    InpLayer = Input((Length, 1))
    InpMask  = Input((Length, 1))

    EmbedOut = BDCEncoder(num_layers = NumLayers, d_model = EmbedDim, num_heads = NumHead, dff = DimFC, t_len=Length)(InpLayer, False, InpMask)

    Projection = Conv1D(filters=OrigDim, kernel_size=11, strides=1, padding='same', dilation_rate=1)(EmbedOut)

    BDC = Model([InpLayer,InpMask ], Projection)
    BDC.compile(loss='mse', optimizer = 'adam')
    BDC.compile(loss='mse', optimizer = 'adam')
    BDC.load_weights(model_path+'BDC.hdf5')

    return DI, DI_D, DI_A, HIVAE, GPVAE, SNM, BDC


# Load models and their weights
def load_models_PPG(model_path):

    # Build and load DA models
    DA = build_model_structure(2)
    DA.build(input_shape=(None, 3000))  # Build the model with the correct input shape
    DA.load_weights(model_path+'DA.hdf5')

    DA_D = build_model_structure(1)
    DA_D.build(input_shape=(None, 3000))  # Build the model with the correct input shape
    DA_D.load_weights(model_path+'DA_D.hdf5')

    DA_A = build_model_structure(0)
    DA_A.build(input_shape=(None, 3000))  # Build the model with the correct input shape
    DA_A.load_weights(model_path+'DA_A.hdf5')

    # Build and load HIVAE model
    HIVAE = HI_VAE(latent_dim=10, data_dim=50, time_length=60, encoder_sizes=[100, 80, 60], encoder=JointEncoderGRU,
                    decoder_sizes=[60,80,100], decoder=GaussianDecoder, M=1, K=1, beta=0.1)
    _ = tf.compat.v1.train.get_or_create_global_step()
    trainable_vars = HIVAE.get_trainable_vars()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    HIVAE.encoder.net.load_weights(model_path+'HIVAE_encoder.hdf5')
    HIVAE.decoder.net.load_weights(model_path+'HIVAE_decoder.hdf5')

    # Build and load GPVAE model
    GPVAE = GP_VAE(latent_dim=10, data_dim=50, time_length=60, encoder_sizes=[100, 80, 60], encoder=BandedJointEncoderGRU,
                    decoder_sizes=[60,80,100], decoder=GaussianDecoder, M=1, K=1, beta=0.1)
    _ = tf.compat.v1.train.get_or_create_global_step()
    trainable_vars = GPVAE.get_trainable_vars()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    GPVAE.encoder.net.load_weights(model_path+'GPVAE_encoder.hdf5')
    GPVAE.decoder.net.load_weights(model_path+'GPVAE_decoder.hdf5')

    # Build and load SNM model
    InpLayer = tf.keras.layers.Input((60, 50))
    InpMask  = tf.keras.layers.Input((60, 50))
    ImputeOut = SupNotMIWAE(n_train_latents=10, n_train_samples=1)([InpLayer, InpMask])
    SNM = Model([InpLayer,InpMask ], ImputeOut)
    SNM.compile(loss='mse', optimizer='adam')
    SNM.load_weights(model_path+'SNM.hdf5')

    # Build and load BDC model
    Length = 3000
    OrigDim = 1
    NumLayers = 2
    EmbedDim = 64
    NumHead = 1
    DimFC = 200

    InpLayer = Input((Length, 1))
    InpMask  = Input((Length, 1))

    EmbedOut = BDCEncoder(num_layers = NumLayers, d_model = EmbedDim, num_heads = NumHead, dff = DimFC, t_len=Length)(InpLayer, False, InpMask)

    Projection = Conv1D(filters=OrigDim, kernel_size=11, strides=1, padding='same', dilation_rate=1)(EmbedOut)

    BDC = Model([InpLayer,InpMask ], Projection)
    BDC.compile(loss='mse', optimizer = 'adam')
    BDC.compile(loss='mse', optimizer = 'adam')
    BDC.load_weights(model_path+'BDC.hdf5')

    return DA, DA_D, DA_A, HIVAE, GPVAE, SNM, BDC
