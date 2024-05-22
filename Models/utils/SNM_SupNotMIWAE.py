'''
This code is a modified version of the Supplementary Material for the paper 
"Probabilistic Imputation for Time-series Classification with Missing Data",
downloaded from the ICLR 2023 OpenReview. 
It was originally submitted in 2022 and has been modified by K. Park.
Unlike the original, this version is intended for imputation purposes.
'''

import numpy as np

from collections import namedtuple
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import backend as K
from tensorflow.keras import layers, initializers, regularizers, constraints, activations

import sys
from SNM_GRUD import *
from SNM_Interpolate import *

Length = 60
OrigDim = 50


class SupNotMIWAE(tf.keras.Model):
    def __init__(
        self, 
        n_hidden:int=128, 
        n_train_latents:int = 10, 
        n_train_samples:int=1, 
        observe_dropout=None, 
        return_sequences=True, 
        z_dim:int = 32,
        min_latent_sigma: float = 0.0001,
        min_sigma: float = 0.001,
    ):
        super(SupNotMIWAE, self).__init__()
        self.n_hidden = n_hidden
        self.feature_dim = None
        self.n_train_samples = n_train_samples
        self.n_train_latents = n_train_latents
        self.return_sequences = return_sequences
        self.z_dim = z_dim
        self.min_latent_sigma = min_latent_sigma
        self.min_sigma = min_sigma
        
        if observe_dropout is None:
            self.observe_dropout = tf.constant(0., dtype=tf.float32)
        else:
            self.observe_dropout = observe_dropout

        # Encoder with GRU-D
        self.encoder = GRUD(self.n_hidden, return_sequences=True, x_imputation="decay", feed_masking=True, name="encoder")
        min_softplus = lambda x: self.min_latent_sigma + (1 - self.min_latent_sigma) * tf.nn.softplus(x)
        self.encoder_mu    = layers.Dense(self.z_dim, activation=None, name="encoder/mu")
        self.encoder_sigma = layers.Dense(self.z_dim, activation=min_softplus, name="encoder/sigma")

        # Prior
        self.prior_gru = layers.GRU(self.z_dim, return_sequences=True,   name="prior/gru")
        self.prior_mu  = layers.Dense(self.z_dim, activation=None,         name="prior/mu")
        self.prior_std = layers.Dense(self.z_dim, activation=min_softplus, name="prior/std")
        
        # Decoder with GRU
        self.decoder = layers.GRU(self.n_hidden, return_sequences=True, name="decoder")
        min_softplus = lambda x: self.min_sigma + (1 - self.min_sigma) * tf.nn.softplus(x)
        self.decoder_mu    = layers.Dense(OrigDim, activation=None,         name="decoder/mu")
        self.decoder_sigma = layers.Dense(OrigDim, activation=min_softplus, name="decoder/sigma")
        
        # Impute
        self.interpolator = DecayInterpolate(name="interpolator")

    def encode(self, times, x_obs, missing_mask):
        times = tf.expand_dims(times, axis=-1)
    
        r = self.encoder(GRUDInput(x_obs, missing_mask, times))
        z_mu = self.encoder_mu(r)
        z_sigma = self.encoder_sigma(r)
        
        q_z = tfd.Normal(loc=z_mu, scale=z_sigma)
        return q_z   
        
    def prior(self, z_samples):
        shape = tf.shape(z_samples)
        z_samples = tf.reshape(z_samples, shape=(-1, Length, self.z_dim))
        r = self.prior_gru(z_samples[:,1:,:],initial_state=z_samples[:,0,:])
        r = tf.reshape(r, shape=(self.n_train_latents, -1, Length-1, self.z_dim))
        p_z_mu = self.prior_mu(r)
        p_z_sigma = self.prior_std(r)
        p_z_mu    = tf.pad(p_z_mu,    [[0, 0], [0, 0], [1, 0], [0, 0]], constant_values=0)
        p_z_sigma = tf.pad(p_z_sigma, [[0, 0], [0, 0], [1, 0], [0, 0]], constant_values=1)
        p_z = tfd.Normal(loc=p_z_mu, scale=p_z_sigma)
        
        return p_z

    def decode(self, times, z):
        shape = tf.shape(z)
        z = tf.reshape(z, shape=(-1, Length, self.z_dim))
        h = self.decoder(z)
        h = tf.reshape(h, shape=(self.n_train_latents, -1, Length, self.n_hidden))
        x_tilde_mu = self.decoder_mu(h)
        x_tilde_sigma = self.decoder_sigma(h)
        p_x_tilde = tfd.Normal(loc=x_tilde_mu, scale=x_tilde_sigma)
                
        return p_x_tilde

    def missing_model_func(self, x):
        return self.missing_model(x)

    def call(self, inputs, training=True):
        x_obs, missing_mask = inputs
        last_observed, mean = compute_last_observed_and_mean(x_obs, missing_mask)
        
        delta_t = compute_delta_t(missing_mask)
        
        single_time_array = tf.range(0, x_obs.shape[1] * 0.5, 0.5, dtype=tf.float32)
        dynamic_shape = tf.shape(x_obs)
        times = tf.broadcast_to(single_time_array, [dynamic_shape[0], dynamic_shape[1]])
        
        # Set feature_dim based on x_obs if not already set
        if self.feature_dim is None:
            self.feature_dim = x_obs.shape[-1]
            
        n_samples = self.n_train_samples
        n_latents = self.n_train_latents
        
        # === VAE ===

        # Encoder
        q_z = self.encode(times, x_obs, missing_mask)
        
        # Latent
        z_samples = q_z.sample(n_latents)
        
        # Prior
        p_z = self.prior(z_samples)
        
        # Decoder
        p_x_tilde = self.decode(times, z_samples)
        
        # Impute
                
        # log p(xᵒ | zₖ)
        log_p_x_obs_given_z = tf.reduce_sum(tf.where(
            tf.cast(missing_mask, tf.bool),
            p_x_tilde.log_prob(x_obs),
            0.
        ), axis=-1)

        log_p_z = p_z.log_prob(z_samples)
        log_q_z_given_x_obs = q_z.log_prob(z_samples)
        
        if self.return_sequences:
            log_p_x_obs_given_z = tf.cumsum(log_p_x_obs_given_z, axis=-1)
            log_p_z = tf.cumsum(log_p_z, axis=-1)
            log_q_z_given_x_obs = tf.cumsum(log_q_z_given_x_obs, axis=-1)
        else:
            log_p_x_obs_given_z = tf.reduce_sum(log_p_x_obs_given_z, axis=-1, keepdims=True)
            log_p_z = tf.reduce_sum(log_p_z, axis=-1, keepdims=True)
            log_q_z_given_x_obs = tf.reduce_sum(log_q_z_given_x_obs, axis=-1, keepdims=True)

        
        # Generate: xₖⱼᵐ ~ p(xₖᵐ | zₖ)
        x_tilde = self.generate(p_x_tilde, training=training)

        # Dropout
        drop_mask, log_m = self.generate_observe_dropout_mask(tf.shape(x_tilde), missing_mask, training=training)

        # Impute
        x_impute = self.impute(times, x_obs, x_tilde, drop_mask)
        
        
        return x_impute
    
    
    def generate(self, p_x_tilde, training=True):
        
        n_samples = self.n_train_samples

        # xₖⱼᵐ ~ p(xₖᵐ | zₖ)
        x_tilde = p_x_tilde.sample(n_samples)

        return x_tilde

    def generate_observe_dropout_mask(self, shape, missing_mask, training=True):
        missing_mask = tf.cast(missing_mask, tf.bool)
        if tf.reduce_any(self.observe_dropout > 0.) and training:
            p_m = tfd.Bernoulli(probs=(1. - self.observe_dropout))
            drop_mask = p_m.sample(shape if self.observe_dropout.ndim == 0 else shape[:-1])

            log_m = tf.reduce_sum(tf.where(
                missing_mask,
                p_m.log_prob(drop_mask),
                0.,
            ), axis=-1)

            drop_mask = tf.cast(drop_mask, dtype=bool) & missing_mask

        else:
            drop_mask = tf.broadcast_to(missing_mask, shape)

            log_m = tf.zeros(shape[:4])

        if self.return_sequences:
            log_m = tf.cumsum(log_m, axis=-1)
        else:
            log_m = tf.reduce_sum(log_m, axis=-1, keepdims=True)

        return drop_mask, log_m

    def impute(self, times, x_obs, x_tilde, missing_mask):
        # Interpolate obs and combine with generated missing
        x_comb = tf.where(tf.cast(missing_mask,tf.bool), x_obs, x_tilde)

        shape = tf.shape(x_comb)
        n_tiles = self.n_train_samples * self.n_train_latents
        
        times = tf.expand_dims(tf.tile(times, multiples=[n_tiles, 1]), axis=-1)
        x_comb = tf.reshape(x_comb, shape=[-1, Length, OrigDim])
        missing_mask = tf.reshape(missing_mask, shape=[n_tiles * shape[2], shape[3], shape[4]])

        x_impute = self.interpolator(InterpolateInput(values=x_comb, mask=missing_mask, times=times))
        x_impute = tf.reshape(x_impute, shape=[shape[0], shape[1], shape[2], shape[3], shape[4]])   
        
        x_impute_mean = tf.reduce_mean(x_impute, axis=[0,1])

        return x_impute_mean
    