import sys
import os
import time
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Add path to custom models
sys.path.append("../../Benchmarks/")
from GPVAE import JointEncoderGRU, GaussianDecoder, HI_VAE

# Enable mixed precision for performance
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Set GPU configuration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Output directory for models
outdir = '../../TrainedModels/PPGCleansing/HIVAE/'
os.makedirs(outdir, exist_ok=True)
checkpoint_prefix = os.path.join(outdir, "ckpt")

refer_mean = 0.448 # mean of MIMIC III PPG clean data
refer_std = 0.146 # std of MIMIC III PPG clean data

# Training parameters
num_epochs = 10000
batch_size = 1500
LatDim = 10

# Load and prepare data
TrSet = np.load('../../TrainDataSet/MIMIC_PPG/MIMIC_PPG_TrSet.npy')
ValSet = np.load('../../TrainDataSet/MIMIC_PPG/MIMIC_PPG_ValSet.npy')

# Frame data preparation
TrDataFrame = tf.signal.frame(TrSet.astype('float32'), 50, 50).numpy()
ValDataFrame = tf.signal.frame(ValSet.astype('float32'), 50, 50).numpy()
np.random.shuffle(TrDataFrame)
np.random.shuffle(ValDataFrame)

m_train_miss = np.zeros_like(TrDataFrame)
m_val_miss = np.zeros_like(ValDataFrame)

m_train_miss[:, :-10, :] = np.random.choice([1, 0], size=TrDataFrame[:, :-10, :].shape, p=[0.1, 0.9])
m_val_miss[:, :-10, :] = np.random.choice([1, 0], size=ValDataFrame[:, :-10, :].shape, p=[0.1, 0.9])
m_train_miss[:, -10:, :] = 1
m_val_miss[:, -10:, :] = 1

# Normalize training data and introduce missing values
Tr_X = TrDataFrame.copy()
random_values = np.random.normal(loc=refer_mean, scale=refer_std, size=TrDataFrame.shape)
Tr_X[m_train_miss == 1] = random_values[m_train_miss == 1]
Tr_X = np.clip(Tr_X, 0.0, 1.0)

Tr_Y = TrDataFrame.copy()
Tr_Y = np.clip(Tr_Y, 0.0, 1.0)

# Normalize validation data and introduce missing values
Val_X = ValDataFrame.copy()
random_values = np.random.normal(loc=80, scale=25, size=ValDataFrame.shape)
Val_X[m_val_miss == 1] = random_values[m_val_miss == 1]
Val_X = np.clip(Val_X, 0.0, 1.0)

Val_Y = ValDataFrame.copy()
Val_Y = np.clip(Val_Y, 0.0, 1.0)

data_dim = TrDataFrame.shape[-1]
time_length = TrDataFrame.shape[1]
tr_sig_nb = len(TrDataFrame)

# Clean up to save memory
del random_values
del TrDataFrame
del ValDataFrame

# Create TensorFlow dataset for training
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf_x_train_miss = (
    tf.data.Dataset.from_tensor_slices((Tr_X, Tr_Y, m_train_miss))
    .shuffle(tr_sig_nb)
    .batch(batch_size)
    .repeat()
    .prefetch(AUTOTUNE)
)

# Model build
encoder = JointEncoderGRU
decoder = GaussianDecoder

model = HI_VAE(
    latent_dim=LatDim,
    data_dim=data_dim,
    time_length=time_length,
    encoder_sizes=[100, 80, 60],
    encoder=encoder,
    decoder_sizes=[60, 80, 100],
    decoder=decoder,
    M=1,
    K=1
)

_ = tf.compat.v1.train.get_or_create_global_step()
trainable_vars = model.get_trainable_vars()

print("Encoder: ", model.encoder.net.summary())
print("Decoder: ", model.decoder.net.summary())

# For global step
global_step = tf.Variable(0, trainable=False)

gradient_clip = 1e4
learning_rate = 0.0005

# Initialize optimizer and learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Checkpoints
checkpoint = {
    "optimizer": optimizer,
    "encoder": model.encoder.net,
    "decoder": model.decoder.net,
    "global_step": global_step
}

saver = tf.train.Checkpoint(**checkpoint)

# TensorBoard
summary_writer = tf.summary.create_file_writer(outdir + 'log/', flush_millis=10000)

# Compute steps and intervals
num_steps = num_epochs * tr_sig_nb // batch_size
print_interval = num_steps // num_epochs

# Function for a single training step
@tf.function
def train_step(x_seq, y_seq, m_seq, model, optimizer):
    with tf.GradientTape() as tape:
        loss = model.compute_loss(x_seq, y_seq, m_mask=m_seq)
    gradients = tape.gradient(loss, model.trainable_variables)
    clipped_grads = [tf.clip_by_value(grad, -gradient_clip, gradient_clip) for grad in gradients]
    optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
    return loss

# Training loop
losses_train = []
losses_val = []
val_loss_check = np.inf
t0 = time.time()

with summary_writer.as_default():
    for i, (x_seq, y_seq, m_seq) in enumerate(tf_x_train_miss.take(num_steps)):
        loss = train_step(x_seq, y_seq, m_seq, model, optimizer)
        losses_train.append(loss.numpy())

        if i % print_interval == 0:
            print("================================================")
            print(f"Learning rate: {optimizer._decayed_lr('float32')} | Global gradient norm: {tf.linalg.global_norm(model.trainable_variables):.2f}")
            print(f"Step {i}) Time = {time.time() - t0:.2f}")
            loss, mse, kl = model.compute_loss(x_seq, y_seq, m_mask=m_seq, return_parts=True)
            print(f"Train loss = {loss:.5f} | mse = {mse:.5f} | KL = {kl:.5f}")
            
            tf.summary.scalar("loss_train", loss, step=i)
            tf.summary.scalar("kl_train", kl, step=i)
            tf.summary.scalar("mse_train", mse, step=i)

            # Validation
            random_indices = np.random.choice(len(Val_X), size=batch_size, replace=False)
            random_batch_X = Val_X[random_indices]
            random_batch_Y = Val_Y[random_indices]
            random_batch_m = m_val_miss[random_indices]
            val_loss, val_mse, val_kl = model.compute_loss(random_batch_X, random_batch_Y, m_mask=random_batch_m, return_parts=True)
            losses_val.append(val_loss.numpy())
            
            print(f"Validation loss = {val_loss:.5f} | mse = {val_mse:.5f} | KL = {val_kl:.5f}")

            tf.summary.scalar("loss_val", val_loss, step=i)
            tf.summary.scalar("kl_val", val_kl, step=i)
            tf.summary.scalar("mse_val", val_mse, step=i)

            if val_loss_check > val_loss:
                val_loss_check = val_loss
                model.encoder.net.save_weights(outdir + f'encoder_hivae_val{val_loss.numpy()}_valmse{val_mse.numpy()}.hdf5')
                model.decoder.net.save_weights(outdir + f'decoder_hivae_val{val_loss.numpy()}_valmse{val_mse.numpy()}.hdf5')

            t0 = time.time()

if __name__ == "__main__":
    app.run(main)
