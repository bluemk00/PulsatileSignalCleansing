import tensorflow as tf
from tensorflow.keras import layers, models

# Positional Encoding
class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles 

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model
        })
        return config


# Self Attention Mechanism
class SelfAttention(layers.Layer):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"

        self.query_weights = layers.Dense(self.head_dim)
        self.key_weights = layers.Dense(self.head_dim)
        self.value_weights = layers.Dense(self.head_dim)
        self.fc_out = layers.Dense(embed_size)

    def call(self, values, keys, query, mask):
        Q = self.query_weights(query)
        K = self.key_weights(keys)
        V = self.value_weights(values)

        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        d_k = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        out = tf.matmul(attention_weights, V)

        return out

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({
            'embed_size': self.embed_size,
            'heads': self.heads,
            'head_dim': self.head_dim
        })
        return config

def conv_block_1d(x, kernelsize, filters, dropout, batchnorm=False): 
    conv = layers.Conv1D(filters, kernelsize, kernel_initializer='he_normal', padding="same")(x)
    if batchnorm:
        conv = layers.BatchNormalization(axis=2)(conv)
    conv = layers.Activation("relu")(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    conv = layers.Conv1D(filters, kernelsize, kernel_initializer='he_normal', padding="same")(conv)
    if batchnorm:
        conv = layers.BatchNormalization(axis=2)(conv)
    conv = layers.Activation("relu")(conv)
    return conv

def SelfAttentionuNet_1D(input_shape, dropout=0.2, batchnorm=True):
    filters = [32, 64]
    kernelsize = 3
  
    inputs = layers.Input(input_shape)

    dn_1 = conv_block_1d(inputs, kernelsize, filters[0], dropout, batchnorm)
    pool_1 = layers.MaxPooling1D(pool_size=2)(dn_1)

    dn_2 = conv_block_1d(pool_1, kernelsize, filters[1], dropout, batchnorm)

    x = PositionalEncoding(dn_1.shape[1], dn_1.shape[2])(dn_1)
    att_2 = SelfAttention(dn_1.shape[2], 1)(x, x, x, mask=None)
    up_2 = layers.UpSampling1D(size=2)(dn_2)
    up_2 = layers.concatenate([up_2, att_2], axis=2)
    up_conv_2 = conv_block_1d(up_2, kernelsize, filters[0], dropout, batchnorm)

    x = PositionalEncoding(up_conv_2.shape[1], up_conv_2.shape[2])(up_conv_2)
    att_final = SelfAttention(up_conv_2.shape[2], 1)(x, x, x, mask=None)

    conv_final = layers.Conv1D(1, kernel_size=1)(att_final)
    outputs = layers.Activation('sigmoid')(conv_final)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model
    
def custom_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    y_true_range = tf.reduce_max(y_true) - tf.reduce_min(y_true)
    y_pred_range = tf.reduce_max(y_pred) - tf.reduce_min(y_pred)
    
    range_difference = tf.abs(y_true_range - y_pred_range)
    
    final_loss = mse_loss + range_difference*mse_loss
     
    return final_loss