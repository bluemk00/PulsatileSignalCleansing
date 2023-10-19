import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation
from tensorflow.keras import Model
import numpy as np

NumLayers=2
EmbedDim = 64
NumHead = 1
DimFC = 200
BottleneckSize = EmbedDim // 8
Groups = NumHead * 2
HeadDim = EmbedDim // NumHead


class DilatedBottleneckBlock(tf.keras.layers.Layer):
    def __init__(self,  out_channel=64, bottleneck=BottleneckSize, kernel_size=15, dilation=1, groups=Groups, firstlayergroups=None):
        super(DilatedBottleneckBlock, self).__init__()
        self.firstlayergroups = firstlayergroups

        # Bottle
        self.bottle = tf.keras.layers.Conv1D(bottleneck, kernel_size=1, groups=groups)
        
        # ReLU
        self.relu = tf.keras.layers.ReLU()

        # BatchNorm
        self.bn = tf.keras.layers.BatchNormalization(center=True, scale=True)

        # Dilated Conv
        dilation_padding = (kernel_size - 1) // 2 * dilation
        if firstlayergroups:
            self.dilated_conv = tf.keras.layers.Conv1D(out_channel, kernel_size, dilation_rate=dilation, padding='same', groups=firstlayergroups)
        else:
            self.dilated_conv = tf.keras.layers.Conv1D(out_channel, kernel_size, dilation_rate=dilation, padding='same', groups=groups)

    def call(self, x):
        bottle_out = self.bottle(x)
        conv_out = self.dilated_conv(bottle_out)
        
        if self.firstlayergroups:
            return self.bn(self.relu(conv_out) + tf.tile(x, [1, 1, 2]))
        else:
            return self.bn(self.relu(conv_out) + x)


class DilatedBottleneckNet(tf.keras.Model):
    def __init__(self,  out_channel=256, bottleneck=BottleneckSize, kernel_size=15, dilation=50, groups=Groups):
        super(DilatedBottleneckNet, self).__init__()

        self.layer0 = DilatedBottleneckBlock( out_channel * 2, bottleneck * 2, kernel_size, dilation, 1, firstlayergroups=groups)
        self.layer1 = DilatedBottleneckBlock( out_channel * 2, bottleneck * 2, kernel_size, dilation * 2, groups)
        self.layer2 = DilatedBottleneckBlock( out_channel * 2, bottleneck * 2, kernel_size, dilation * 4, groups)
        self.layer3 = DilatedBottleneckBlock( out_channel * 2, bottleneck * 2, kernel_size, dilation * 8, groups)
        self.layer4 = DilatedBottleneckBlock( out_channel * 2, bottleneck * 2, kernel_size, dilation * 16, groups)
        self.layer5 = DilatedBottleneckBlock( out_channel * 2, bottleneck * 2, kernel_size, dilation * 32, groups)

    def call(self, x):
            x0 = self.layer0(x)
            x1 = self.layer1(x0)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            x5 = self.layer5(x4)
            
            return x5 # return shape (batch, length, embed)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)# (..., seq_len_q, seq_len_k)
    
    
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class CustomAttention(tf.keras.layers.Layer):
    
    def __init__(self, d_model, num_heads, bottleneck=BottleneckSize, groups=Groups, rate=0.1, t_len=-1):
        super(CustomAttention, self).__init__()
    
        self.num_heads = num_heads
        self.d_model = d_model
        self.scaling = num_heads ** -0.5
        self.t_len = t_len 

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        
        self.DBN = DilatedBottleneckNet(out_channel =d_model , bottleneck = bottleneck, groups=groups)
        
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        self.dr = tf.keras.layers.Dropout(rate)
        
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size,  self.t_len, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    
    def call(self,  x, training, mask):
        q, k =  tf.split(self.DBN(x), 2, axis=-1)
        batch_size = tf.shape(q)[0]
        v = self.wv(x)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaling q
        q = q * self.scaling

        # Computing the batch matrix multiplication
        AttOutWeights =  tf.linalg.matmul(q, k, transpose_b=True)
        
        # Masking before normalization for proper attention distribution
        Mask_repeated = tf.tile(mask[:, None], [1, self.num_heads, 1, 1])
        masked_out_weights = tf.multiply(AttOutWeights, Mask_repeated) 
        
        # Subtracting max value for numerical stability before taking exponential
        max_vals = tf.math.reduce_max(masked_out_weights, axis=-1, keepdims=True)
        AttOutputWeights = tf.exp(masked_out_weights - max_vals) 
        
        # Applying normalization
        sum_vals = tf.math.reduce_sum(AttOutputWeights, axis=-1, keepdims=True)
        AttOutputWeights /= sum_vals
        
        # Applying dropout
        AttOutputWeights = self.dr(AttOutputWeights)

        
        # Computing attention output using batched matrix multiplication
        AttOutput = tf.linalg.matmul(AttOutputWeights, v)

        # Reshaping and transposing tensors
        AttOutput = tf.reshape(tf.transpose(AttOutput, perm=[0, 2, 1, 3]), [-1, self.t_len, self.d_model])
        
        
        '''
        scaled_attention, attention_weights = scaled_dot_product_attention( q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,  (-1, self.t_len, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        '''
        
        return AttOutput


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])



class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, bottleneck=BottleneckSize, groups=Groups,  rate=0.1, t_len=-1):
        super(EncoderLayer, self).__init__()

        self.catt = CustomAttention(d_model, num_heads, bottleneck=bottleneck, t_len=t_len)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
        self.wv = tf.keras.layers.Dense(d_model)
        
        
    def call(self, x, training, mask):

        
        attn_output = self.catt(x, training, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        
            
        return out2


class BDCEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, bottleneck=BottleneckSize, groups=Groups,  rate=0.1, t_len=-1):
        super(BDCEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.convembed = tf.keras.layers.Conv1D(filters=d_model, kernel_size=11, strides=1, padding='same', dilation_rate=1)
        self.bn = tf.keras.layers.BatchNormalization()
        
      
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate=rate, t_len=t_len)  for _ in range(num_layers)]
        

        #self.dropout = tf.keras.layers.Dropout(rate)

    
    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]
        
        x = self.convembed(x)
        x = self.bn(x)
        x = tf.nn.relu(x)

     
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        
        return x  # (batch_size, input_seq_len, d_model)