'''
This is a modified version from the source: https://github.com/rehg-lab/pulseimpute
Modifications made by J. Kim
'''

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation
from tensorflow.keras import Model

NumLayers=2
EmbedDim = 64
NumHead = 1
DimFC = 200
BottleneckSize = EmbedDim // 8
Groups = NumHead * 2
HeadDim = EmbedDim // NumHead


class DilatedBottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, out_channel=64, bottleneck=BottleneckSize, kernel_size=15, dilation=1, groups=Groups, firstlayergroups=None):
        super(DilatedBottleneckBlock, self).__init__()
        self.firstlayergroups = firstlayergroups

        # Bottle
        self.bottle = tf.keras.layers.Conv1D(bottleneck, kernel_size=1, groups=groups)
        
        # ReLU
        self.relu = tf.keras.layers.ReLU()

        # BatchNorm
        self.bn = tf.keras.layers.BatchNormalization(center=True, scale=True)

        # Dilated Conv
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

    def get_config(self):
        config = super(DilatedBottleneckBlock, self).get_config()
        config.update({
            'firstlayergroups': self.firstlayergroups,
            'bottleneck': self.bottle.filters,
            'kernel_size': self.bottle.kernel_size,
            'groups': self.bottle.groups,
            'out_channel': self.dilated_conv.filters,
            'dilation': self.dilated_conv.dilation_rate,
        })
        return config


class DilatedBottleneckNet(tf.keras.Model):
    def __init__(self, out_channel=256, bottleneck=BottleneckSize, kernel_size=15, dilation=50, groups=Groups):
        super(DilatedBottleneckNet, self).__init__()
        self.layer0 = DilatedBottleneckBlock(out_channel * 2, bottleneck * 2, kernel_size, dilation, 1, firstlayergroups=groups)
        self.layer1 = DilatedBottleneckBlock(out_channel * 2, bottleneck * 2, kernel_size, dilation * 2, groups)
        self.layer2 = DilatedBottleneckBlock(out_channel * 2, bottleneck * 2, kernel_size, dilation * 4, groups)
        self.layer3 = DilatedBottleneckBlock(out_channel * 2, bottleneck * 2, kernel_size, dilation * 8, groups)
        self.layer4 = DilatedBottleneckBlock(out_channel * 2, bottleneck * 2, kernel_size, dilation * 16, groups)
        self.layer5 = DilatedBottleneckBlock(out_channel * 2, bottleneck * 2, kernel_size, dilation * 32, groups)

    def call(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return x5  # return shape (batch, length, embed)

    def get_config(self):
        config = super(DilatedBottleneckNet, self).get_config()
        config.update({
            'out_channel': self.layer0.dilated_conv.filters // 2,
            'bottleneck': self.layer0.bottle.filters // 2,
            'kernel_size': self.layer0.dilated_conv.kernel_size,
            'dilation': self.layer0.dilated_conv.dilation_rate,
            'groups': self.layer0.dilated_conv.groups,
        })
        return config


class CustomAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, bottleneck=BottleneckSize, groups=Groups, rate=0.1, t_len=-1):
        super(CustomAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.scaling = num_heads ** -0.5
        self.t_len = t_len 

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        
        self.DBN = DilatedBottleneckNet(out_channel=d_model, bottleneck=bottleneck, groups=groups)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        self.dr = tf.keras.layers.Dropout(rate)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, self.t_len, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, training, mask):
        q, k = tf.split(self.DBN(x), 2, axis=-1)
        batch_size = tf.shape(q)[0]
        v = self.wv(x)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        q = q * self.scaling
        AttOutWeights = tf.linalg.matmul(q, k, transpose_b=True)
        Mask_repeated = tf.tile(mask[:, None], [1, self.num_heads, 1, 1])
        masked_out_weights = tf.multiply(AttOutWeights, Mask_repeated)
        max_vals = tf.math.reduce_max(masked_out_weights, axis=-1, keepdims=True)
        AttOutputWeights = tf.exp(masked_out_weights - max_vals)
        sum_vals = tf.math.reduce_sum(AttOutputWeights, axis=-1, keepdims=True)
        AttOutputWeights /= sum_vals
        AttOutputWeights = self.dr(AttOutputWeights)
        AttOutput = tf.linalg.matmul(AttOutputWeights, v)
        AttOutput = tf.reshape(tf.transpose(AttOutput, perm=[0, 2, 1, 3]), [-1, self.t_len, self.d_model])
        return AttOutput

    def get_config(self):
        config = super(CustomAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'bottleneck': self.DBN.layer0.bottle.filters // 2,
            'groups': self.DBN.layer0.dilated_conv.groups,
            'rate': self.dr.rate,
            't_len': self.t_len,
        })
        return config


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, bottleneck=BottleneckSize, groups=Groups, rate=0.1, t_len=-1):
        super(EncoderLayer, self).__init__()
        self.catt = CustomAttention(d_model, num_heads, bottleneck=bottleneck, t_len=t_len)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        attn_output = self.catt(x, training, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            'd_model': self.catt.d_model,
            'num_heads': self.catt.num_heads,
            'dff': self.ffn.layers[0].units,
            'bottleneck': self.catt.DBN.layer0.bottle.filters // 2,
            'groups': self.catt.DBN.layer0.dilated_conv.groups,
            'rate': self.dropout1.rate,
            't_len': self.catt.t_len,
        })
        return config


class BDCEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, bottleneck=BottleneckSize, groups=Groups, rate=0.1, t_len=-1):
        super(BDCEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.convembed = tf.keras.layers.Conv1D(filters=d_model, kernel_size=11, strides=1, padding='same', dilation_rate=1)
        self.bn = tf.keras.layers.BatchNormalization()
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, bottleneck=bottleneck, groups=groups, rate=rate, t_len=t_len) for _ in range(num_layers)]
        
    def call(self, x, training, mask):
        x = self.convembed(x)
        x = self.bn(x)
        x = tf.nn.relu(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x

    def get_config(self):
        config = super(BDCEncoder, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'convembed_filters': self.convembed.filters,
            'convembed_kernel_size': self.convembed.kernel_size,
            'convembed_strides': self.convembed.strides,
            'convembed_padding': self.convembed.padding,
            'convembed_dilation_rate': self.convembed.dilation_rate,
            'num_heads': self.enc_layers[0].catt.num_heads,
            'dff': self.enc_layers[0].ffn.layers[0].units,
            'bottleneck': self.enc_layers[0].catt.DBN.layer0.bottle.filters // 2,
            'groups': self.enc_layers[0].catt.DBN.layer0.dilated_conv.groups,
            'rate': self.enc_layers[0].dropout1.rate,
            't_len': self.enc_layers[0].catt.t_len,
        })
        return config
