import numpy as np

from collections import namedtuple
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers, initializers, regularizers, constraints, activations
from tensorflow.keras import backend as K

Length = 60
OrigDim = 50

CONSTANT_INIT = initializers.Constant(0.05)
InterpolateInput = namedtuple("InterpolateInput", ["values", "mask", "times"])
InterpolateState = namedtuple("InterpolateState", ["x_keep", "t_keep"])


def replace_column(tensor, replacement, column_idx):
    left = tensor[:, :column_idx]
    right = tensor[:, column_idx+1:]
    new_tensor = tf.concat([left, tf.expand_dims(replacement, axis=1), right], axis=1)
    return new_tensor

def compute_last_observed_and_mean(x, mask):
    """
    Args:
    - x: A tensor of shape [batch_size, time_len, feature_dim]
    - mask: A tensor of shape [batch_size, time_len, feature_dim]
    
    Returns:
    - last_observed: A tensor representing the last observed value for each feature over time.
    - mean: A tensor representing the mean value for each feature over time.
    """

    # x shape: [batch_size, time_len, feature_dim]
    # mask shape: [batch_size, time_len, feature_dim]
    # print('clom-1',x)
    # print('clom-2',mask)

    # Initialize last_observed tensor with zeros
    last_observed = tf.zeros_like(x)
    # last_observed shape: [batch_size, time_len, feature_dim]
    # print('clom-3',last_observed)

    # Initialize last_values tensor with zeros
    dynamic_shape = tf.shape(x)
    last_values = tf.zeros([dynamic_shape[0], OrigDim])

    # last_values shape: [batch_size, feature_dim]
    # print('clom-4',last_values)

    # Iteratively compute the last observed values
    for t in range(x.shape[1]):
        last_values = mask[:, t] * x[:, t] + (1 - mask[:, t]) * last_values
        last_observed = replace_column(last_observed, last_values, t)
    # last_observed shape: [batch_size, time_len, feature_dim]
    # print('clom-5',last_observed)

    # Compute mean across the time axis and account for the mask
    mean = tf.math.reduce_sum(x, axis=1) / tf.math.reduce_sum(mask, axis=1)
    # print('clom-6',mean)
    # mean shape: [batch_size, feature_dim]

    mean = tf.expand_dims(mean, axis=1)
    # mean shape: [batch_size, 1, feature_dim]
    # print('clom-7',mean)

    mean = tf.tile(mean, [1, x.shape[1], 1])
    # mean shape: [batch_size, time_len, feature_dim]
    # print('clom-8',mean)

    return last_observed, mean

def compute_delta_t(s):
    """
    Compute the time differences between consecutive missing values based on the mask s.
    
    Args:
    - s: A tensor of shape [batch_size, time_len, feature_dim] containing binary values (1 for observed and 0 for missing).
    
    Returns:
    A tensor of shape [batch_size, time_len, feature_dim] containing time differences between consecutive missing values.
    """
    # s shape: [batch_size, time_len, feature_dim]

    # Find positions of missing values
    missing_positions = tf.where(s == 0)
    # missing_positions shape: [num_missing, 3] where num_missing is the number of zeros in s

    # Compute cumulative sum of s and use difference to get distances
    cumsum_s = tf.cumsum(s, axis=1)
    # cumsum_s shape: [batch_size, time_len, feature_dim]

    zeros_tensor = tf.zeros((tf.shape(s)[0], 1, tf.shape(s)[2]), dtype=s.dtype)
    # zeros_tensor shape: [batch_size, 1, feature_dim]

    distances = tf.concat([zeros_tensor, cumsum_s[:, :-1]], axis=1)
    # distances shape after concat: [batch_size, time_len, feature_dim]

    distances = cumsum_s - distances
    # distances shape: [batch_size, time_len, feature_dim]

    # Create a delta_t tensor initialized with distances for missing values
    delta_t = tf.where(s == 0, distances, 0)
    # delta_t shape: [batch_size, time_len, feature_dim]

    # For initial missing values, fill with their index + 1
    initial_missing = tf.math.cumprod(s, axis=1) == 0
    # initial_missing shape: [batch_size, time_len, feature_dim]

    arange_tensor = tf.reshape(tf.range(1, tf.shape(s)[1] + 1, dtype=s.dtype), (1, -1, 1))
    # arange_tensor shape after reshape: [1, time_len, 1]

    arange_tensor = tf.broadcast_to(arange_tensor, tf.shape(s))
    # arange_tensor shape after broadcast: [batch_size, time_len, feature_dim]

    delta_t = tf.where(initial_missing, arange_tensor, delta_t)
    # delta_t shape: [batch_size, time_len, feature_dim]
    
    return delta_t


def exp_relu(x):
    return K.exp(-K.relu(x))

def exp_softplus(x):
    return K.exp(-K.softplus(x))


class DecayCell(layers.AbstractRNNCell):
    def __init__(
        self,
        use_bias=True,
        decay="exp_softplus",
        decay_initializer="zeros",
        bias_initializer=CONSTANT_INIT,
        decay_regularizer=None,
        bias_regularizer=None,
        decay_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.use_bias = use_bias
        with tf.keras.utils.custom_object_scope({"exp_relu": exp_relu, "exp_softplus": exp_softplus}):
            self.decay = None if decay is None else activations.get(decay)

        self.decay_initializer = initializers.get(decay_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.decay_regularizer = regularizers.get(decay_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.decay_constraint = constraints.get(decay_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        self._input_dim = input_shape[0][-1]

        self.decay_kernel = self.add_weight(
            shape=(self._input_dim,),
            name="decay_kernel",
            initializer=self.decay_initializer,
            regularizer=self.decay_regularizer,
            constraint=self.decay_constraint,
        )
        if self.use_bias:
            self.decay_bias = self.add_weight(
                shape=(self._input_dim,),
                name="decay_bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )

        self.built = True

    def call(self, inputs, states, training=None):
        # print('dc-1',inputs)  # [None, feature_dim], [None, feature_dim], [None, 1]
        # print('dc-2',states)  # [None, feature_dim], [None, feature_dim]
        x_input, m_input, t_input = inputs
        x_last, t_last = states
        t_delta = t_input - t_last
        # print('dc-3',t_delta)  # [None, feature_dim]

        gamma_di = t_delta * self.decay_kernel
        # print('dc-4',gamma_di)  # [None, feature_dim]
        if self.use_bias:
            gamma_di = K.bias_add(gamma_di, self.decay_bias)
        gamma_di = self.decay(gamma_di)
        # print('dc-5',gamma_di)  # [None, feature_dim]

        m_input = tf.cast(m_input, tf.bool)
        # print('m',m_input)
        # print('x',x_input)
        # print('g',gamma_di)
        # print('x_l',x_last)
        x_t    = tf.where(m_input, x_input, gamma_di * x_last + (1 - gamma_di) * x_input)
        # print('dc-6',x_t)  # [None, feature_dim]
        # x_t    = tf.where(m_input, x_input, gamma_di * x_last)
        x_keep = tf.where(m_input, x_input, x_last)
        # print('dc-7',x_keep)  # [None, feature_dim]
        t_keep = tf.where(m_input, t_input, t_last)
        # print('dc-8',t_keep)  # [None, feature_dim]
        return x_t, InterpolateState(x_keep, t_keep)

    @property
    def state_size(self):
        return InterpolateState(x_keep=self._input_dim, t_keep=self._input_dim)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is None:
            return InterpolateState(
                tf.zeros((batch_size, self._input_dim), dtype=dtype),
                tf.zeros((batch_size, self._input_dim), dtype=dtype),
            )
        else:
            return InterpolateState(
                tf.zeros((batch_size, self._input_dim), dtype=dtype),
                inputs.times[:, 0, :],
            )


class ForwardCell(layers.AbstractRNNCell):
    def build(self, input_shape):
        self._input_dim = input_shape[0][-1]
        self.built = True

    def call(self, inputs, states, training=None):
        x_input, m_input, t_input = inputs
        x_last, t_last = states

        x_keep = tf.where(m_input, x_input, x_last)
        t_keep = tf.where(m_input, t_input, t_last)
        state = InterpolateState(x_keep, t_keep)
        return state, state

    def state_size(self):
        return InterpolateState(x_keep=self._input_dim, t_keep=self._input_dim)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is None:
            return InterpolateState(
                tf.zeros((batch_size, self._input_dim), dtype=dtype),
                tf.zeros((batch_size, self._input_dim), dtype=dtype),
            )
        else:
            return InterpolateState(
                tf.zeros((batch_size, self._input_dim), dtype=dtype),
                tf.reduce_max(inputs.times, axis=1) if self.go_backwards else inputs.times[:, 0, :],
            )


class LinearScan(layers.RNN):
    def __init__(
        self,
        unroll=False,
        go_backwards=False,
        **kwargs,
    ):

        cell = ForwardCell(
            dtype=kwargs.get("dtype"),
            trainable=kwargs.get("trainable", False),
        )

        super().__init__(
            cell,
            return_sequences=True,
            return_state=False,
            go_backwards=go_backwards,
            stateful=False,
            unroll=unroll,
            **kwargs,
        )


class DecayInterpolate(layers.RNN):
    def __init__(
        self,
        use_bias=True,
        decay="exp_relu",
        decay_initializer="zeros",
        bias_initializer=CONSTANT_INIT,
        decay_regularizer=None,
        bias_regularizer=None,
        decay_constraint=None,
        bias_constraint=None,
        unroll=False,
        **kwargs,
    ):

        cell = DecayCell(
            use_bias=use_bias,
            decay=decay,
            decay_initializer=decay_initializer,
            decay_regularizer=decay_regularizer,
            decay_constraint=decay_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            dtype=kwargs.get("dtype"),
            trainable=kwargs.get("trainable", True),
        )

        super().__init__(
            cell,
            return_sequences=True,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=unroll,
            **kwargs,
        )


class LinearInterpolate(layers.Layer):
    def __init__(
        self,
        unroll=False,
        **kwargs,
    ):
        super().__init__()

        self.forward_scan  = LinearScan(unroll=unroll, go_backwards=False, **kwargs)
        self.backward_scan = LinearScan(unroll=unroll, go_backwards=True,  **kwargs)

    def call(self, inputs, mask=None, training=None):
        forwards  = self.forward_scan(inputs, mask=mask, training=training)
        backwards = self.backward_scan(inputs, mask=mask, training=training)

        x_t, t = inputs.values, inputs.times
        x_last, t_last = forwards.x_keep, forwards.t_keep
        x_next = tf.reverse(backwards.x_keep, axis=[1])
        t_next = tf.reverse(backwards.t_keep, axis=[1])

        # Linear interpolation. See https://en.wikipedia.org/wiki/Linear_interpolation
        x_itp = (x_last * (t_next - t) + x_next * (t - t_last)) / (t_next - t_last)
        x_itp = tf.where(tf.math.is_nan(x_itp), 0., x_itp)
        x_itp = tf.where(tf.math.is_inf(x_itp), 0., x_itp)

        return tf.where(inputs.mask, x_t, x_itp)
