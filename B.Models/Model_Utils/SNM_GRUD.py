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
from tensorflow.keras import layers, initializers, regularizers, constraints, activations
from tensorflow.keras import backend as K


__all__ = [
    "GRUDCell",
    "GRUD",
    "GRUDInput",
    "GRUDState",
]

CONSTANT_INIT = initializers.Constant(0.05)
IMPUTATION_TYPES = ["miwae", "decay", "forward", "zero", "raw"]

GRUDInput = namedtuple("GRUDInput", ["values", "mask", "times"])
GRUDState = namedtuple("GRUDState", ["h", "x_keep", "s_prev"])


def exp_relu(x):
    return K.exp(-K.relu(x))

def exp_softplus(x):
    return K.exp(-K.softplus(x))


class GRUDCell(layers.Layer):
    def __init__(
        self,
        units,
        x_imputation="decay",
        input_decay="exp_relu",
        hidden_decay="exp_relu",
        masking_decay=None,
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        use_bias=True,
        use_decay_bias=True,
        feed_masking=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        decay_initializer=CONSTANT_INIT,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        decay_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        decay_constraint=None,
        dropout=0.,
        recurrent_dropout=0.,
        reset_after=False,
        **kwargs,
    ):

        if units < 0:
            raise ValueError(
                f"Received an invalid value for argument `units`, expected a positive integer, got {units}."
            )

        if x_imputation not in IMPUTATION_TYPES:
            raise ValueError(f"Unsupported imputation method: {x_imputation}")

        if kwargs.pop("implementation", 1) != 1:
            raise ValueError(f"GRUDCell only supports implementation=1")

        if reset_after is True:  # TODO: Check if this is correct.
            raise ValueError("GRUDCell only support reset_after=False")

        # NOTE: Disable caching device for GRUDCell to avoid potential errors.
        # By default use cached variable under v2 mode, see b/143699808.
        # if tf.compat.v1.executing_eagerly_outside_functions():
        #     self._enable_caching_device = kwargs.pop("enable_caching_device", True)
        # else:
        #     self._enable_caching_device = kwargs.pop("enable_caching_device", False)

        super().__init__(**kwargs)

        self.x_imputation = x_imputation
        self.use_decay_bias = use_decay_bias
        self.feed_masking = feed_masking

        self.units = units
        self.decay_initializer = initializers.get(decay_initializer)
        self.decay_regularizer = regularizers.get(decay_regularizer)
        self.decay_constraint = constraints.get(decay_constraint)

        if x_imputation not in ["decay", "miwae"]:
            input_decay = None

        with tf.keras.utils.custom_object_scope({"exp_relu": exp_relu, "exp_softplus": exp_softplus}):
            self.input_decay = None if input_decay is None else activations.get(input_decay)
            self.hidden_decay = None if hidden_decay is None else activations.get(hidden_decay)

            if self.feed_masking:
                self.masking_decay = None if masking_decay is None else activations.get(masking_decay)
                self._masking_dropout_mask = None
            else:
                self.masking_decay = None

        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))

        self.reset_after = False
        self.output_size = self.units

        self._input_dim = None

    @property
    def state_size(self):
        return (self.units, self._input_dim, self._input_dim)

    def build(self, input_shape):
        super().build(input_shape)

        self._input_dim = input_dim = input_shape[0][-1]

        # NOTE: Disable caching device for GRUDCell to avoid potential errors.
        # default_caching_device = rnn_utils.caching_device(self)

        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            # caching_device=default_caching_device,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            # caching_device=default_caching_device,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(3 * self.units,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                # caching_device=default_caching_device,
            )
        else:
            self.bias = None

        if self.input_decay is not None:
            self.input_decay_kernel = self.add_weight(
                shape=(input_dim,),
                name="input_decay_kernel",
                initializer=self.decay_initializer,
                regularizer=self.decay_regularizer,
                constraint=self.decay_constraint,
                # caching_device=default_caching_device,
            )
            if self.use_decay_bias:
                self.input_decay_bias = self.add_weight(
                    shape=(input_dim,),
                    name="input_decay_bias",
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    # caching_device=default_caching_device,
                )
            else:
                self.input_decay_bias = None
        else:
            self.input_decay_kernel = None

        if self.hidden_decay is not None:
            self.hidden_decay_kernel = self.add_weight(
                shape=(self._input_dim, self.units),
                name="hidden_decay_kernel",
                initializer=self.decay_initializer,
                regularizer=self.decay_regularizer,
                constraint=self.decay_constraint,
                # caching_device=default_caching_device,
            )
            if self.use_decay_bias:
                self.hidden_decay_bias = self.add_weight(
                    shape=(self.units,),
                    name="hidden_decay_bias",
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    # caching_device=default_caching_device,
                )
            else:
                self.hidden_decay_bias = None
        else:
            self.hidden_decay_kernel = None

        if self.feed_masking:
            self.masking_kernel = self.add_weight(
                shape=(self._input_dim, self.units * 3),
                name="masking_kernel",
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                # caching_device=default_caching_device,
            )
            if self.masking_decay is not None:
                self.masking_decay_kernel = self.add_weight(
                    shape=(self._input_dim,),
                    name="masking_decay_kernel",
                    initializer=self.decay_initializer,
                    regularizer=self.decay_regularizer,
                    constraint=self.decay_constraint,
                    # caching_device=default_caching_device,
                )
                if self.use_decay_bias:
                    self.masking_decay_bias = self.add_weight(
                        shape=(self._input_dim,),
                        name="masking_decay_bias",
                        initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint,
                        # caching_device=default_caching_device,
                    )
                else:
                    self.masking_decay_bias = None
            else:
                self.masking_decay_kernel = None
        else:
            self.masking_kernel = None

        self.built = True

        
    def get_dropout_mask_for_cell(self, inputs, training, count=3):
        if training:
            masks = []
            for i in range(count):
                masks.append(tf.nn.dropout(tf.ones_like(inputs[i]), self.dropout))
            return masks
        else:
            return [tf.ones_like(inputs[i]) for i in range(count)]


    def get_recurrent_dropout_mask_for_cell(self, h_tm1, training, count=1):
        if training:
            return [tf.nn.dropout(tf.ones_like(h_tm1), self.recurrent_dropout) for _ in range(count)]
        else:
            return [tf.ones_like(h_tm1) for _ in range(count)]

    
    def call(self, inputs, states, training=None):
        input_x, input_m, input_t = inputs
        h_tm1, x_keep_tm1, t_prev_tm1 = states

        delta_t = input_t - t_prev_tm1

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=3)

        if self.feed_masking:
            if 0. < self.dropout < 1. and self._masking_dropout_mask is None:
                self._masking_dropout_mask = self._create_dropout_mask(
                    tf.ones_like(input_m, dtype=tf.float32), training=training, count=3,
                )

            m_dp_mask = self._masking_dropout_mask
        
        if self.input_decay is not None:
            gamma_di = delta_t * self.input_decay_kernel
            if self.use_decay_bias:
                gamma_di = K.bias_add(gamma_di, self.input_decay_bias)
            gamma_di = self.input_decay(gamma_di)

        if self.hidden_decay is not None:
            gamma_dh = K.dot(delta_t, self.hidden_decay_kernel)
            if self.use_decay_bias:
                gamma_dh = K.bias_add(gamma_dh, self.hidden_decay_bias)
            gamma_dh = self.hidden_decay(gamma_dh)

        if self.feed_masking and self.masking_decay is not None:
            gamma_dm = delta_t * self.masking_decay_kernel
            if self.use_decay_bias:
                gamma_dm = K.bias_add(gamma_dm, self.masking_decay_bias)
            gamma_dm = self.masking_decay(gamma_dm)

        # weighted sum between decayed last observation and grown 0 (empirical mean)
        x_keep_t = tf.where(input_m > 0.5, input_x, x_keep_tm1)
        x_t = tf.where(input_m > 0.5, input_x, gamma_di * x_keep_t)

        if self.hidden_decay is not None:
            h_tm1d = gamma_dh * h_tm1
        else:
            h_tm1d = h_tm1

        if self.feed_masking:
            m_t = 1. - tf.cast(input_m, tf.float32)
            if self.masking_decay is not None:
                m_t = gamma_dm * m_t

        x_z, x_r, x_h = x_t, x_t, x_t

        if 0. < self.dropout < 1.:
            x_z *= dp_mask[0]
            x_r *= dp_mask[1]
            x_h *= dp_mask[2]

            if self.feed_masking:
                m_z, m_r, m_h = m_t * m_dp_mask[0], m_t * m_dp_mask[1], m_t * m_dp_mask[2]
        else:
            if self.feed_masking:
                m_z, m_r, m_h = m_t, m_t, m_t

        h_tm1_z, h_tm1_r = h_tm1d, h_tm1d

        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z *= rec_dp_mask[0]
            h_tm1_r *= rec_dp_mask[1]

        z_t = K.dot(x_z, self.kernel[:, : self.units])
        r_t = K.dot(x_r, self.kernel[:, self.units : self.units * 2])
        hh_t = K.dot(x_h, self.kernel[:, self.units * 2 :])

        z_t = z_t + K.dot(h_tm1_z, self.recurrent_kernel[:, : self.units])
        r_t = r_t + K.dot(h_tm1_r, self.recurrent_kernel[:, self.units : self.units * 2])

        if self.feed_masking:
            z_t += K.dot(m_z, self.masking_kernel[:, : self.units])
            r_t += K.dot(m_r, self.masking_kernel[:, self.units : self.units * 2])
            hh_t += K.dot(m_h, self.masking_kernel[:, self.units * 2 :])

        if self.use_bias:
            z_t = K.bias_add(z_t, self.bias[: self.units])
            r_t = K.bias_add(r_t, self.bias[self.units : self.units * 2])
            hh_t = K.bias_add(hh_t, self.bias[self.units * 2 :])

        z_t = self.recurrent_activation(z_t)
        r_t = self.recurrent_activation(r_t)

        if 0. < self.recurrent_dropout < 1.:
            h_tm1_h = r_t * h_tm1d * rec_dp_mask[2]
        else:
            h_tm1_h = r_t * h_tm1d

        hh_t = self.activation(hh_t + K.dot(h_tm1_h, self.recurrent_kernel[:, self.units * 2 :]))
        h_t = z_t * h_tm1 + (1 - z_t) * hh_t
        t_prev_t = tf.where(input_m > 0.5, K.tile(input_t, [1, self.state_size[-1]]), t_prev_tm1)
        
        return h_t, (h_t, x_keep_t, t_prev_t)

    def reset_masking_dropout_mask(self):
        self._masking_dropout_mask = None

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        dtype = dtype or self.dtype
        if inputs is None:
            return (
                tf.zeros([batch_size, self.units], dtype=dtype),
                tf.zeros([batch_size, self._input_dim], dtype=dtype),
                tf.zeros([batch_size, self._input_dim], dtype=dtype)
            )
        else:
            if self.go_backwards:
                initial_t = tf.reduce_max(inputs[2], axis=1, keepdims=True)
            else:
                initial_t = tf.expand_dims(inputs[2][:, 0], axis=1)

            input_dim = inputs[0].shape[-1]

            return (
                tf.zeros([batch_size, self.units], dtype=dtype),
                tf.zeros([batch_size, input_dim], dtype=dtype),
                tf.tile(initial_t, [1, input_dim])
            )


class GRUD(layers.RNN):
    def __init__(
        self,
        units,
        x_imputation="decay",
        input_decay="exp_relu",
        hidden_decay="exp_relu",
        masking_decay=None,
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        use_bias=True,
        use_decay_bias=True,
        feed_masking=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        decay_initializer=CONSTANT_INIT,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        decay_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        decay_constraint=None,
        dropout=0.,
        recurrent_dropout=0.,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        time_major=False,
        reset_after=False,
        **kwargs,
    ):

        cell = GRUDCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            reset_after=reset_after,
            x_imputation=x_imputation,
            input_decay=input_decay,
            hidden_decay=hidden_decay,
            use_decay_bias=use_decay_bias,
            feed_masking=feed_masking,
            masking_decay=masking_decay,
            decay_initializer=decay_initializer,
            decay_regularizer=decay_regularizer,
            decay_constraint=decay_constraint,
            dtype=kwargs.get("dtype"),
            trainable=kwargs.get("trainable", True),
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            time_major=time_major,
            **kwargs,
        )
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        def reset_dropout_mask():
            self.cell.dropout_mask = None
            
        def reset_recurrent_dropout_mask():
            self.cell.recurrent_dropout_mask = None
            
        def reset_masking_dropout_mask():
            self.cell.masking_dropout_mask = None

        reset_dropout_mask()
        reset_recurrent_dropout_mask()
        reset_masking_dropout_mask()
        
        # The input should be dense, padded with zeros. If a ragged input is fed
        # into the layer, it is padded and the row lengths are used for masking.
        if isinstance(inputs, tf.RaggedTensor):
            is_ragged_input = True
            inputs, row_lengths = inputs.to_tensor(), inputs.row_lengths()
        else:
            is_ragged_input = False
            row_lengths = None

        self._validate_args_if_ragged(is_ragged_input, mask)
        
        # GRU does not support constants. Ignore it during process.
        inputs, initial_state, _ = self._process_inputs(
            inputs, initial_state, None
        )
        
        if isinstance(mask, list):
            mask = mask[0]
        input_shape = K.int_shape(inputs[0])
        timesteps = input_shape[0] if self.time_major else input_shape[1]

        kwargs = {"training": training}
        self._maybe_reset_cell_dropout_mask(self.cell)

        def step(cell_inputs, cell_states):
            return self.cell(cell_inputs, cell_states, **kwargs)

        last_output, outputs, states = K.rnn(
            step,
            inputs,
            initial_state,
            constants=None,
            go_backwards=self.go_backwards,
            mask=mask,
            unroll=self.unroll,
            input_length=(row_lengths if row_lengths is not None else timesteps),
            time_major=self.time_major,
            zero_output_for_mask=self.zero_output_for_mask,
            # return_all_outputs=self.return_sequences,
        )
        if self.stateful:
            updates = [self.states[0].assign(states[0])]
            self.add_update(updates)
        if self.return_sequences:
            if is_ragged_input:
                output = tf.RaggedTensor.from_tensor(outputs, lengths=row_lengths)
            else:
                output = outputs
        else:
            output = last_output
        if self.return_state:
            return [output] + list(states)
        else:
            return output
