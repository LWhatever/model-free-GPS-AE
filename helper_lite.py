import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,Layer,Flatten

def p_norm(p, x, fun=lambda x: tf.square(tf.abs(x))):
    return tf.reduce_sum(p * fun(x))

def logBase(x,base):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator

def log2(x):
    return logBase(x,2)

def Complex_Layer(x):
    return tf.complex(x[:,0],x[:,1])

# %%
def dB2lin(dB,dBtype):
    if dBtype == 'db' or dBtype == 'dB':
        fact = 0.0
    elif dBtype == 'dbm' or dBtype == 'dBm':
        fact = -30.0
    elif dBtype == 'dbu' or dBtype == 'dBu':
        fact = -60.0
    else:
        raise ValueError('dBtype can only be dB, dBm or dBu.')

    fact = tf.constant(fact,dB.dtype)
    ten = tf.constant(10,dB.dtype)

    return ten**( (dB+fact)/ten )


# %%
def _encoder(layer, nHidden, nLayers, activation, nOutput=2, kernel_initializer='glorot_uniform', name='encoder'):
    for i in range(1,nLayers):
        layer_name = name+str(i)
        layer = Dense(nHidden, activation=activation, kernel_initializer=kernel_initializer, name=layer_name)(layer)

    layer_name = name+'_out'
    layer = Dense(nOutput, name=layer_name)(layer)

    return layer

class zeroUpsampling(Layer):
    def __init__(self, up_factor=2, **kwargs):
        self.up_factor = up_factor
        self.is_placeholder = True
        super(zeroUpsampling, self).__init__(**kwargs)
    
    def build(self, input_shape=None):
        super(zeroUpsampling, self).build(input_shape)
    
    def call(self, inputs):
        # inputs = inputs[:,:,tf.newaxis]
        zero_padding = [tf.zeros_like(inputs)]*(self.up_factor-1)
        inputs_US = tf.concat([inputs]+zero_padding, axis=-1)
        inputs_US = Flatten()(inputs_US)
        return inputs_US[:,:,tf.newaxis]
    def get_config(self):
        config = {"up_factor":self.up_factor}
        base_config = super(zeroUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def GMIcal(x_in, y_in, QAMorder, norm_constellation, coding_bits, Px, epsilon=1e-12):
    norm_constellation = norm_constellation.ravel()
    BitsPerSymbol = int(np.log2(QAMorder))
    data_emit_bin = coding_bits

    datalength = len(x_in)
    data_originate_bin = np.zeros((datalength, BitsPerSymbol))

    # Decode transmitted data
    rxdatasym = np.argmin(np.abs(x_in[:,np.newaxis] - norm_constellation), axis=1)

    data_originate_bin = coding_bits[rxdatasym]

    # Calculate noise power
    noise_power = np.sum(np.abs(y_in - x_in) ** 2) / len(y_in)

    GMI = 0  # Initialize GMI

    b = np.sum(
        (1 / np.sqrt(np.pi * noise_power))
        * np.exp(-np.abs(y_in[:, np.newaxis] - norm_constellation) ** 2 / noise_power)
        * Px,
        axis=1, keepdims = True
    )
    b[b==0] = epsilon
    a = np.sum(
    np.expand_dims((1 / np.sqrt(np.pi * noise_power))
    * np.exp(-np.abs(y_in[:, np.newaxis] - norm_constellation) ** 2 / noise_power)
    * Px, axis = -1)
    *(data_originate_bin[:, np.newaxis] == data_emit_bin),
    axis=1,
    )
    a_div_b = a / b
    a_div_b[a_div_b==0] = epsilon
    GMI = np.sum(np.log2(a_div_b))
    GMI_out = -np.sum(Px * np.log2(Px)) + GMI / datalength
    return GMI_out


## tensorflow version
def GMIcal_tf(x_in, y_in, QAMorder, norm_constellation, coding_bits, Px, epsilon=1e-12):
    dtype = Px.dtype
    norm_constellation = tf.reshape(norm_constellation, [-1])
    BitsPerSymbol = int(np.log2(QAMorder))
    data_emit_bin = coding_bits
    datalength = tf.shape(x_in)[0]
    data_originate_bin = tf.zeros((datalength, BitsPerSymbol))

    # Decode transmitted data
    rxdatasym = tf.argmin(tf.abs(tf.expand_dims(x_in, axis=1) - norm_constellation), axis=1)
    data_originate_bin = tf.gather(coding_bits, rxdatasym)

    # Calculate noise power
    noise_power = tf.reduce_sum(tf.abs(y_in - x_in) ** 2) / tf.cast(tf.shape(y_in)[0], dtype=dtype)

    GMI = tf.constant(0.0)  # Initialize GMI

    b = tf.reduce_sum(
        (1 / tf.sqrt(np.pi * noise_power)) *
        tf.exp(-tf.abs(tf.expand_dims(y_in, axis=1) - norm_constellation) ** 2 / noise_power) *
        Px,
        axis=1,
        keepdims=True
    )
    b = tf.where(tf.equal(b, 0), epsilon, b)

    a = tf.reduce_sum(
        tf.expand_dims((1 / tf.sqrt(np.pi * noise_power)) *
        tf.exp(-tf.abs(tf.expand_dims(y_in, axis=1) - norm_constellation) ** 2 / noise_power) *
        Px, axis=-1) *
        tf.cast(tf.equal(tf.expand_dims(data_originate_bin, axis=1), tf.expand_dims(data_emit_bin, axis=0)), dtype=dtype),
        axis=1,
    )

    a_div_b = a / b
    a_div_b = tf.where(tf.equal(a_div_b, 0), epsilon, a_div_b)

    log2_base = tf.cast(tf.math.log(2.0), dtype=dtype)
    GMI = tf.reduce_sum(tf.math.log(a_div_b))/log2_base
    GMI_out = -tf.reduce_sum(Px * tf.math.log(Px)/log2_base) + GMI / tf.cast(datalength, dtype=dtype)
    
    return GMI_out