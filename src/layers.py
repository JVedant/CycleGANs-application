from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
    Activation
)
from config import *

# “Ck denotes a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2"
def CK(inputs, k, use_instancenorm=True):
    block = Conv2D(
        k,
        kernel_size=(4, 4),
        strides=2,
        padding="same",
        kernel_initializer=weight_initializer,
    )(inputs)
    if use_instancenorm:
        block = InstanceNormalization(axis=-1)(block)
    block = LeakyReLU(0.2)(block)
    return block


# "c7s1-k denotes a 7×7 Convolution-InstanceNorm-ReLU with k filters and stride 1"
def c7s1k(inputs, k, activation):
    block = Conv2D(
        k,
        kernel_size=(7, 7),
        strides=1,
        padding="same",
        kernel_initializer=weight_initializer,
    )(inputs)
    block = InstanceNormalization(axis=-1)(block)
    block = Activation(activation)(block)
    return block


# "dk denotes a 3×3 Convolution-InstanceNorm-ReLU with k filters and stride 2"
def dk(inputs, k):
    block = Conv2D(
        k,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        kernel_initializer=weight_initializer,
    )(inputs)
    block = InstanceNormalization(axis=-1)(block)
    block = Activation(tf.nn.relu)(block)
    return block


# "Rk denotes a residual block that contains two 3×3 convolutional layers with k filters on each layer"
def rk(inputs, k):
    block = Conv2D(
        k,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        kernel_initializer=weight_initializer,
    )(inputs)
    block = InstanceNormalization(axis=-1)(block)
    block = Activation(tf.nn.relu)(block)
    block = Conv2D(
        k,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        kernel_initializer=weight_initializer,
    )(inputs)
    block = InstanceNormalization(axis=-1)(block)
    return block + inputs


# "uk denotes a 3×3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k filters and stride ½" (Conv2d with 0.5 strides is Conv2dTranspose with 2 strides)
def uk(inputs, k):
    block = Conv2DTranspose(
        k,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        kernel_initializer=weight_initializer,
    )(inputs)
    block = InstanceNormalization(axis=-1)(block)
    block = Activation(tf.nn.relu)(block)
    return block
