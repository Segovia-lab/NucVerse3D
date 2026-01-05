# https://youtu.be/L5iV5BHkMzM
"""

Attention U-net:
https://arxiv.org/pdf/1804.03999.pdf

Recurrent residual Unet (R2U-Net) paper
https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
(Check fig 4.)

Note: Batch normalization should be performed over channels after a convolution,
In the following code axis is set to 3 as our inputs are of shape
[None, height, width, channel]. Channel is axis=3.

Original code from below link but heavily modified.
https://github.com/MoleImg/Attention_UNet/blob/master/AttResUNet.py
"""

import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K

from keras.models import Model
from keras.layers import (
    Input,
    Conv3D,
    MaxPooling3D,
    UpSampling3D,
    concatenate,
    Conv3DTranspose,
    BatchNormalization,
    Dropout,
    Lambda,
)
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate


def conv_block_3d(x, filter_size, size, dropout, batch_norm=False):
    conv = layers.Conv3D(size, (filter_size, filter_size, filter_size), padding="same")(
        x
    )
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=-1)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv3D(size, (filter_size, filter_size, filter_size), padding="same")(
        conv
    )
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=-1)(conv)
    conv = layers.Activation("relu")(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv


def repeat_elem_3d(tensor, rep):
    return layers.Lambda(
        lambda x, repnum: K.repeat_elements(x, repnum, axis=-1),
        arguments={"repnum": rep},
    )(tensor)


def res_conv_block_3d(x, filter_size, size, dropout, batch_norm=False):
    conv = layers.Conv3D(size, (filter_size, filter_size, filter_size), padding="same")(
        x
    )
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=-1)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv3D(size, (filter_size, filter_size, filter_size), padding="same")(
        conv
    )
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=-1)(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv3D(size, kernel_size=(1, 1, 1), padding="same")(x)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=-1)(shortcut)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation("relu")(res_path)
    return res_path


def gating_signal_3d(input, out_size, batch_norm=False):
    x = layers.Conv3D(out_size, (1, 1, 1), padding="same")(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def attention_block_3d(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = layers.Conv3D(inter_shape, (2, 2, 2), strides=(2, 2, 2), padding="same")(
        x
    )
    shape_theta_x = K.int_shape(theta_x)

    phi_g = layers.Conv3D(inter_shape, (1, 1, 1), padding="same")(gating)
    upsample_g = layers.Conv3DTranspose(
        inter_shape,
        (3, 3, 3),
        strides=(
            shape_theta_x[1] // shape_g[1],
            shape_theta_x[2] // shape_g[2],
            shape_theta_x[3] // shape_g[3],
        ),
        padding="same",
    )(phi_g)
    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation("relu")(concat_xg)
    psi = layers.Conv3D(1, (1, 1, 1), padding="same")(act_xg)
    sigmoid_xg = layers.Activation("sigmoid")(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling3D(
        size=(
            shape_x[1] // shape_sigmoid[1],
            shape_x[2] // shape_sigmoid[2],
            shape_x[3] // shape_sigmoid[3],
        )
    )(sigmoid_xg)

    upsample_psi = repeat_elem_3d(upsample_psi, shape_x[4])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv3D(shape_x[4], (1, 1, 1), padding="same")(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn


def Attention_ResUNet_3D(input_shape, dropout_rate=0.2, batch_norm=True, FILTER_NUM=16):

    FILTER_SIZE = 3
    UP_SAMP_SIZE = 2

    inputs = layers.Input(input_shape, dtype=tf.float32)
    axis = -1

    conv_128 = res_conv_block_3d(
        inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm
    )
    pool_64 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv_128)
    conv_64 = res_conv_block_3d(
        pool_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm
    )
    pool_32 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv_64)
    conv_32 = res_conv_block_3d(
        pool_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm
    )
    pool_16 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv_32)
    conv_16 = res_conv_block_3d(
        pool_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm
    )
    pool_8 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv_16)

    conv_8 = res_conv_block_3d(
        pool_8, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm
    )

    ##DECODER 1
    gating_16 = gating_signal_3d(conv_8, 8 * FILTER_NUM, batch_norm)
    att_16 = attention_block_3d(conv_16, gating_16, 8 * FILTER_NUM)
    up_16 = layers.UpSampling3D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE, UP_SAMP_SIZE))(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=axis)
    up_conv_16 = res_conv_block_3d(
        up_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm
    )

    gating_32 = gating_signal_3d(up_conv_16, 4 * FILTER_NUM, batch_norm)
    att_32 = attention_block_3d(conv_32, gating_32, 4 * FILTER_NUM)
    up_32 = layers.UpSampling3D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE, UP_SAMP_SIZE))(
        up_conv_16
    )
    up_32 = layers.concatenate([up_32, att_32], axis=axis)
    up_conv_32 = res_conv_block_3d(
        up_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm
    )

    gating_64 = gating_signal_3d(up_conv_32, 2 * FILTER_NUM, batch_norm)
    att_64 = attention_block_3d(conv_64, gating_64, 2 * FILTER_NUM)
    up_64 = layers.UpSampling3D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE, UP_SAMP_SIZE))(
        up_conv_32
    )
    up_64 = layers.concatenate([up_64, att_64], axis=axis)
    up_conv_64 = res_conv_block_3d(
        up_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm
    )

    gating_128 = gating_signal_3d(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block_3d(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling3D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE, UP_SAMP_SIZE))(
        up_conv_64
    )
    up_128 = layers.concatenate([up_128, att_128], axis=axis)
    up_conv_128 = res_conv_block_3d(
        up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm
    )

    conv_final = layers.Conv3D(2, kernel_size=(1, 1, 1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=axis)(conv_final)
    output_1 = layers.Activation("softmax")(conv_final)

    ## DECODER 2
    gating_16_2 = gating_signal_3d(conv_8, 8 * FILTER_NUM, batch_norm)
    att_16_2 = attention_block_3d(conv_16, gating_16_2, 8 * FILTER_NUM)
    up_16_2 = layers.UpSampling3D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE, UP_SAMP_SIZE))(
        conv_8
    )
    up_16_2 = layers.concatenate([up_16_2, att_16_2], axis=axis)
    up_conv_16_2 = res_conv_block_3d(
        up_16_2, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm
    )

    gating_32_2 = gating_signal_3d(up_conv_16_2, 4 * FILTER_NUM, batch_norm)
    att_32_2 = attention_block_3d(conv_32, gating_32_2, 4 * FILTER_NUM)
    up_32_2 = layers.UpSampling3D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE, UP_SAMP_SIZE))(
        up_conv_16_2
    )
    up_32_2 = layers.concatenate([up_32_2, att_32_2], axis=axis)
    up_conv_32_2 = res_conv_block_3d(
        up_32_2, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm
    )

    gating_64_2 = gating_signal_3d(up_conv_32_2, 2 * FILTER_NUM, batch_norm)
    att_64_2 = attention_block_3d(conv_64, gating_64_2, 2 * FILTER_NUM)
    up_64_2 = layers.UpSampling3D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE, UP_SAMP_SIZE))(
        up_conv_32_2
    )
    up_64_2 = layers.concatenate([up_64_2, att_64_2], axis=axis)
    up_conv_64_2 = res_conv_block_3d(
        up_64_2, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm
    )

    gating_128_2 = gating_signal_3d(up_conv_64_2, FILTER_NUM, batch_norm)
    att_128_2 = attention_block_3d(conv_128, gating_128_2, FILTER_NUM)
    up_128_2 = layers.UpSampling3D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE, UP_SAMP_SIZE))(
        up_conv_64_2
    )
    up_128_2 = layers.concatenate([up_128_2, att_128_2], axis=axis)
    up_conv_128_2 = res_conv_block_3d(
        up_128_2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm
    )

    output_2 = layers.Conv3D(3, kernel_size=(1, 1, 1))(up_conv_128_2)

    model = models.Model(inputs, [output_1, output_2], name="Attention_ResUNet_3D")
    return model
