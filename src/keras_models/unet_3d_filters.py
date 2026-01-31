# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:22:50 2024

@author: WORKSTATION 2
"""

# Building Unet by dividing encoder and decoder into blocks
# This is exactly the same unet used in earlier exercises for 2D images.
# Except here we change conv2D to conv3D, Maxpooling2D to Maxpooling3D
# upsampling2D to upsampling3D and Conv2DTranspose to Conv3DTranspose.
# We also change the filter size and pooling sizes from 3x3 and 2x2 to
# 3x3x3 and 2x2x2, respectively.

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


def conv_block(input, num_filters):
    x = Conv3D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)  # Not in the original network.
    x = Activation("relu")(x)

    x = Conv3D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  # Not in the original network
    x = Activation("relu")(x)
    return x


# Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPooling3D((2, 2, 2))(x)
    return x, p


# Decoder block
# skip features gets input from encoder for concatenation


def decoder_block(input, skip_features, num_filters):
    x = Conv3DTranspose(num_filters, (2, 2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


# Build Unet using the blocks
def build_unet(input_shape, n_classes, n_filters=16):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, n_filters * 2 ** 0)
    s2, p2 = encoder_block(p1, n_filters * 2 ** 1)
    s3, p3 = encoder_block(p2, n_filters * 2 ** 2)
    s4, p4 = encoder_block(p3, n_filters * 2 ** 3)

    b1 = conv_block(p4, n_filters * 2 ** 4)  # Bridge

    d1 = decoder_block(b1, s4, n_filters * 2 ** 3)
    d2 = decoder_block(d1, s3, n_filters * 2 ** 2)
    d3 = decoder_block(d2, s2, n_filters * 2 ** 1)
    d4 = decoder_block(d3, s1, n_filters * 2 ** 0)

    if n_classes == 1:  # Binary
        activation = "sigmoid"
    else:
        activation = "softmax"

    outputs = Conv3D(n_classes, 1, padding="same", activation=activation)(
        d4
    )  # Change the activation based on n_classes
    print(activation)

    model = Model(inputs, outputs, name="U-Net")
    return model
