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
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate


def conv_block(input, num_filters):
    x = Conv3D(num_filters, 3, padding='same')(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv3D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)
    return x

def build_res_unet(input_shape, n_classes_1 = 1, n_classes_2 = 1):
    inputs = Input(input_shape)

    conv1 = conv_block(inputs, 32)
    conc1 = concatenate([inputs, conv1], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc1)

    conv2 = conv_block(pool1, 64)
    conc2 = concatenate([pool1, conv2], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc2)

    conv3 = conv_block(pool2, 128)
    conc3 = concatenate([pool2, conv3], axis=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc3)

    conv4 = conv_block(pool3, 256)
    conc4 = concatenate([pool3, conv4], axis=4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc4)
    ###
    conv5 = conv_block(pool4, 512)
    conc5 = concatenate([pool4, conv5], axis=4)
    ###
    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc5), conv4], axis=4)
    conv6 = conv_block(up6, 256)
    conc6 = concatenate([up6, conv6], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc6), conv3], axis=4)
    conv7 = conv_block(up7, 128)
    conc7 = concatenate([up7, conv7], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc7), conv2], axis=4)
    conv8 = conv_block(up8, 64)
    conc8 = concatenate([up8, conv8], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc8), conv1], axis=4)
    conv9 = conv_block(up9, 32)
    conc9 = concatenate([up9, conv9], axis=4)

    if n_classes_1 == 1:  #Binary
      activation = 'sigmoid'
    else:
      activation = 'softmax'


    output_1 = Conv3D(n_classes_1, 1, padding="same", activation=activation)(conc9)

    up6_2 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc5), conv4], axis=4)
    conv6_2 = conv_block(up6_2, 256)
    conc6_2 = concatenate([up6_2, conv6_2], axis=4)

    up7_2 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc6_2), conv3], axis=4)
    conv7_2 = conv_block(up7_2, 128)
    conc7_2 = concatenate([up7_2, conv7_2], axis=4)

    up8_2 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc7_2), conv2], axis=4)
    conv8_2 = conv_block(up8_2, 64)
    conc8_2 = concatenate([up8_2, conv8_2], axis=4)

    up9_2 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc8_2), conv1], axis=4)
    conv9_2 = conv_block(up9_2, 32)
    conc9_2 = concatenate([up9_2, conv9_2], axis=4)

    output_2 = Conv3D(1, 1, padding="same", activation='sigmoid')(conc9_2)

    model = Model(inputs=inputs, outputs=[output_1, output_2])

    return model
