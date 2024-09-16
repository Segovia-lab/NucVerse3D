from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate

def conv_block(input, num_filters):
    x = Conv3D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv3D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)
    return x

def conv_block_simple(input, num_filters):
    x = Conv3D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)
    return x

#Encoder block: Conv block followed by maxpooling

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPooling3D((2, 2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = Conv3DTranspose(num_filters, (2, 2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_unet_2_decoder(input_shape, n_classes_1, n_classes_2, filters = 16):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, filters)
    s2, p2 = encoder_block(p1, filters*2**1)
    s3, p3 = encoder_block(p2, filters*2**2)
    s4, p4 = encoder_block(p3, filters*2**3)

    b1 = conv_block(p4, filters*2**4) #Bridge

    d1_1 = decoder_block(b1, s4, filters*2**3)
    d1_2 = decoder_block(d1_1, s3, filters*2**2)
    d1_3 = decoder_block(d1_2, s2, filters*2**1)
    d1_4 = decoder_block(d1_3, s1, filters)
    
    # pre_output_1 = conv_block_simple(d1_4, 32)
    
    if n_classes_1 == 1:  #Binary
      activation = 'sigmoid'
    else:
      activation = 'softmax'
    
    pre_output_1 = conv_block_simple(d1_4, 32)
    output_1 = Conv3D(n_classes_1, 1, padding="same", activation=activation)(pre_output_1)  #Change the activation based on n_classes

    d2_1 = decoder_block(b1, s4, filters*2**3)
    d2_2 = decoder_block(d2_1, s3, filters*2**2)
    d2_3 = decoder_block(d2_2, s2, filters*2**1)
    d2_4 = decoder_block(d2_3, s1, filters)
    
    pre_output_2 = conv_block_simple(d2_4, 32)
    output_2 = Conv3D(3, 1, padding="same")(pre_output_2)
    model = Model(inputs, [output_1, output_2], name="U-Net")
    return model

# Build unet 1 decoder split last conv
def build_unet_last(input_shape, n_classes_1, n_classes_2, filters = 16):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, filters)
    s2, p2 = encoder_block(p1, filters*2**1)
    s3, p3 = encoder_block(p2, filters*2**2)
    s4, p4 = encoder_block(p3, filters*2**3)

    b1 = conv_block(p4, filters*2**4) #Bridge

    d1_1 = decoder_block(b1, s4, filters*2**3)
    d1_2 = decoder_block(d1_1, s3, filters*2**2)
    d1_3 = decoder_block(d1_2, s2, filters*2**1)
    d1_4 = decoder_block(d1_3, s1, filters)
    
    pre_output_1 = conv_block_simple(d1_4, 32)
    
    if n_classes_1 == 1:  #Binary
      activation = 'sigmoid'
    else:
      activation = 'softmax'
     
    output_1 = Conv3D(n_classes_1, 1, padding="same", activation=activation)(pre_output_1)  #Change the activation based on n_classes


    pre_output_2 = conv_block_simple(d1_4, 32)
    output_2 = Conv3D(3, 1, padding="same")(pre_output_2)
    model = Model(inputs, [output_1, output_2], name="U-Net")
    return model