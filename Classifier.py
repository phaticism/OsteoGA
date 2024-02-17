import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_CLASSES = 3

import os
import gc
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.applications import (
    InceptionResNetV2, DenseNet201, ResNet152V2,
    EfficientNetV2M, ResNet50V2, Xception,
    InceptionV3, EfficientNetV2S
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Activation, BatchNormalization,
    MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Lambda,
    Dropout, Input, concatenate, add, Conv2DTranspose,
    SpatialDropout2D, Cropping2D, UpSampling2D, LeakyReLU,
    ZeroPadding2D, Reshape, Concatenate, Multiply, Permute, Add, Subtract
)

def adjust_pretrained_weights(model_cls, input_size, name=None):
    weights_model = model_cls(weights='imagenet',
                              include_top=False,
                              input_shape=(*input_size, 3))
    target_model = model_cls(weights=None,
                             include_top=False,
                             input_shape=(*input_size, 1))
    weights = weights_model.get_weights()
    weights[0] = np.sum(weights[0], axis=2, keepdims=True)
    target_model.set_weights(weights)

    del weights_model
    tf.keras.backend.clear_session()
    gc.collect()
    if name:
        target_model._name = name
    return target_model

from keras import backend as K
def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block

    Args:
        input: input tensor
        filters: number of output filters

    Returns: a keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = int(init.shape[channel_axis])
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = Multiply()([init, se])
    return x


def spatial_squeeze_excite_block(input):
    ''' Create a spatial squeeze-excite block

    Args:
        input: input tensor

    Returns: a keras tensor

    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input)

    x = Multiply()([input, se])
    return x


def channel_spatial_squeeze_excite(input, ratio=16):
    ''' Create a spatial squeeze-excite block

    Args:
        input: input tensor
        filters: number of output filters

    Returns: a keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''

    cse = squeeze_excite_block(input, ratio)
    sse = spatial_squeeze_excite_block(input)

    x = Add()([cse, sse])
    return x

def DoubleConv(filters, kernel_size, initializer='glorot_uniform'):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding='same', kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
        x = Conv2D(filters, kernel_size, padding='same', kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
        return x
    return layer

def UpSampling2D_block(filters, kernel_size=(3, 3), upsample_rate=(2, 2), interpolation='bilinear',
                       initializer='glorot_uniform', skip=None):
    def layer(input_tensor):
        x = UpSampling2D(size=upsample_rate, interpolation=interpolation)(input_tensor)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = DoubleConv(filters, kernel_size, initializer=initializer)(x)
        x = channel_spatial_squeeze_excite(x)
        return x
    return layer

def Conv2DTranspose_block(filters, transpose_kernel_size=(3, 3), upsample_rate=(2, 2),
                          initializer='glorot_uniform', skip=None, met_input=None, sat_input=None):
    def layer(input_tensor):
        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate, padding='same')(input_tensor)
        if skip is not None:
            x = Concatenate()([x, skip])

        x = DoubleConv(filters, transpose_kernel_size, initializer=initializer)(x)
        x = channel_spatial_squeeze_excite(x)
        return x
    return layer

def PixelShuffle_block(filters, kernel_size=(3, 3), upsample_rate=2,
                          initializer='glorot_uniform', skip=None, met_input=None, sat_input=None):
    def layer(input_tensor):
        x = Conv2D(filters * (upsample_rate ** 2), kernel_size, padding="same",
                   activation="swish", kernel_initializer='Orthogonal')(input_tensor)
        x = tf.nn.depth_to_space(x, upsample_rate)
        if skip is not None:
            x = Concatenate()([x, skip])

        x = DoubleConv(filters, kernel_size, initializer=initializer)(x)
        x = channel_spatial_squeeze_excite(x)
        return x
    return layer

def get_efficient_unet(name=None,
                       option='full',
                       input_shape=(224, 224, 1),
                       encoder_weights=None,
                       block_type='conv-transpose',
                       output_activation='sigmoid',
                       kernel_initializer='glorot_uniform'):
    if encoder_weights == 'imagenet':
        encoder = adjust_pretrained_weights(EfficientNetV2S, input_shape[:-1], name)
    elif encoder_weights is None:
        encoder = EfficientNetV2S(weights=None,
                                  include_top=False,
                                  input_shape=input_shape)
        encoder._name = name
    else:
        raise ValueError(encoder_weights)

    if option == 'encoder':
        return encoder

    MBConvBlocks = []

    skip_candidates = ['1b', '2d', '3d', '4f']

    for mbblock_nr in skip_candidates:
        mbblock = encoder.get_layer('block{}_add'.format(mbblock_nr)).output
        MBConvBlocks.append(mbblock)

    head = encoder.get_layer('top_activation').output
    blocks = MBConvBlocks + [head]

    if block_type == 'upsampling':
        UpBlock = UpSampling2D_block
    elif block_type == 'conv-transpose':
        UpBlock = Conv2DTranspose_block
    elif block_type == 'pixel-shuffle':
        UpBlock = PixelShuffle_block
    else:
        raise ValueError(block_type)

    o = blocks.pop()
    o = UpBlock(512, initializer=kernel_initializer, skip=blocks.pop())(o)
    o = UpBlock(256, initializer=kernel_initializer, skip=blocks.pop())(o)
    o = UpBlock(128, initializer=kernel_initializer, skip=blocks.pop())(o)
    o = UpBlock(64, initializer=kernel_initializer, skip=blocks.pop())(o)
    o = UpBlock(32, initializer=kernel_initializer, skip=None)(o)
    o = Conv2D(input_shape[-1], (1, 1), padding='same', activation=output_activation, kernel_initializer=kernel_initializer)(o)

    model = Model(encoder.input, o, name=name)

    if option == 'full':
        return model, encoder
    elif option == 'model':
        return model
    else:
        raise ValueError(option)

c = tf.constant([
    [1, 3, 6, 7, 9],
    [4, 1, 4, 5, 7],
    [6, 4, 1, 3, 5],
    [9, 7, 4, 1, 4],
    [11, 9, 7, 5, 1]
], dtype=tf.float32)

def ordinal_loss(y_true, y_pred):
    modified_prob = (1-y_pred)*y_true + (1-y_true)*y_pred
    return tf.reduce_sum(tf.gather(c, tf.argmax(y_true, axis=-1)) * modified_prob, axis=-1)

BACKBONES = {
    'xception': Xception,
    'resnet50v2': ResNet50V2,
    'resnet152v2': ResNet152V2,
    'inceptionv3': InceptionV3,
    'inception-resnetv2': InceptionResNetV2,
    'densenet201': DenseNet201,
    'efficientnetv2m': EfficientNetV2M,
    'efficientnetv2s': EfficientNetV2S
}

def get_by_backbone(backbone='xception',
                    input_shape=(224, 224, 1),
                    backbone_weights=None,
                    backbone_trainable=True,
                    errors_input_shape=None,
                    dropout_rate=0.0,
                    output_regularizers=None,
                    name=None):
    input_tensor = Input(shape=input_shape)
    if backbone_weights == 'imagenet':
        if input_shape[-1] != 3:
            backbone_layer = adjust_pretrained_weights(BACKBONES[backbone], input_shape[:-1])
        else:
            backbone_layer = BACKBONES[backbone](weights=backbone_weights,
                                                 include_top=False,
                                                 input_shape=input_shape)
    elif backbone_weights is None:
        backbone_layer = BACKBONES[backbone](weights=backbone_weights,
                                             include_top=False,
                                             input_shape=input_shape)
    else:
        backbone_layer = BACKBONES[backbone](weights=None,
                                             include_top=False,
                                             input_shape=input_shape)
        backbone_layer.set_weights(backbone_weights)

    if not backbone_trainable:
        backbone_layer.trainable = False
    backbone_tensor = backbone_layer(input_tensor)
    gpooling = GlobalAveragePooling2D(name='gpooling')(backbone_tensor)
    if dropout_rate > 0.:
        gpooling = Dropout(dropout_rate)(gpooling)
    else:
        inputs = [input_tensor]
        output = Dense(NUM_CLASSES,
                       activation = 'softmax',
                       kernel_regularizer=output_regularizers)(gpooling)

    model = Model(inputs=inputs, outputs=output, name=name)
    return model

main_model = get_by_backbone(backbone='efficientnetv2s',
                             backbone_weights='imagenet',
                             backbone_trainable=True,
                             input_shape=(224, 224, 1),
                             name='main')
backbone = main_model.get_layer('efficientnetv2-s')

input_shape = (224, 224, 1)
