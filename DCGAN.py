import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gc
import numpy as np
from tqdm import tqdm
from skimage import exposure
import streamlit as st
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['savefig.pad_inches'] = 0
matplotlib.use('Agg')

import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.__internal__.layers import BaseRandomLayer

from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Activation, BatchNormalization,
    MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,
    Dropout, Input, concatenate, add, Conv2DTranspose, Lambda,
    SpatialDropout2D, Cropping2D, UpSampling2D, LeakyReLU,
    ZeroPadding2D, Reshape, Concatenate, Multiply, Permute, Add
)

from tensorflow.keras.applications import (
    InceptionResNetV2, DenseNet201, ResNet152V2, VGG19,
    EfficientNetV2M, ResNet50V2, Xception, InceptionV3,
    EfficientNetV2S, EfficientNetV2B3, ResNet50, ConvNeXtBase,
    RegNetX032
)

st.set_option('deprecation.showPyplotGlobalUse', False)

import ultralytics
from ultralytics import YOLO

IMAGE_SIZE = 224
NUM_CLASSES = 3

yolo_weight = './weights_yolo/oai_s_best4.pt'
seg_model = YOLO(yolo_weight)

def normalize_tuple(value, n, name, allow_zero=False):
    """Transforms non-negative/positive integer/integers into an integer tuple.
    Args:
      value: The value to validate and convert. Could an int, or any iterable of
        ints.
      n: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.
      allow_zero: Default to False. A ValueError will raised if zero is received
        and this param is False.
    Returns:
      A tuple of n integers.
    Raises:
      ValueError: If something else than an int/long or iterable thereof or a
      negative value is
        passed.
    """
    error_msg = (
        f"The `{name}` argument must be a tuple of {n} "
        f"integers. Received: {value}"
    )

    if isinstance(value, int):
        value_tuple = (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError(error_msg)
        if len(value_tuple) != n:
            raise ValueError(error_msg)
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                error_msg += (
                    f"including element {single_value} of "
                    f"type {type(single_value)}"
                )
                raise ValueError(error_msg)

    if allow_zero:
        unqualified_values = {v for v in value_tuple if v < 0}
        req_msg = ">= 0"
    else:
        unqualified_values = {v for v in value_tuple if v <= 0}
        req_msg = "> 0"

    if unqualified_values:
        error_msg += (
            f" including {unqualified_values}"
            f" that does not satisfy the requirement `{req_msg}`."
        )
        raise ValueError(error_msg)

    return value_tuple


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


class DropBlockNoise(BaseRandomLayer):
    def __init__(
        self,
        rate,
        block_size,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"rate must be a number between 0 and 1. " f"Received: {rate}"
            )

        self._rate = rate
        (
            self._dropblock_height,
            self._dropblock_width,
        ) = normalize_tuple(
            value=block_size, n=2, name="block_size", allow_zero=False
        )
        self.seed = seed

    def call(self, x, training=None):
        if not training or self._rate == 0.0:
            return x

        _, height, width, _ = tf.split(tf.shape(x), 4)

        # Unnest scalar values
        height = tf.squeeze(height)
        width = tf.squeeze(width)

        dropblock_height = tf.math.minimum(self._dropblock_height, height)
        dropblock_width = tf.math.minimum(self._dropblock_width, width)

        gamma = (
            self._rate
            * tf.cast(width * height, dtype=tf.float32)
            / tf.cast(dropblock_height * dropblock_width, dtype=tf.float32)
            / tf.cast(
                (width - self._dropblock_width + 1)
                * (height - self._dropblock_height + 1),
                tf.float32,
            )
        )

        # Forces the block to be inside the feature map.
        w_i, h_i = tf.meshgrid(tf.range(width), tf.range(height))
        valid_block = tf.logical_and(
            tf.logical_and(
                w_i >= int(dropblock_width // 2),
                w_i < width - (dropblock_width - 1) // 2,
            ),
            tf.logical_and(
                h_i >= int(dropblock_height // 2),
                h_i < width - (dropblock_height - 1) // 2,
            ),
        )

        valid_block = tf.reshape(valid_block, [1, height, width, 1])

        random_noise = self._random_generator.random_uniform(
            tf.shape(x), dtype=tf.float32
        )
        valid_block = tf.cast(valid_block, dtype=tf.float32)
        seed_keep_rate = tf.cast(1 - gamma, dtype=tf.float32)
        block_pattern = (1 - valid_block + seed_keep_rate + random_noise) >= 1
        block_pattern = tf.cast(block_pattern, dtype=tf.float32)

        window_size = [1, self._dropblock_height, self._dropblock_width, 1]

        # Double negative and max_pool is essentially min_pooling
        block_pattern = -tf.nn.max_pool(
            -block_pattern,
            ksize=window_size,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        return (
            x * tf.cast(block_pattern, x.dtype)
        )


def get_efficient_unet(name=None,
                       option='full',
                       input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
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


def acc(y_true, y_pred, threshold=0.5):
    threshold = tf.cast(threshold, y_pred.dtype)
    y_pred = tf.cast(y_pred > threshold, y_pred.dtype)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true-y_pred))


def inv_ssim(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def inv_msssim(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0, filter_size=4))


def inv_msssim_l1(y_true, y_pred, alpha=0.8):
    return alpha*inv_msssim(y_true, y_pred) + (1-alpha)*mae(y_true, y_pred)


# def inv_msssim_gaussian_l1(y_true, y_pred, alpha=0.8):
#     l1_diff = tf.abs(y_true-y_pred)
#     gaussian_l1 = tfa.image.gaussian_filter2d(l1_diff, filter_shape=(11, 11), sigma=1.5)
#     return alpha*inv_msssim(y_true, y_pred) + (1-alpha)*gaussian_l1


def psnr(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1.0))


class MultipleTrackers():
    def __init__(self, callback_lists: list):
        self.callbacks_list = callback_lists

    def __getattr__(self, attr):
        def helper(*arg, **kwarg):
            for cb in self.callbacks_list:
                getattr(cb, attr)(*arg, **kwarg)
        if attr in self.__class__.__dict__:
            return getattr(self, attr)
        else:
            return helper


class DCGAN():
    def __init__(self,
                 input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                 architecture='two-stage',
                 pretrain_weights=None,
                 output_activation='sigmoid',
                 block_type='conv-transpose',
                 kernel_initializer='glorot_uniform',
                 noise=None,
                 C=1.):

        self.C = C
        # Build
        kwargs = dict(input_shape=input_shape,
                      output_activation=output_activation,
                      encoder_weights=pretrain_weights,
                      block_type=block_type,
                      kernel_initializer=kernel_initializer)

        if architecture == 'two-stage':
            encoder = get_efficient_unet(name='dcgan_disc',
                                         option='encoder',
                                         **kwargs)

            self.generator = get_efficient_unet(name='dcgan_gen', option='model', **kwargs)
        elif architecture == 'shared':

            self.generator, encoder = get_efficient_unet(name='dcgan', option='full', **kwargs)
        else:
            raise ValueError(f'Unsupport architecture: {architecture}')

        gpooling = GlobalAveragePooling2D()(encoder.output)
        prediction = Dense(1, activation='sigmoid')(gpooling)
        self.discriminator = Model(encoder.input, prediction, name='dcgan_disc')

        tf.keras.backend.clear_session()
        _ = gc.collect()

        if noise:
            gen_inputs = self.generator.input
            corrupted_inputs = noise(gen_inputs)
            outputs = self.generator(corrupted_inputs)
            self.generator = Model(gen_inputs, outputs, name='dcgan_gen')

            tf.keras.backend.clear_session()
            _ = gc.collect()

        if output_activation == 'tanh':

            self.process_input = layers.Lambda(lambda img: (img*2.-1.), name='dcgan_normalize')
            self.process_output = layers.Lambda(lambda img: (img*0.5+0.5), name='dcgan_denormalize')
            gen_inputs = self.generator.input
            process_inputs = self.process_input(gen_inputs)
            process_inputs = self.generator(process_inputs)
            gen_outputs = self.process_output(process_inputs)
            self.generator = Model(gen_inputs, gen_outputs, name='dcgan_gen')

            disc_inputs = self.discriminator.input
            process_inputs = self.process_input(disc_inputs)
            disc_outputs = self.discriminator(process_inputs)
            self.discriminator = Model(disc_inputs, disc_outputs, name='dcgan_disc')

            tf.keras.backend.clear_session()
            _ = gc.collect()

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()

    def compile(self,
                generator_optimizer=Adam(5e-4, 0.5),
                discriminator_optimizer=Adam(5e-4),
                reconstruction_loss=mae,
                discriminative_loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                reconstruction_metrics=[],
                discriminative_metrics=[]):

        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator.compile(optimizer=self.discriminator_optimizer)

        self.generator_optimizer = generator_optimizer
        self.generator.compile(optimizer=self.generator_optimizer)

        self.loss = discriminative_loss
        self.reconstruction_loss = reconstruction_loss
        self.d_loss_tracker = tf.keras.metrics.Mean()
        self.g_loss_tracker = tf.keras.metrics.Mean()
        self.g_recon_tracker = tf.keras.metrics.Mean()
        self.g_disc_tracker = tf.keras.metrics.Mean()

        self.g_metric_trackers = [(tf.keras.metrics.Mean(), metric) for metric in reconstruction_metrics]
        self.d_metric_trackers = [(tf.keras.metrics.Mean(), tf.keras.metrics.Mean(), tf.keras.metrics.Mean(), metric) for metric in discriminative_metrics]

        all_trackers = [self.d_loss_tracker, self.g_loss_tracker, self.g_recon_tracker, self.g_disc_tracker] + \
                       [tracker for tracker,_ in self.g_metric_trackers] + \
                       [tracker for t in self.d_metric_trackers for tracker in t[:-1]]
        self.all_trackers = MultipleTrackers(all_trackers)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        total_loss = 0.5*(real_loss + fake_loss)
        return total_loss

    def generator_loss(self, fake_output):
        return self.loss(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        masked, original = images
        n_samples = tf.shape(original)[0]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(masked, training=True)

            real_output = self.discriminator(original, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_disc_loss = self.generator_loss(fake_output)
            recon_loss = self.reconstruction_loss(original, generated_images)
            gen_loss = self.C*recon_loss + gen_disc_loss
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        self.d_loss_tracker.update_state(tf.repeat([[disc_loss]], repeats=n_samples, axis=0))
        self.g_loss_tracker.update_state(tf.repeat([[gen_loss]], repeats=n_samples, axis=0))
        self.g_recon_tracker.update_state(tf.repeat([[recon_loss]], repeats=n_samples, axis=0))
        self.g_disc_tracker.update_state(tf.repeat([[gen_disc_loss]], repeats=n_samples, axis=0))

        logs = {'d_loss': self.d_loss_tracker.result()}

        for tracker, real_tracker, fake_tracker, metric in self.d_metric_trackers:
            v_real = metric(tf.ones_like(real_output), real_output)
            v_fake = metric(tf.zeros_like(fake_output), fake_output)
            v = 0.5*(v_real + v_fake)
            tracker.update_state(tf.repeat([[v]], repeats=n_samples, axis=0))
            real_tracker.update_state(tf.repeat([[v_real]], repeats=n_samples, axis=0))
            fake_tracker.update_state(tf.repeat([[v_fake]], repeats=n_samples, axis=0))

            metric_name = metric.__name__
            logs['d_' + metric_name] = tracker.result()
            logs['d_real_' + metric_name] = real_tracker.result()
            logs['d_fake_' + metric_name] = fake_tracker.result()

        logs['g_loss'] = self.g_loss_tracker.result()
        logs['g_recon'] = self.g_recon_tracker.result()
        logs['g_disc'] = self.g_disc_tracker.result()

        for tracker, metric in self.g_metric_trackers:
            v = metric(original, generated_images)
            tracker.update_state(tf.repeat([[v]], repeats=n_samples, axis=0))
            logs['g_' + metric.__name__] = tracker.result()

        return logs

    @tf.function
    def val_step(self, images):
        masked, original = images
        n_samples = tf.shape(original)[0]

        generated_images = self.generator(masked, training=False)

        real_output = self.discriminator(original, training=False)
        fake_output = self.discriminator(generated_images, training=False)

        gen_disc_loss = self.generator_loss(fake_output)
        recon_loss = self.reconstruction_loss(original, generated_images)
        gen_loss = self.C*recon_loss + gen_disc_loss
        disc_loss = self.discriminator_loss(real_output, fake_output)

        self.d_loss_tracker.update_state(tf.repeat([[disc_loss]], repeats=n_samples, axis=0))
        self.g_loss_tracker.update_state(tf.repeat([[gen_loss]], repeats=n_samples, axis=0))
        self.g_recon_tracker.update_state(tf.repeat([[recon_loss]], repeats=n_samples, axis=0))
        self.g_disc_tracker.update_state(tf.repeat([[gen_disc_loss]], repeats=n_samples, axis=0))

        logs = {'val_d_loss': self.d_loss_tracker.result()}

        for tracker, real_tracker, fake_tracker, metric in self.d_metric_trackers:
            v_real = metric(tf.ones_like(real_output), real_output)
            v_fake = metric(tf.zeros_like(fake_output), fake_output)
            v = 0.5*(v_real + v_fake)
            tracker.update_state(tf.repeat([[v]], repeats=n_samples, axis=0))
            real_tracker.update_state(tf.repeat([[v_real]], repeats=n_samples, axis=0))
            fake_tracker.update_state(tf.repeat([[v_fake]], repeats=n_samples, axis=0))

            metric_name = metric.__name__
            logs['val_d_' + metric_name] = tracker.result()
            logs['val_d_real_' + metric_name] = real_tracker.result()
            logs['val_d_fake_' + metric_name] = fake_tracker.result()

        logs['val_g_loss'] = self.g_loss_tracker.result()
        logs['val_g_recon'] = self.g_recon_tracker.result()
        logs['val_g_disc'] = self.g_disc_tracker.result()

        for tracker, metric in self.g_metric_trackers:
            v = metric(original, generated_images)
            tracker.update_state(tf.repeat([[v]], repeats=n_samples, axis=0))
            logs['val_g_' + metric.__name__] = tracker.result()

        return logs

    def fit(self,
            trainset,
            valset=None,
            trainsize=-1,
            valsize=-1,
            epochs=1,
            display_per_epochs=5,
            generator_callbacks=[],
            discriminator_callbacks=[]):

        print('🌊🐉 Start Training 🐉🌊')
        gen_callback_tracker = tf.keras.callbacks.CallbackList(
            generator_callbacks, add_history=True, model=self.generator
        )

        disc_callback_tracker = tf.keras.callbacks.CallbackList(
            discriminator_callbacks, add_history=True, model=self.discriminator
        )

        callbacks_tracker = MultipleTrackers([gen_callback_tracker, disc_callback_tracker])

        logs = {}
        callbacks_tracker.on_train_begin(logs=logs)

        for epoch in range(epochs):
            print(f'Epochs {epoch+1}/{epochs}:')
            callbacks_tracker.on_epoch_begin(epoch, logs=logs)

            batches = tqdm(trainset,
                           desc="Train",
                           total=trainsize,
                           unit="step",
                           position=0,
                           leave=True)

            for batch, image_batch in enumerate(batches):

                callbacks_tracker.on_batch_begin(batch, logs=logs)
                callbacks_tracker.on_train_batch_begin(batch, logs=logs)

                train_logs = {k:v.numpy() for k, v in self.train_step(image_batch).items()}
                logs.update(train_logs)

                callbacks_tracker.on_train_batch_end(batch, logs=logs)
                callbacks_tracker.on_batch_end(batch, logs=logs)
                batches.set_postfix({'d_loss': train_logs['d_loss'],
                                     'g_loss': train_logs['g_loss']
                                    })

                # Presentation
            stats = ", ".join("{}={:.3g}".format(k, v) for k, v in logs.items() if 'val_' not in k and 'loss' not in k)
            print('Train:', stats)

            batches.close()
            if valset:
                self.all_trackers.reset_state()

                batches = tqdm(valset,
                               desc="Valid",
                               total=valsize,
                               unit="step",
                               position=0,
                               leave=True)

                for batch, image_batch in enumerate(batches):
                    callbacks_tracker.on_batch_begin(batch, logs=logs)
                    callbacks_tracker.on_test_batch_begin(batch, logs=logs)
                    val_logs = {k:v.numpy() for k, v in self.val_step(image_batch).items()}
                    logs.update(val_logs)

                    callbacks_tracker.on_test_batch_end(batch, logs=logs)
                    callbacks_tracker.on_batch_end(batch, logs=logs)
                    # Presentation
                    batches.set_postfix({'val_d_loss': val_logs['val_d_loss'],
                                         'val_g_loss': val_logs['val_g_loss']
                                        })

                stats = ", ".join("{}={:.3g}".format(k, v) for k, v in logs.items() if 'val_' in k and 'loss' not in k)
                print('Valid:', stats)

                batches.close()

            if epoch % display_per_epochs == 0:
                print('-'*128)
                self.visualize_samples((image_batch[0][:2], image_batch[1][:2]))

            self.all_trackers.reset_state()

            callbacks_tracker.on_epoch_end(epoch, logs=logs)
#             tf.keras.backend.clear_session()
            _ = gc.collect()

            if self.generator.stop_training or self.discriminator.stop_training:
                break
            print('-'*128)

        callbacks_tracker.on_train_end(logs=logs)
        tf.keras.backend.clear_session()
        _ = gc.collect()
        gen_history = None
        for cb in gen_callback_tracker:
            if isinstance(cb, tf.keras.callbacks.History):
                gen_history = cb
                gen_history.history = {k:v for k,v in cb.history.items() if 'd_' not in k}

        disc_history = None
        for cb in disc_callback_tracker:
            if isinstance(cb, tf.keras.callbacks.History):
                disc_history = cb
                disc_history.history = {k:v for k,v in cb.history.items() if 'g_' not in k}

        return {'generator':gen_history,
                'discriminator':disc_history}

    def visualize_samples(self, samples, figsize=(12, 2)):
        x, y = samples
        y_pred = self.generator.predict(x[:2], verbose=0)
        fig, axs = plt.subplots(1, 6, figsize=figsize)
        for i in range(2):
            pos = 3*i
            axs[pos].imshow(x[i], cmap='gray', vmin=0., vmax=1.)
            axs[pos].set_title('Masked')
            axs[pos].axis('off')
            axs[pos+1].imshow(y[i], cmap='gray', vmin=0., vmax=1.)
            axs[pos+1].set_title('Original')
            axs[pos+1].axis('off')
            axs[pos+2].imshow(y_pred[i], cmap='gray', vmin=0., vmax=1.)
            axs[pos+2].set_title('Predicted')
            axs[pos+2].axis('off')
        plt.show()

#         tf.keras.backend.clear_session()
        del y_pred
        _ = gc.collect()