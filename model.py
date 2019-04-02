from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import keras
from utils import vgg_19, bilinear_sampler


# Loss Helper Functions
def VGG19_slim(input, type, reuse=False):
    # Define the feature to extract according to the type of perceptual
    if type == 'VGG54':
        target_layer = 'vgg_19/conv5/conv5_4'
    elif type == 'VGG22':
        target_layer = 'vgg_19/conv2/conv2_2'
    else:
        raise NotImplementedError('Unknown perceptual type')
    _, output = vgg_19(input, reuse=reuse)
    output = output[target_layer]

    return output


def l1_loss(Ipred, Iref, axis=[3]):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(Ipred - Iref), axis=axis))  # L1 Norm


def l2_loss(Ipred, Iref, axis=[3]):
    return tf.reduce_mean(tf.reduce_sum(tf.square(Ipred - Iref), axis=axis))  # L2 Norm


def reconstruction_loss(Ipred, Iref):
    Ipred = tf.image.convert_image_dtype(Ipred, dtype=tf.uint8)
    Iref = tf.image.convert_image_dtype(Iref, dtype=tf.uint8)

    Ipred = tf.cast(Ipred, dtype=tf.float32)
    Iref = tf.cast(Iref, dtype=tf.float32)  # TODO: Check whether required

    # tf.reduce_mean(tf.norm(tf.math.subtract(Ipred, Iref), ord=1, axis=[3]))
    return l1_loss(Ipred, Iref)


def perceptual_loss(Ipred, Iref, layers="VGG54", scope="perceptual_loss"):
    with tf.variable_scope(scope):
        # Note name scope is ignored in varibale naming (scope)
        with tf.name_scope("vgg19_Ipred"):
            Ipred_features = VGG19_slim(Ipred, layers, reuse=False)
        with tf.name_scope("vgg19_Iref"):
            Iref_features = VGG19_slim(Iref, layers, reuse=True)

        return l2_loss(Ipred_features, Iref_features)


# Optical flow range must be [-1, 1]
def wrapping_loss(frame0, frame1, frameT, F01, F10, Fdasht0, Fdasht1):
    return l1_loss(frame0, bilinear_sampler(frame1, F01)) + \
           l1_loss(frame1, bilinear_sampler(frame0, F10)) + \
           l1_loss(frameT, bilinear_sampler(frame0, Fdasht0)) + \
           l1_loss(frameT, bilinear_sampler(frame1, Fdasht1))


def smoothness_loss(F01, F10):
    deltaF01 = tf.reduce_mean(tf.abs(F01[:, 1:, :, :] - F01[:, :-1, :, :])) + tf.reduce_mean(
        tf.abs(F01[:, :, 1:, :] - F01[:, :, :-1, :]))
    deltaF10 = tf.reduce_mean(tf.abs(F10[:, 1:, :, :] - F10[:, :-1, :, :])) + tf.reduce_mean(
        tf.abs(F10[:, :, 1:, :] - F10[:, :, :-1, :]))
    return 0.5 * (deltaF01 + deltaF10)


# Model Helper Functions
def conv2d(batch_input, output_channels, kernel_size=3, stride=1, scope="conv", activation=None):
    with tf.variable_scope(scope):
        activation_fn = None
        if activation == 'leaky_relu':
            activation_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.2)
        elif activation == 'relu':
            activation_fn = tf.nn.relu

        return slim.conv2d(batch_input, output_channels, [kernel_size, kernel_size], stride=stride,
                           data_format='NHWC',
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           activation_fn=activation_fn)


def lrelu_keras(input, alpha=0.2):
    return keras.layers.LeakyReLU(alpha=alpha)(input)  # TODO: Pure tensorflow, Keras Not Allowed


def lrelu(input, alpha=0.2):
    return tf.nn.leaky_relu(input, alpha=alpha)


def average_pool(input, kernel_size, stride=2, scope="avg_pool"):
    return tf.contrib.layers.avg_pool2d(input, [kernel_size, kernel_size], stride, scope=scope)


def bilinear_upsampling(input, scale=2, scope="bi_upsample"):
    with tf.variable_scope(scope):
        _, h, w, _ = input.get_shape()
        return tf.image.resize_bilinear(input, [scale * h, scale * w])


def encoder_block(inputs, output_channel, conv_kernel=3, pool_kernel=2, lrelu_alpha=0.1, scope="enc_block"):
    with tf.variable_scope(scope):
        net = conv2d(inputs, output_channel, kernel_size=conv_kernel)
        conv = lrelu(net, lrelu_alpha)
        pool = average_pool(conv, pool_kernel)
        return conv, pool


def decoder_block(input, skip_conn_input, output_channel, conv_kernel=3, up_scale=2, lrelu_alpha=0.1,
                  scope="dec_block"):
    with tf.variable_scope(scope):
        upsample = bilinear_upsampling(input, scale=up_scale)

        block_input = tf.concat([upsample, skip_conn_input], 3)
        _, _, _, upsample_channels = upsample.get_shape()
        _, _, _, skip_conn_channels = skip_conn_input.get_shape()
        _, _, _, total_channels = block_input.get_shape()
        tf.debugging.assert_equal(upsample_channels + skip_conn_channels, total_channels)  # TODO: Remove This Part

        net = conv2d(block_input, output_channel, kernel_size=conv_kernel)
        net = lrelu(net, lrelu_alpha)
        return net


def UNet(inputs, output_channels, first_kernel=7, second_kernel=5, scope='unet', output_activation=None, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("encoder"):
            econv1, epool1 = encoder_block(inputs, 32, conv_kernel=first_kernel, scope="en_conv1")
            econv2, epool2 = encoder_block(epool1, 64, conv_kernel=second_kernel, scope="en_conv2")
            econv3, epool3 = encoder_block(epool2, 128, scope="en_conv3")
            econv4, epool4 = encoder_block(epool3, 256, scope="en_conv4")
            econv5, epool5 = encoder_block(epool4, 512, scope="en_conv5")
            with tf.variable_scope("en_conv6"):
                econv6 = conv2d(epool5, 512)
                econv6 = lrelu(econv6, alpha=0.1)

        with tf.variable_scope("decoder"):
            net = decoder_block(econv6, econv5, 512, scope="dec_conv1")
            net = decoder_block(net, econv4, 256, scope="dec_conv2")
            net = decoder_block(net, econv3, 128, scope="dec_conv3")
            net = decoder_block(net, econv2, 64, scope="dec_conv4")
            net = decoder_block(net, econv1, 32, scope="dec_conv5")

        with tf.variable_scope("unet_output"):
            net = conv2d(net, output_channels, scope="output")
            if output_activation is not None:
                if output_activation == "tanh":
                    net = tf.nn.tanh(net)
                elif output_activation == "lrelu":
                    net = lrelu(net, alpha=0.1)
                else:
                    raise ValueError("only lrelu|tanh allowed")
            return net, econv6
