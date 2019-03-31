from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import keras


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


def lrelu(input, alpha=0.2):
    return keras.layers.LeakyReLU(alpha=alpha)(input)


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

        block_input = tf.concat([input, skip_conn_input], 3)
        _, _, _, inp_channels = input.get_shape()
        _, _, _, skip_conn_channels = skip_conn_input.get_shape()
        _, _, _, total_channels = block_input.get_shape()
        tf.debugging.assert_equal(inp_channels + skip_conn_channels, total_channels)  # TODO: Remove This Part




def UNet(inputs, output_channels, scope='unet', reuse=False, FLAGS=None):
    # Check the flag
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for UNet')

    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("encoder"):
            econv1, epool1 = encoder_block(inputs, 32, conv_kernel=7, scope="en_conv1")
            econv2, epool2 = encoder_block(epool1, 64, conv_kernel=5, scope="en_conv2")
            econv3, epool3 = encoder_block(epool2, 128, scope="en_conv3")
            econv4, epool4 = encoder_block(epool3, 256, scope="en_conv4")
            econv5, epool5 = encoder_block(epool4, 512, scope="en_conv5")
            with tf.variable_scope("en_conv6"):
                econv6 = conv2d(epool5, 512)
                econv6 = lrelu(econv6, alpha=0.1)

        with tf.variable_scope("decoder"):

    return None
