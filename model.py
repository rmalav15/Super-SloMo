from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import keras


def conv2d(batch_input, output_channels, kernel_size=3, stride=1, scope="conv", activation='relu'):
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

def lrelu(input, alpha):
    return keras.layers.LeakyReLU(alpha=alpha)(input)

def average_pool(input, kernel_size, stride=2, scope="avg_pool"):
    return tf.contrib.layers.avg_pool2d(input, [kernel_size, kernel_size], stride, scope=scope)

def bilinear_upsampling(input, scale=2, scope = "bi_upsample"):
    with tf.variable_scope(scope):
        _, h, w, _ = input.get_shape()
        return tf.image.resize_bilinear(input, [scale*h, scale*w])

def UNet(inputs, output_channels, reuse=False, FLAGS=None):
    return None
