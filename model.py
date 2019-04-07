from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import vgg_19, flow_back_wrap
import collections


# Loss Helper Functions
def VGG19_slim(input, type, reuse=tf.AUTO_REUSE):
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


def perceptual_loss(Ipred, Iref, layers="VGG54"):
    # Note name scope is ignored in varibale naming (scope)
    with tf.name_scope("vgg19_Ipred"):
        Ipred_features = VGG19_slim(Ipred, layers, reuse=tf.AUTO_REUSE)
    with tf.name_scope("vgg19_Iref"):
        Iref_features = VGG19_slim(Iref, layers, reuse=tf.AUTO_REUSE)

    return l2_loss(Ipred_features, Iref_features)


# Optical flow range must be [-1, 1]
def wrapping_loss(frame0, frame1, frameT, F01, F10, Fdasht0, Fdasht1):
    return l1_loss(frame0, flow_back_wrap(frame1, F01)) + \
           l1_loss(frame1, flow_back_wrap(frame0, F10)) + \
           l1_loss(frameT, flow_back_wrap(frame0, Fdasht0)) + \
           l1_loss(frameT, flow_back_wrap(frame1, Fdasht1))


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


# def lrelu_keras(input, alpha=0.2):
#     return keras.layers.LeakyReLU(alpha=alpha)(input)  # TODO: Pure tensorflow, Keras Not Allowed


def lrelu(input, alpha=0.2):
    return tf.nn.leaky_relu(input, alpha=alpha)


def average_pool(input, kernel_size, stride=2, scope="avg_pool"):
    return tf.contrib.layers.avg_pool2d(input, [kernel_size, kernel_size], stride, scope=scope)


def bilinear_upsampling(input, scale=2, scope="bi_upsample"):
    with tf.variable_scope(scope):
        shape = tf.shape(input)
        h, w = shape[1], shape[2]
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

        upsample_shape = tf.shape(upsample)  # get_shape() - Static, Tf.shape() = dynamic
        skip_conn_shape = tf.shape(skip_conn_input)

        # upsample shape can differ from skip conn input (becouse of avg-pool and then bi-upsample in case of odd shape)
        xdiff, ydiff = skip_conn_shape[1] - upsample_shape[1], skip_conn_shape[2] - upsample_shape[2]
        upsample = tf.pad(upsample, tf.convert_to_tensor([[0, 0], [0, xdiff], [0, ydiff], [0, 0]], dtype=tf.int32))
        block_input = tf.concat([upsample, skip_conn_input], 3)

        net = conv2d(block_input, output_channel, kernel_size=conv_kernel)
        net = lrelu(net, lrelu_alpha)
        return net


def UNet(inputs, output_channels, decoder_extra_input=None, first_kernel=7, second_kernel=5, scope='unet',
         output_activation=None, reuse=False):
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
            decoder_input = econv6
            if decoder_extra_input is not None:
                decoder_input = tf.concat([decoder_input, decoder_extra_input], axis=3)
            net = decoder_block(decoder_input, econv5, 512, scope="dec_conv1")
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


# SloMo vanila model
def SloMo_model(frame0, frame1, frameT, FLAGS, reuse=False, timestamp=0.5):
    # Define the container of the parameter
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    Network = collections.namedtuple('Network', 'total_loss, reconstruction_loss, perceptual_loss, \
                                                wrapping_loss,  smoothness_loss, pred_frameT   \
                                                Ft0, Ft1, Vt0,\
                                                grads_and_vars, train, global_step, learning_rate')
    with tf.variable_scope("SloMo_model", reuse=reuse):
        with tf.variable_scope("flow_computation"):
            flow_comp_input = tf.concat([frame0, frame1], axis=3)
            flow_comp_out, flow_comp_enc_out = UNet(flow_comp_input,
                                                    output_channels=4,  # 2 channel for each flow
                                                    first_kernel=FLAGS.first_kernel,
                                                    second_kernel=FLAGS.second_kernel)
            flow_comp_out = lrelu(flow_comp_out)
            F01, F10 = flow_comp_out[:, :, :, :2], flow_comp_out[:, :, :, 2:]
            print("Flow Computation Graph Initialized !!!!!! ")

        with tf.variable_scope("flow_interpolation"):
            Fdasht0 = (-1 * (1 - timestamp) * timestamp * F01) + (timestamp * timestamp * F10)
            Fdasht1 = ((1 - timestamp) * (1 - timestamp) * F01) - (timestamp * (1 - timestamp) * F10)  # TODO :Remove ()

            flow_interp_input = tf.concat([frame0, frame1,
                                           flow_back_wrap(frame1, Fdasht1),
                                           flow_back_wrap(frame0, Fdasht0),
                                           Fdasht0, Fdasht1], axis=3)
            flow_interp_output, _ = UNet(flow_interp_input,
                                         output_channels=5,  # 2 channels for each flow, 1 visibilty map.
                                         decoder_extra_input=flow_comp_enc_out,
                                         first_kernel=3,
                                         second_kernel=3)
            # TODO: check Vto is 4 shaped or 3, use expand dimension
            deltaFt0, deltaFt1, Vt0 = flow_interp_output[:, :, :, :2], flow_interp_output[:, :, :, 2:4], \
                                      flow_interp_output[:, :, :, 4:5]

            deltaFt0 = lrelu(deltaFt0)
            deltaFt1 = lrelu(deltaFt1)
            Vt0 = tf.sigmoid(Vt0)
            Vt0 = tf.tile(Vt0, [1, 1, 1, 3])  # Copy same in all three channels
            Vt1 = 1 - Vt0

            Ft0, Ft1 = Fdasht0 + deltaFt0, Fdasht1 + deltaFt1

            normalization_factor = 1 / ((1 - timestamp) * Vt0 + timestamp * Vt1 + FLAGS.epsilon)
            pred_frameT = tf.multiply((1 - timestamp) * Vt0, flow_back_wrap(frame0, Ft0)) + \
                          tf.multiply(timestamp * Vt1, flow_back_wrap(frame1, Ft1))  # TODO: check flow_back_wrap
            pred_frameT = tf.multiply(normalization_factor, pred_frameT)
            print("Flow Interpolation Graph Initialized !!!!!! ")

    rec_loss = reconstruction_loss(pred_frameT, frameT)
    percep_loss = perceptual_loss(pred_frameT, frameT, layers=FLAGS.perceptual_mode)
    wrap_loss = wrapping_loss(frame0, frame1, frameT, F01, F10, Fdasht0, Fdasht1)
    smooth_loss = smoothness_loss(F01, F10)

    total_loss = FLAGS.reconstruction_scaling * rec_loss + \
                 FLAGS.perceptual_scaling * percep_loss + \
                 FLAGS.wrapping_scaling * wrap_loss + \
                 FLAGS.smoothness_scaling * smooth_loss

    with tf.variable_scope("global_step_and_learning_rate", reuse=reuse):
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step,
                                                   FLAGS.decay_rate,
                                                   staircase=FLAGS.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope("optimizer", reuse=reuse):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SloMo_model')
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
            grads_and_vars = optimizer.compute_gradients(total_loss, tvars)
            train_op = optimizer.apply_gradients(grads_and_vars)

    # TODO: add more if needed.
    return Network(
        total_loss=total_loss,
        reconstruction_loss=rec_loss,
        perceptual_loss=percep_loss,
        wrapping_loss=wrap_loss,
        smoothness_loss=smooth_loss,
        pred_frameT=pred_frameT,
        Ft0=Ft0,
        Ft1=Ft1,
        Vt0=Vt0,
        grads_and_vars=grads_and_vars,
        train=tf.group(total_loss, incr_global_step, train_op),
        global_step=global_step,
        learning_rate=learning_rate
    )
