from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import tensorflow as tf
from utils import print_configuration_op, compute_psnr
from data_loader import DataLoader
from model import SloMo_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

Flags = tf.app.flags

# The system parameter
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint.'
                                        'Checkpoint folder (Latest checkpoint will be taken)')
Flags.DEFINE_boolean('pre_trained_model', False,
                     'If set True, the weight will be loaded but the global_step will still '
                     'be 0. If set False, you are going to continue the training. That is, '
                     'the global_step will be initiallized from the checkpoint, too')
Flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'path to checkpoint file for the vgg19')
# Flags.DEFINE_string('task', None, 'The task: Slomo, Slogan')
# Flags.DEFINE_string('pre_trained_model_type', 'Slomo', 'The type of pretrained model (Slomo or Slogan)')

# DataLoader Parameters
Flags.DEFINE_string('train_video_dir',
                    '/mnt/069A453E9A452B8D/Ram/slomo_data/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos/'
                    'original_high_fps_videos',
                    'Video data folder')
Flags.DEFINE_string('val_video_dir', "/mnt/069A453E9A452B8D/Ram/slomo_data/tmp/",
                    'The directory to extract videos temporarily')
Flags.DEFINE_string('tfrecord_train_dir', "/mnt/069A453E9A452B8D/Ram/slomo_data/tfrecords/train",
                    'The directory to extract training tfrecords')
Flags.DEFINE_string('tfrecord_val_dir', "/mnt/069A453E9A452B8D/Ram/slomo_data/tfrecords/val",
                    'The directory to extract validation tfrecords, should be different from tf record train dir')
Flags.DEFINE_integer('batch_size', 2, 'Batch size of the input batch')
Flags.DEFINE_integer('in_between_frames', 1, 'The frames to predict in between 1|3|7. Currently Allowed 1')
Flags.DEFINE_integer('batch_thread', 4, 'The number of threads to process image queue for generating batches')
Flags.DEFINE_integer('slim_num_readers', 4, 'The number reader for slim TFreader')
Flags.DEFINE_integer('tfrecord_threads', 5, 'The number of threads for tfrecord extraction.')
Flags.DEFINE_integer('resize_width', 320, 'The width of the training image')
Flags.DEFINE_integer('resize_height', 240, 'The width of the training image')
Flags.DEFINE_integer('train_data_count', 116241, 'The number of samples in training tfrecords')
Flags.DEFINE_integer('val_data_count', None, 'The number of samples in training tfrecords')

# model configurations
Flags.DEFINE_integer('first_kernel', 7, 'First conv kernel size in flow computation network')
Flags.DEFINE_integer('second_kernel', 5, 'First conv kernel size in flow computation network')
Flags.DEFINE_float('epsilon', 1e-12, 'The eps added to prevent nan')
Flags.DEFINE_string('perceptual_mode', 'VGG54', 'The type of feature used in perceptual loss')
Flags.DEFINE_float('reconstruction_scaling', 0.1, 'The scaling factor for the reconstruction loss')
Flags.DEFINE_float('perceptual_scaling', 1.0, 'The scaling factor for the perceptual loss')
Flags.DEFINE_float('wrapping_scaling', 1.0, 'The scaling factor for the wrapping loss')
Flags.DEFINE_float('smoothness_scaling', 50.0, 'The scaling factor for the smoothness loss')

# Trainer Parameters
Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving checkpoint')

FLAGS = Flags.FLAGS

# Print the configuration of the model
print_configuration_op(FLAGS)

# Check Directories
if FLAGS.output_dir is None or FLAGS.summary_dir is None:
    raise ValueError('The output directory and summary directory are needed')

# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

# Check the summary directory to save the event
if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

# Check in between frame constrain
if FLAGS.in_between_frames is not 1:
    raise ValueError("only 1 frame prediction is implemented")

# Initialize DataLoader
data_loader = DataLoader(FLAGS)

# The training mode
if FLAGS.mode == 'train':

    # check train tfrecord empty, extract otherwise
    if not os.listdir(FLAGS.tfrecord_train_dir):
        print("training tfrecord directory empty extracting tfrecords")
        data_loader.extract_tfrecords("train")
    else:
        print("WARNING: Skipping train tfrecord extraction")
        if FLAGS.train_data_count is None:
            raise ValueError("train_data_count needed if data extraction is not done")

    # check val tfrecord empty, extract otherwise
    # TODO: Implement validation part later
    # if not os.listdir(FLAGS.tfrecord_val_dir):
    #     print("training tfrecord directory empty extracting tfrecords")
    #     data_loader.extract_tfrecords("val")
    # else:
    #     print("WARNING: Skipping val tfrecord extraction")

    # get train and val batch
    data_train = data_loader.get_data("train")
    # data_val = data_loader.get_data("val")

    # get slomo output
    net_train = SloMo_model(data_train.frame0, data_train.frame1, data_train.frameT, FLAGS, reuse=False)
    # net_val = SloMo_model(data_val.frame0, data_val.frame1, data_val.frameT, FLAGS, reuse=True)

    print('Finish building the network!!!')

    # Convert back to uint8
    frame0 = tf.image.convert_image_dtype(data_train.frame0, dtype=tf.uint8)
    frame1 = tf.image.convert_image_dtype(data_train.frame1, dtype=tf.uint8)
    frameT = tf.image.convert_image_dtype(data_train.frameT, dtype=tf.uint8)
    pred_frameT = tf.image.convert_image_dtype(net_train.pred_frameT, dtype=tf.uint8)

    # Compute PSNR
    with tf.name_scope("compute_psnr"):
        psnr = compute_psnr(frameT, pred_frameT)

    # Add Image summaries
    with tf.name_scope("input_summaries"):
        tf.summary.image("frame0", frame0)
        tf.summary.image("frame1", frame1)

    with tf.name_scope("target_summary"):
        tf.summary.image("frameT", frameT)

    with tf.name_scope("output_summaries"):
        tf.summary.image("predicted_frameT", pred_frameT)
        tf.summary.image("FTO", tf.expand_dims(net_train.Ft0[:, :, :, 0], -1))  # showing only one channel
        tf.summary.image("FT1", tf.expand_dims(net_train.Ft1[:, :, :, 0], -1))  # showing only one channel
        tf.summary.image("VTO", net_train.Vt0)

    # Add scalar summaries
    tf.summary.scalar("PSNR", psnr)
    tf.summary.scalar("total_loss", net_train.total_loss)
    tf.summary.scalar("wrapping_loss", net_train.wrapping_loss)
    tf.summary.scalar("smoothness_loss", net_train.smoothness_loss)
    tf.summary.scalar("perceptual_loss", net_train.perceptual_loss)
    tf.summary.scalar("reconstruction_loss", net_train.reconstruction_loss)
    tf.summary.scalar('learning_rate', net_train.learning_rate)

    # Define the saver and weight initiallizer
    saver = tf.train.Saver(max_to_keep=10)

    # Get trainable variable
    train_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="SloMo_model")
    weight_initializer = tf.train.Saver(train_var_list)

    # Restore VGG
    vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
    vgg_restore = tf.train.Saver(vgg_var_list)
    # print(vgg_var_list)

    # Start the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Use supervisor to coordinate all queue and summary writer
    # TODO: Deprecated, Update with tf.train.MonitoredTrainingSession
    sv = tf.train.Supervisor(logdir=FLAGS.summary_dir, save_summaries_secs=0, saver=None)

    with sv.managed_session(config=config) as sess:
        if (FLAGS.checkpoint is not None) and (FLAGS.pre_trained_model is False):
            print('Loading model from the checkpoint...')
            checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
            saver.restore(sess, checkpoint)

        elif (FLAGS.checkpoint is not None) and (FLAGS.pre_trained_model is True):
            print('Loading weights from the pre-trained model')
            weight_initializer.restore(sess, FLAGS.checkpoint)

        if FLAGS.vgg_ckpt is not None:
            vgg_restore.restore(sess, FLAGS.vgg_ckpt)
        else:
            raise ValueError("VGG Checkpoint is needed")  # TODO: if not exist write script for downloading

        # Performing the training
        if FLAGS.max_epoch is None:
            if FLAGS.max_iter is None:
                raise ValueError('one of max_epoch or max_iter should be provided')
            else:
                max_iter = FLAGS.max_iter
        else:
            max_iter = FLAGS.max_epoch * data_train.steps_per_epoch

        print('Optimization starts!!!')
        start = time.time()

        for step in range(max_iter):
            fetches = {
                "train": net_train.train,
                "global_step": sv.global_step,
            }

            if ((step + 1) % FLAGS.display_freq) == 0:
                fetches["total_loss"] = net_train.total_loss
                fetches["wrapping_loss"] = net_train.wrapping_loss
                fetches["smoothness_loss"] = net_train.smoothness_loss
                fetches["perceptual_loss"] = net_train.perceptual_loss
                fetches["reconstruction_loss"] = net_train.reconstruction_loss
                fetches["PSNR"] = psnr
                fetches["learning_rate"] = net_train.learning_rate
                fetches["global_step"] = net_train.global_step

            if ((step + 1) % FLAGS.summary_freq) == 0:
                fetches["summary"] = sv.summary_op

            results = sess.run(fetches)

            if ((step + 1) % FLAGS.summary_freq) == 0:
                print('Recording summary !!!!')
                sv.summary_writer.add_summary(results['summary'], results['global_step'])

            if ((step + 1) % FLAGS.display_freq) == 0:
                train_epoch = math.ceil(results["global_step"] / data_train.steps_per_epoch)
                train_step = (results["global_step"] - 1) % data_train.steps_per_epoch + 1
                rate = (step + 1) * FLAGS.batch_size / (time.time() - start)
                remaining = (max_iter - step) * FLAGS.batch_size / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    train_epoch, train_step, rate, remaining / 60))
                print("global_step", results["global_step"])
                print("learning_rate", results['learning_rate'])
                print("PSNR", results["PSNR"])
                print("total_loss", results["total_loss"])
                print("wrapping_loss", results["wrapping_loss"])
                print("smoothness_loss", results["smoothness_loss"])
                print("perceptual_loss", results["perceptual_loss"])
                print("reconstruction_loss", results["reconstruction_loss"])

            if ((step + 1) % FLAGS.save_freq) == 0:
                print('Save the checkpoint !!!!')
                saver.save(sess, os.path.join(FLAGS.output_dir, 'model'), global_step=sv.global_step)

        print('Optimization done!!!!!!!!!!!!')

else:
    raise ValueError("inference|test mode not implemented yet")
