# Note: Not a efficient implementation
# TODO: Restructure code to separate submodules from slomo

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from tqdm import tqdm
import tensorflow as tf
from model import SloMo_model_infer
import cv2
import numpy as np

from utils import print_configuration_op

Flags = tf.app.flags

# The system parameter
Flags.DEFINE_string('input_video_path', None, 'Input video')
Flags.DEFINE_string('output_video_path', None, 'Output slow motion video')
Flags.DEFINE_integer('slomo_rate', 2, 'Number of frames to insert between frames')
Flags.DEFINE_string('checkpoint', None, 'Slomo model path')

# model configurations
Flags.DEFINE_integer('first_kernel', 7, 'First conv kernel size in flow computation network')
Flags.DEFINE_integer('second_kernel', 5, 'First conv kernel size in flow computation network')
Flags.DEFINE_float('epsilon', 1e-12, 'The eps added to prevent nan')
Flags.DEFINE_string('perceptual_mode', 'VGG54', 'The type of feature used in perceptual loss')
Flags.DEFINE_float('reconstruction_scaling', 0.1, 'The scaling factor for the reconstruction loss')
Flags.DEFINE_float('perceptual_scaling', 1.0, 'The scaling factor for the perceptual loss')
Flags.DEFINE_float('wrapping_scaling', 1.0, 'The scaling factor for the wrapping loss')
Flags.DEFINE_float('smoothness_scaling', 50.0, 'The scaling factor for the smoothness loss')
Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')

FLAGS = Flags.FLAGS

# Print the configuration of the model
print_configuration_op(FLAGS)


def video_to_slomo(sess, fetch, frame0_ph, frame1_ph):
    in_video = cv2.VideoCapture(FLAGS.input_video_path)

    frame_width = int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(in_video.get(cv2.CAP_PROP_FPS))
    frame_count = int(in_video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(FLAGS.output_video_path, fourcc, fps, (frame_width, frame_height))

    frame0 = None
    frame1 = None
    count = 0
    pbar = tqdm(total=frame_count)
    while in_video.isOpened():
        pbar.update(1)
        ret, frame = in_video.read()
        if not ret or frame is None:
            break
        frame = frame.astype(np.float32)
        frame = frame / 255.0
        count = count + 1
        if count == 1:
            frame0 = frame
            continue
        frame1 = frame
        results = sess.run(fetch, feed_dict={frame0_ph: np.expand_dims(frame0, axis=0),
                                             frame1_ph: np.expand_dims(frame1, axis=0)})
        results = [(255 * r.pred_frameT).astype(np.uint8) for r in results]
        out_video.write((255 * frame0).astype(np.uint8))
        for f in results:
            out_video.write(f)
        frame0 = frame
    pbar.close()
    if frame1 is not None:
        out_video.write((255 * frame1).astype(np.uint8))
    in_video.release()
    out_video.release()


def main():
    start = time.clock()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    frame0_ph = tf.placeholder(tf.float32, shape=[1, None, None, 3], name="frame0")
    frame1_ph = tf.placeholder(tf.float32, shape=[1, None, None, 3], name="frame1")
    fetch = [
        SloMo_model_infer(frame0_ph, frame1_ph, FLAGS, reuse=tf.AUTO_REUSE,
                    timestamp=float(t + 1) / (FLAGS.slomo_rate + 1))
        for t in range(FLAGS.slomo_rate)]
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SloMo_model')
    weight_initializer = tf.train.Saver(var_list)
    if FLAGS.checkpoint is None:
        raise ValueError("model checkpoint needed")
    weight_initializer.restore(sess, FLAGS.checkpoint)
    video_to_slomo(sess, fetch, frame0_ph, frame1_ph)
    sess.close()
    time_taken = time.clock() - start
    print("Total time taken to process video: ", time_taken)


if __name__ == "__main__":
    main()
