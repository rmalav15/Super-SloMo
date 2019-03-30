from __future__ import division

import tensorflow as tf
import os
import cv2
import collections
import numpy as np
import six
import threading
import sys
from datetime import datetime

slim = tf.contrib.slim

ALLOWED_VIDEO_EXTENSION = ["MOV"]


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_list_feature(value):
    """Wrapper for inserting a list of bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if isinstance(value, six.string_types):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def check_video_extension(name):
    for ext in ALLOWED_VIDEO_EXTENSION:
        if name.endswith(ext):
            return True
    return False

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._ndarray_data = tf.placeholder(dtype=tf.uint8)
    self._encode_jpeg = tf.image.encode_jpeg(self._ndarray_data, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def encode_jpeg(self, image_data):
    return self._sess.run(self._encode_jpeg,
                          feed_dict={self._ndarray_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _process_image(image_data, coder):
  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return height, width

def convert_frames_to_example(coder, frames, video_name, frame_nos, frames_in_input):
    height, width = _process_image(image_data=frames[0], coder = coder)
    assert frames_in_input == 3  # Only 1 frame prediction supported for now. # TODO
    channels = 3

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/channels': int64_feature(channels),
        'frames/frameNos': int64_feature(frame_nos),
        'frames/videoName': bytes_feature(video_name.encode('utf8')),
        'frames/frame0': bytes_feature(frames[0]),
        'image/frameT': bytes_feature(frames[1]),
        'image/frame1': bytes_feature(frames[2])}))
    return example


def cv2_process_video_list(coder,  thread_index, name, thread_videos_dist, tfrecord_ext_dir, frames_in_input):
    video_list = thread_videos_dist[thread_index]

    for video_path in video_list:
        video_name = video_path.split("/")[-1].split(".")[0]
        output_file = '%s-%s.tfrecord' % (name, video_name)
        output_path = os.path.join(tfrecord_ext_dir, output_file)
        writer = tf.python_io.TFRecordWriter(output_path)
        vidObj = cv2.VideoCapture(video_path)
        count, success = 0, 1
        frames = []
        frame_nos = []
        while True:
            success, image = vidObj.read()
            if not success:
                break
            image = image.astype(np.uint8)
            frames.append(coder.encode_jpeg(image))
            frame_nos.append(count)
            count += 1
            if not count % frames_in_input:
                example = convert_frames_to_example(coder, frames, video_name, frame_nos, frames_in_input)
                writer.write(example.SerializeToString())
                frames = []
                frame_nos = []

        writer.close()
        print('%s [thread %d]: Wrote %s images to %s' %
              (datetime.now(), thread_index, video_name, output_file))
        sys.stdout.flush()


def convert_TFrecord(name, video_dir, tfrecord_ext_dir, num_thread,
                     frames_in_input=3):  # TODO: Add multiprocessing/multithreading here
    video_list = os.listdir(video_dir)
    video_list = [_ for _ in video_list if check_video_extension(_)]
    video_list = [os.path.join(video_dir, video_name) for video_name in video_list]

    assert len(video_list) % num_thread == 0
    batch_per_thread = int(len(video_list) / num_thread)
    thread_videos_dist = [video_list[i * batch_per_thread: (i + 1) * batch_per_thread] for i in range(num_thread)]

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []

    for thread_index in range(num_thread):
        args = (coder, thread_index, name, thread_videos_dist, tfrecord_ext_dir, frames_in_input)
        t = threading.Thread(target=cv2_process_video_list, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('Finished writing all %d videos in data set.' % (len(video_list)))
    sys.stdout.flush()
    return video_list


class DataLoader:

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.train_video_list = None
        self.val_video_list = None

        # Check the input directory
        if (self.FLAGS.train_video_dir is None) or (not os.path.exists(self.FLAGS.train_video_dir)): # TODO: add val directory
            raise ValueError('Input directory is not provided')

        if not os.path.exists(self.FLAGS.tfrecord_train_dir):
            os.makedirs(self.FLAGS.tfrecord_train_dir)

        if not os.path.exists(self.FLAGS.tfrecord_val_dir):
            os.makedirs(self.FLAGS.tfrecord_val_dir)

    def extract_tfrecords(self, name):
        if name == "train":
            self.train_video_list = convert_TFrecord(name, self.FLAGS.train_video_dir, self.FLAGS.tfrecord_train_dir,
                                                   self.FLAGS.tfrecord_threads)
        if name == "val":
            pass


    def getTrainData(self, name):
        Data = collections.namedtuple('Data', 'input, paths')
        return None

    def delete_tmp_folder(self):
        return None


# test the data loader
def data_main(FLAGS):
    data = DataLoader(FLAGS)
    data.extract_tfrecords("train")
    # var_init = tf.global_variables_initializer()
    # table_init = tf.tables_initializer()
    # data = DataLoader(FLAGS)
    # output = data.getTrainData()
    #
    # with tf.Session() as sess:
    #     print("run................")
    #     sess.run((var_init, table_init))
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     while not coord.should_stop():
    #         print(sess.run(output))
    #     # print(sess.run(output))
    #     print("done................")
    #     coord.request_stop()
    #     coord.join(threads)
