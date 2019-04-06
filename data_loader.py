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
lock = threading.Lock()

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
    height, width = _process_image(image_data=frames[0], coder=coder)
    assert frames_in_input == 3  # Only 1 frame prediction supported for now. # TODO
    channels = 3
    image_format = 'jpeg'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/channels': int64_feature(channels),
        'image/shape': int64_feature([height, width, channels]),
        'image/format': bytes_feature(image_format),
        'frames/frameNos': int64_feature(frame_nos),
        'frames/videoName': bytes_feature(video_name.encode('utf8')),
        'frames/frame0': bytes_feature(frames[0]),
        'frames/frameT': bytes_feature(frames[1]),
        'frames/frame1': bytes_feature(frames[2])}))
    return example


def cv2_process_video_list(coder, thread_index, name, thread_videos_dist, tfrecord_ext_dir, frames_in_input,
                           width, height, thread_data_count):
    video_list = thread_videos_dist[thread_index]
    data_count = 0
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
            image = cv2.resize(image, (width, height))
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
        data_count += count
        print('%s [thread %d]: Wrote %s images to %s' %
              (datetime.now(), thread_index, video_name, output_file))
        sys.stdout.flush()
    lock.acquire()
    thread_data_count.append(data_count)
    lock.release()


def convert_TFrecord(name, video_dir, tfrecord_ext_dir, num_thread, width, height,
                     frames_in_input=3):  # TODO: Add multiprocessing/multithreading here
    video_list = os.listdir(video_dir)
    video_list = [_ for _ in video_list if check_video_extension(_)]
    video_list = [os.path.join(video_dir, video_name) for video_name in video_list]

    if len(video_list) == 0:
        raise ValueError(name + ": No videos found of allowed format")

    if len(video_list) < num_thread:
        num_thread = len(video_list)
        print('new num of thread for tfrecoder extractor: %d == len(video_list).' % (len(video_list)))
        sys.stdout.flush()
    else:
        assert len(video_list) % num_thread == 0

    total_data_count = 0
    batch_per_thread = int(len(video_list) / num_thread)
    thread_videos_dist = [video_list[i * batch_per_thread: (i + 1) * batch_per_thread] for i in range(num_thread)]

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    thread_data_count = []
    for thread_index in range(num_thread):
        args = (coder, thread_index, name, thread_videos_dist, tfrecord_ext_dir, frames_in_input,
                width, height, thread_data_count)
        t = threading.Thread(target=cv2_process_video_list, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    for c in thread_data_count:
        total_data_count += c

    print('Finished writing all %d videos (%d frames triplets) in data set.' % (len(video_list), total_data_count))
    sys.stdout.flush()

    return video_list, total_data_count


class DataLoader:

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.train_video_list = None
        self.val_video_list = None
        self.train_data_count = None
        self.val_data_count = None

        # Check the input directory
        if (self.FLAGS.train_video_dir is None) or (
                not os.path.exists(self.FLAGS.train_video_dir)):  # TODO: add val directory
            raise ValueError('Input directory is not provided')

        if not os.path.exists(self.FLAGS.tfrecord_train_dir):
            os.makedirs(self.FLAGS.tfrecord_train_dir)

        if not os.path.exists(self.FLAGS.tfrecord_val_dir):
            os.makedirs(self.FLAGS.tfrecord_val_dir)

    def extract_tfrecords(self, name):
        if name == "train":
            self.train_video_list, self.train_data_count = convert_TFrecord(name, self.FLAGS.train_video_dir,
                                                                            self.FLAGS.tfrecord_train_dir,
                                                                            self.FLAGS.tfrecord_threads,
                                                                            self.FLAGS.resize_width,
                                                                            self.FLAGS.resize_height)
        elif name == "val":
            self.val_video_list, self.val_data_count = convert_TFrecord(name, self.FLAGS.val_video_dir,
                                                                        self.FLAGS.tfrecord_val_dir,
                                                                        self.FLAGS.tfrecord_threads,
                                                                        self.FLAGS.resize_width,
                                                                        self.FLAGS.resize_height)
        else:
            raise ValueError("only train|val is allowed")

    def get_data(self, name):
        Data = collections.namedtuple('Data', 'frame0, frame1, frameT, video_name, frame_nos, steps_per_epoch')
        keys_to_features = {
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/channels': tf.FixedLenFeature([1], tf.int64),
            'image/shape': tf.FixedLenFeature([3], tf.int64),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'frames/frameNos': tf.FixedLenFeature([self.FLAGS.in_between_frames + 2], tf.int64),
            'frames/videoName': tf.FixedLenFeature((), tf.string, default_value=''),
            'frames/frame0': tf.FixedLenFeature((), tf.string, default_value=''),
            'frames/frame1': tf.FixedLenFeature((), tf.string, default_value=''),
            'frames/frameT': tf.FixedLenFeature((), tf.string, default_value='')
        }
        items_to_handlers = {
            'frame0': slim.tfexample_decoder.Image('frames/frame0', 'image/format'),
            'frame1': slim.tfexample_decoder.Image('frames/frame1', 'image/format'),
            'frameT': slim.tfexample_decoder.Image('frames/frameT', 'image/format'),
            'videoName': slim.tfexample_decoder.Tensor('frames/videoName'),
            'shape': slim.tfexample_decoder.Tensor('image/shape'),
            'frameNos': slim.tfexample_decoder.Tensor('frames/frameNos')
        }
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

        if name == "train":
            file_pattern = os.path.join(self.FLAGS.tfrecord_train_dir, "train-*")
            print(file_pattern)
            num_samples = self.train_data_count
        elif name == "val":
            file_pattern = os.path.join(self.FLAGS.tfrecord_val_dir, "val-*")
            num_samples = self.val_data_count
        else:
            raise ValueError("unkown name :" + name)

        dataset = slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=tf.TFRecordReader,
            decoder=decoder,
            num_samples=num_samples,
            items_to_descriptions=None)

        with tf.name_scope('dataset_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=self.FLAGS.slim_num_readers,
                common_queue_capacity=32 * self.FLAGS.batch_size,
                common_queue_min=8 * self.FLAGS.batch_size,
                shuffle=True if name == "train" else False)
            # num_epochs=self.FLAGS.num_epochs)

        [frame0, frame1, frameT, video_name, frame_nos] = provider.get(
            ['frame0', 'frame1', 'frameT', 'videoName', 'frameNos'])

        frame0 = tf.image.convert_image_dtype(frame0, dtype=tf.float32)
        frame0.set_shape([self.FLAGS.resize_height, self.FLAGS.resize_width, 3])

        frame1 = tf.image.convert_image_dtype(frame1, dtype=tf.float32)
        frame1.set_shape([self.FLAGS.resize_height, self.FLAGS.resize_width, 3])

        frameT = tf.image.convert_image_dtype(frameT, dtype=tf.float32)
        frameT.set_shape([self.FLAGS.resize_height, self.FLAGS.resize_width, 3])

        # TODO: Pre-processing image, labels and bboxes.

        output = tf.train.batch([frame0, frame1, frameT, video_name, frame_nos],
                                dynamic_pad=False,
                                batch_size=self.FLAGS.batch_size,
                                allow_smaller_final_batch=False,
                                num_threads=self.FLAGS.batch_thread,
                                capacity=64 * self.FLAGS.batch_size)

        return Data(
            frame0=output[0],
            frame1=output[1],
            frameT=output[2],
            video_name=output[3],
            frame_nos=output[4],
            steps_per_epoch=int(num_samples/self.FLAGS.batch_size)
        )

    def delete_tmp_folder(self):
        return None
