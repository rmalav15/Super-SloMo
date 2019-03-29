import tensorflow as tf
import os
import cv2
import collections
import random
import numpy as np

ALLOWED_VIDEO_EXTENSION = ["mov"]
EXTRACT_FRAMES_NAME_START = 1000000


def check_video_extension(name):
    for ext in ALLOWED_VIDEO_EXTENSION:
        if name.endswith(ext):
            return True
    return False


def pad_function(images_list):  # TODO: make it efficinet
    min = np.Inf
    out_list = []
    for images in images_list:
        if len(images) < min:
            min = len(images)

    for images in images_list:
        out_list.append(images[0:min])
    return out_list


def extract_frames(video_dir, tmp_ext_dir, refresh=False):  # TODO: Add multiprocessing/multithreading here
    video_list = os.listdir(video_dir)
    video_list = [_ for _ in video_list if check_video_extension(_)]
    image_dirs = []

    for video_name in video_list:
        print("Extracting: ", video_name)
        video_path = os.path.join(video_dir, video_name)
        extract_path = os.path.join(tmp_ext_dir, video_name.split(".")[0])
        image_dirs.append(extract_path)

        # Exists and non empty
        if not refresh:
            if os.path.exists(extract_path) and os.listdir(extract_path) != []:
                print(extract_path, "already exists, skipping")
                continue

        if not os.path.exists(extract_path):
            os.mkdir(extract_path)

        vidObj = cv2.VideoCapture(video_path)
        count, success = 0, 1

        while success:
            success, image = vidObj.read()
            cv2.imwrite(os.path.join(extract_path, str(EXTRACT_FRAMES_NAME_START + count) + ".png"), image)
            count += 1

    return video_list, image_dirs


class DataLoader:

    def __init__(self, FLAGS):
        self.queue_capacity = FLAGS.queue_capacity
        self.video_dir = FLAGS.video_dir
        self.tmp_ext_dir = FLAGS.tmp_ext_dir
        self.batch_size = FLAGS.batch_size
        self.in_between_frames = FLAGS.in_between_frames
        self.delete_tmp_folder = FLAGS.delete_tmp_folder
        self.resize_height = FLAGS.resize_height
        self.resize_width = FLAGS.resize_width
        self.queue_thread = FLAGS.queue_thread

        # Check the input directory
        if (self.video_dir is None) or (not os.path.exists(self.video_dir)):
            raise ValueError('Input directory is not provided')

        if not os.path.exists(self.tmp_ext_dir):
            os.makedirs(self.tmp_ext_dir)

        self.video_names, self.image_dirs = extract_frames(self.video_dir, self.tmp_ext_dir)

        if len(self.video_names) is 0:
            raise ValueError('No training video available.')

        if len(self.video_names) < self.batch_size:
            print("Reducing batch size to: ", len(self.video_names))  # todo: Replace with logger
            self.batch_size = len(self.video_names)

    def getTrainData(self):

        # Data = collections.namedtuple('Data', 'paths, inputs')
        Data = collections.namedtuple('Data', 'paths')
        image_paths_lists = [[os.path.join(path, imageName) for imageName in os.listdir(path)] for path in
                             self.image_dirs]
        image_paths_lists = pad_function(image_paths_lists)
        #print(image_paths_lists)
        # TODO: add backward images also.
        # image_paths_lists_deep_copy_list = [random.sample(image_paths_lists, len(image_paths_lists)) for _ in
        #                                     range(self.batch_size)]
        image_paths_lists_deep_copy_list = [image_paths_lists.copy() for _ in  range(self.batch_size)]

        with tf.variable_scope('data_loader'):
            # This  makes shallow copy
            # images_paths_list_tensor = tf.convert_to_tensor(image_paths_lists, dtype=tf.string)
            # images_paths_list_tensor_list = [tf.random.shuffle(tf.identity(images_paths_list_tensor)) for _ in
            #                                  range(self.batch_size)]

            images_paths_list_tensor_list = tf.train.shuffle_batch()

            # images_paths_list_tensor_list = [tf.convert_to_tensor(shuffled_image_paths_lists, dtype=tf.string) for
            #                                  shuffled_image_paths_lists in image_paths_lists_deep_copy_list]
            images_path_list = tf.train.slice_input_producer(images_paths_list_tensor_list, shuffle=True,
                                                             capacity=self.queue_capacity)
            images_path = tf.train.slice_input_producer(images_path_list, shuffle=False, capacity=self.queue_capacity)
            batch_images_path = tf.train.batch(images_path,
                                               batch_size=self.in_between_frames + 2,
                                               num_threads=self.queue_thread,
                                               allow_smaller_final_batch=True)

            with tf.name_scope('image_load'):
                batch_images_list = [tf.map_fn(lambda image_path: tf.read_file(image_path), batch_path) for batch_path in
                                     batch_images_path]
                print(batch_images_list)
                batch_images_list = [
                    tf.map_fn(lambda image: tf.image.decode_png(image, channels=3), batch_images, dtype=tf.uint8) for
                    batch_images in batch_images_list]
                batch_images_list = [
                    tf.map_fn(lambda image: tf.image.convert_image_dtype(image, dtype=tf.float32), batch_images,
                              dtype=tf.float32) for batch_images in batch_images_list]

            # TODO: Check assertion for channel 3
            # with tf.name_scope('image_resizing'):
            #     input_batch_images = [tf.identity(image) for image in batch_images]
            #     # TODO: tf.image.resize_images, Require 4-d images with batch, add extra dimension, then remove it

            return Data(
                #inputs=batch_images_list,
                paths=batch_images_path
            )


    def delete_tmp_folder(self):
        return None


# test the data loader
def data_main(FLAGS):
    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()
    data = DataLoader(FLAGS)
    output = data.getTrainData()

    with tf.Session() as sess:
        print("run................")
        sess.run((var_init, table_init))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print(sess.run(output))
        # print(sess.run(output))
        print("done................")
        coord.request_stop()
        coord.join(threads)
