import tensorflow as tf
from utils import print_configuration_op
from data_loader import data_main
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

Flags = tf.app.flags

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
Flags.DEFINE_integer('in_between_frames', 1, 'The frames to predict in between. Currently Allowed 1|3|7 (as per paper)')
Flags.DEFINE_integer('batch_thread', 1, 'The numner of threads to process image queue for generating batches')
Flags.DEFINE_integer('slim_num_readers', 1, 'The number reader for slim TFreader')
Flags.DEFINE_integer('tfrecord_threads', 6, 'The threads of the queue (More threads can speedup the training process.')
Flags.DEFINE_integer('resize_width', 320, 'The width of the training image')
Flags.DEFINE_integer('resize_height', 240, 'The width of the training image')

# Trainer Parametes
Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
Flags.DEFINE_integer('num_epochs', 100000, 'Training/Validation epochs, used in TFreader')

FLAGS = Flags.FLAGS

# Print the configuration of the model
print_configuration_op(FLAGS)

def main():
    data_main(FLAGS)


if __name__ == "__main__":
    main()
