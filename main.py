import tensorflow as tf
from utils import print_configuration_op
from DataLoader import data_main

Flags = tf.app.flags

# DataLoader Parameters
Flags.DEFINE_string('videoDir', None, 'Video data folder')
Flags.DEFINE_string('tmpExtDir', "tmp/", 'The directory to extract videos temporarily')
Flags.DEFINE_integer('batchSize', 2, 'Batch size of the input batch')
Flags.DEFINE_integer('inBetweenFrames', 1, 'The frames to predict in between. Currently Allowed 1|3|7 (as per paper)')
Flags.DEFINE_boolean("deleteTmpFolder", False, 'Whether to delete extracted frames folder')

FLAGS = Flags.FLAGS

# Print the configuration of the model
print_configuration_op(FLAGS)


def main():
    data_main(FLAGS)


if __name__ == "__main__":
    main()
