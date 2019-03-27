import tensorflow as tf
import os
import cv2

ALLOWED_VIDEO_EXTENSION = [""]
EXTRACT_FRAMES_NAME_START = 1000000


def checkVideoExtension(name):
    for ext in ALLOWED_VIDEO_EXTENSION:
        if name.endswith(ext):
            return True
    return False


def extractFrames(videoDir, tmpExtDir):
    videoList = os.listdir(videoDir)
    videoList = [_ for _ in videoList if checkVideoExtension(_)]
    imageDirs = []

    for videoName in videoList:
        videoPath = os.path.join(videoDir, videoName)
        extractPath = os.path.join(tmpExtDir, videoName.split(".")[0])
        imageDirs.append(extractPath)

        if not os.path.exists(extractPath):
            os.mkdir(extractPath)

        vidObj = cv2.VideoCapture(videoPath)
        count, success = 0, 1

        while success:
            success, image = vidObj.read()
            cv2.imwrite(os.path.join(extractPath, str(EXTRACT_FRAMES_NAME_START + count) + ".png"), image)
            count += 1

    return videoList, imageDirs


class DataLoader:

    def __init__(self, FLAGS):
        self.videoDir = FLAGS.videoDir
        self.tmpExtDir = FLAGS.tmpExtDir
        self.batchSize = FLAGS.batchSize
        self.inBetweenFrames = FLAGS.inBetweenFrames
        self.deleteTmpFolder = FLAGS.deleteTmpFolder

        # Check the input directory
        if (self.videoDir is None) or (not os.path.exists(self.videoDir)):
            raise ValueError('Input directory is not provided')

        if not os.path.exists(self.tmpExtDir):
            os.makedirs(self.tmpExtDir)

        self.videoNames, self.imageDirs = extractFrames(self.videoDir, self.tmpExtDir)

        if len(self.videoNames) is 0:
            raise ValueError('No training video available.')

        if len(self.videoNames) < self.batchSize:
            print("Reducing batch size to: ", len(self.videoNames))  # todo: Replace with logger
            self.batchSize = len(self.videoNames)

    def getTrainData(self):
        imageDirsTesnor = tf.convert_to_tensor(self.imageDirs)
        batchDirs = [tf.train.slice_input_producer(imageDirsTesnor) for _ in range(self.batchSize)]

    def deleteTmpFolder(self):
        return None


# test the data loader
def data_main(FLAGS):
    return 0
