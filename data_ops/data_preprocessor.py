import tensorflow as tf
from keras.utils import to_categorical


class DataPreprocessor:
    def __init__(self, config):
        self.params = config.pre_processing_params

    def resize_img(self, x, y):
        image_resized = tf.image.resize_images(x, [self.params.resize_size, self.params.resize_size])
        return image_resized, y

    def make_label_categorical(self, x, y):
        return x, to_categorical(y)
