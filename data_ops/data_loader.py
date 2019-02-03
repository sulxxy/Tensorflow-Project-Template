import tensorflow as tf


class DataLoader:
    def __init__(self, config, batch_size, shuffle_buffer_size=1000, pre_processor=None):
        self.config = config
        # load data here
        (x_train, y_train), (x_test, y_test) = globals()[config.dataset].load_data()
        self.train_dataset = tf.data.Dataset.from_sparse_tensor_slices((x_train, y_train))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        if pre_processor is not None:
            self.train_dataset = self.train_dataset.map(pre_processor)
            self.test_dataset = self.test_dataset.map(pre_processor)

        self.train_dataset = self.train_dataset.shuffle(shuffle_buffer_size).repeat().batch(batch_size)
        self.test_dataset = self.test_dataset.shuffle(shuffle_buffer_size).repeat().batch(batch_size)
        self.train_dataset_iter = self.train_dataset.make_one_shot_iterator()
        self.test_dataset_iter = self.test_dataset.make_one_shot_iterator()

    def next_train_batch(self):
        return self.train_dataset_iter.get_next()

    def next_test_batch(self):
        return self.test_dataset_iter.get_next()

    def get_train_set_iterator(self):
        return self.train_dataset_iter

    def get_test_set_iterator(self):
        return self.test_dataset_iter
