# -*- coding: utf-8 -*-
"""Data providers.
This module provides classes for loading data and iterating over batches of
data points using TensorFlow's queue runners.

TODO: Logger and tags and docstrings
"""

import tensorflow as tf
import multiprocessing
import numpy as np


class BatchInputFunctions(object):

    def __init__(self,
                 filenames,
                 label_map_file,
                 batch_size=20,
                 num_epochs=None,
                 num_tags=50,
                 selective_tags=None,
                 num_samples=131072,
                 data_shape='image',
                 shuffle=True):

        """Class to load the data and provide batches to
        the calling function. Every run a batch is returned.
        Functions include either tfr or csv.

        :param filenames: File name and location of tfrecord
        :param batch_size: The size of batches to be provided
        :param num_epochs: The number of epochs to be returned. If None,
            and indefinite number is provided.
        :param num_tags: The number of tags in the target tensor
        :param selective_tags: If specific names are given, targets
            provided in the list only are returned.
        :param num_samples: Number of samples in the feature tensors
        :param data_shape: The shape of the data. 'flat' for fully
            connected networks and 'image' for convolutional inputs.
        :param shuffle: Boolean whether to shuffle the batches or not
        :return:
        """

        self._filenames = filenames
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._num_tags = num_tags
        self._selective_tags = selective_tags
        self._num_samples = num_samples
        self._data_shape = data_shape
        self._shuffle = shuffle

        self._label_map = np.load(label_map_file)['label_map'].tolist()

    def tfr_input_fn(self):
        filename_queue = tf.train.string_input_producer(
            self._filenames, num_epochs=self._num_epochs)

        reader = tf.TFRecordReader()

        _, serialized_example = reader.read_up_to(filename_queue,
                                                  num_records=self._batch_size)

        features = tf.parse_example(
            serialized_example,
            features={
                'num_samples': tf.FixedLenFeature([], tf.int64),
                'sample_depth': tf.FixedLenFeature([], tf.int64),
                'num_tags': tf.FixedLenFeature([], tf.int64),
                'tags': tf.FixedLenFeature([], tf.string),
                'song': tf.FixedLenFeature([], tf.string)
            }
        )

        all_song = tf.decode_raw(features['song'], tf.int32)
        all_tags = tf.decode_raw(features['tags'], tf.int32)

        max_num_samples = tf.cast(features['num_samples'], tf.int64)
        samples_depth = tf.cast(features['sample_depth'], tf.int64)
        max_num_tags = tf.cast(features['num_tags'], tf.int64)

        # Reduce tags or selective
        if self._selective_tags is not None:
            raise NotImplementedError('Selective tags not implemented yet!')
        else:
            if self._num_tags is None or self._num_tags == -1:
                tags = all_tags
            elif self._num_tags > 0:
                target_size = min(self._num_tags, max_num_tags)
                tags = tf.slice(all_tags, [0, 0], [-1, target_size])
            else:
                raise ValueError('target_size must be -1 or > 0')

        # Reduce samples
        song = all_song
        if self._num_samples is None or self._num_samples == -1:
            pass
        elif self._num_samples > 1:
            num_samples = min(self._num_samples, max_num_samples)
            if samples_depth == 1:
                song = tf.slice(all_song, [0, 0], [-1, num_samples])
            else:
                raise NotImplementedError('Not implemented for FBanks!')

        # Shuffle
        if self._shuffle:
            # This operation builds up a buffer of rows so that, even between batches,
            # rows are fed to training in a suitably randomized order.
            song = tf.train.shuffle_batch(
                song,
                self._batch_size,
                capacity=self._batch_size * 10,
                min_after_dequeue=self._batch_size * 2 + 1,
                num_threads=multiprocessing.cpu_count(),
                enqueue_many=True,
                allow_smaller_final_batch=True
            )

        # Return depending on shape
        if self._data_shape == 'flat':
            return song, tags
        elif self._data_shape == 'image':
            if samples_depth == 1:
                song = tf.expand_dims(song, axis=2)
                return song, tags
            else:
                raise NotImplementedError('Not implemented for FBanks!')
        else:
            raise ValueError('Shape not recognized!')

    def csv_input_fn(self):
        raise NotImplementedError('Input from CSV not implemented yet!')
