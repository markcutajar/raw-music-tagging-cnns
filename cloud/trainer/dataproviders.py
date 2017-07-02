# -*- coding: utf-8 -*-
"""Data providers.
This module provides classes for loading data and iterating over batches of
data points using TensorFlow's queue runners.

TODO: docstrings
"""
import os
import json
import tensorflow as tf
import multiprocessing
from tensorflow.python.lib.io import file_io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DataProvider(object):

    def __init__(self,
                 filenames,
                 metadata_file,
                 batch_size=20,
                 num_epochs=None,
                 num_tags=None,
                 selective_tags=None,
                 num_samples=None,
                 data_shape='image',
                 window_size=None,
                 shuffle=True):

        """Class to load the data and provide batches to
        the calling function. Every run a batch is returned.
        Functions include either tfr or csv.

        :param filenames: File name and location of tfrecord
        :param batch_size: The size of batches to be provided
        :param num_epochs: The number of epochs to be returned.
            If None, and indefinite number is provided.
        :param num_tags: The number of tags in the target tensor
        :param selective_tags: If specific names are given, targets
            provided in the list only are returned. If list is
            rank 2, the ones inside will be merged.
        :param num_samples: Number of samples in the feature tensors
        :param data_shape: The shape of the data. 'flat' for fully
            connected networks and 'image' for convolutional inputs.
        :param shuffle: Boolean whether to shuffle the batches or not
        :return:
        """
        self._batch_size = batch_size
        self._num_tags = num_tags
        self._window_size = window_size

        self._num_samples = num_samples
        self._data_shape = data_shape
        self._shuffle = shuffle
        self._selective_tags = None

        with file_io.FileIO(metadata_file, 'r') as f:
            metadata = json.load(f)

        self._label_map = metadata['label_map']
        self._max_samples = metadata['max_num_samples']
        self._max_tags = metadata['max_num_tags']
        self._sample_depth = metadata['sample_depth']

        # Find indices if selective tags is not None
        if selective_tags is not None:
            with file_io.FileIO(selective_tags, 'r') as f:
                self._selective_tags = json.load(f)['selective_tags']

            if type(self._selective_tags) == str:
                expanded_list = []
                for tag in self._selective_tags:
                    expanded_list.append([tag])
                self._selective_tags = expanded_list

            elif type(self._selective_tags) == list:
                pass

            else:
                raise ValueError('List of strings or list of lists for selective tags')

            selective_tag_indices = []
            for tag_group in self._selective_tags:
                to_merge = []
                for tag in tag_group:
                    to_merge.append(self._label_map.index(tag))
                selective_tag_indices.append(to_merge)
            self._selective_tags = selective_tag_indices

        self._filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs)

    def batch_in(self):

        # Load the data from the
        with tf.name_scope('InputGenerator'):
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read_up_to(self._filename_queue,
                                                      num_records=self._batch_size)
            features = tf.parse_example(
                serialized_example,
                features={
                    'tags': tf.FixedLenFeature([], tf.string),
                    'song': tf.FixedLenFeature([], tf.string)
                }
            )

        with tf.name_scope('Decoding'):
            if self._sample_depth != 1:
                original_song = tf.cast(tf.decode_raw(features['song'], tf.float64), tf.float32)
                all_song = tf.reshape(original_song, [-1, self._max_samples, self._sample_depth])
            else:
                all_song = tf.cast(tf.decode_raw(features['song'], tf.int32), tf.float32)
            all_tags = tf.cast(tf.decode_raw(features['tags'], tf.int32), tf.float32)

        # Reduce tags, selective tags, merge tags
        with tf.name_scope('TagPrep'):
            tags = tf.slice(all_tags, [0, 0], [-1, 50])

        # Reduce samples depending on num_samples
        # This is not definite if window_size is given
        with tf.name_scope('SamplePrep'):
            if self._num_samples is None or self._num_samples == -1:
                num_samples = self._max_samples
            else:
                num_samples = min(self._num_samples, self._max_samples)

            difference = self._max_samples - num_samples
            start_red = int(difference / 2.0)

            if self._sample_depth == 1:
                song = tf.slice(all_song, [0, start_red], [-1, num_samples])
            else:
                song = tf.slice(all_song, [0, start_red, 0], [-1, num_samples, -1])


        # Trim all zero samples
        with tf.name_scope('RmUnused'):
            target_sums = tf.reduce_sum(tags, axis=1, name='target_sum')
            where = tf.not_equal(target_sums, tf.constant(0, dtype=tf.float32))
            indices = tf.squeeze(tf.where(where), axis=1, name='squeeze_indices')
            song = tf.gather(song, indices, name='song_reduction')
            tags = tf.gather(tags, indices, name='tags_reduction')


        with tf.name_scope('Normalization'):
            scale_max = tf.reduce_max(song)
            scale_min = tf.reduce_min(song)
            factor = tf.maximum(scale_max, scale_min)
            song = tf.div(song, factor)


        # Batch and enqueue
        with tf.name_scope('Shuffle'):
            if self._shuffle:
                song, tags = tf.train.shuffle_batch(
                    [song, tags],
                    batch_size=self._batch_size,
                    capacity=self._batch_size*10,
                    num_threads=multiprocessing.cpu_count(),
                    enqueue_many=True,
                    min_after_dequeue=self._batch_size)

        # Reshape depending on shape parameter
        with tf.name_scope('SetShape'):
            if self._sample_depth == 1:
                feats = tf.expand_dims(song, axis=2)
            else:
                feats = tf.expand_dims(song, axis=3)

        return feats, tags
