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

    def raw_input_fn(self):

        # Load the data from the records
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read_up_to(self._filename_queue,
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

        all_song = tf.cast(tf.decode_raw(features['song'], tf.int32), tf.float32)
        all_tags = tf.cast(tf.decode_raw(features['tags'], tf.int32), tf.float32)

        if self._sample_depth != 1:
            all_song = tf.reshape(all_song, [-1, self._num_samples, self._sample_depth])

        # Reduce tags, selective tags, merge tags
        if self._selective_tags is not None:
            selective_group_tags = []
            for group in self._selective_tags:
                merged_group_tags = []
                for tag in group:
                    merged_group_tags.append(tf.slice(all_tags, [0, tag], [-1, 1]))
                selective_group_tags.append(tf.clip_by_value(tf.add_n(merged_group_tags),
                                                             tf.constant(0, dtype=tf.float32),
                                                             tf.constant(1, dtype=tf.float32)))
            tags = tf.concat(selective_group_tags, axis=1)

        else:
            if self._num_tags is None or self._num_tags == -1:
                target_size = self._max_tags
            elif self._num_tags > 0:
                target_size = min(self._num_tags, self._max_tags)
            else:
                raise ValueError('target_size must be -1 or > 0')
            tags = tf.slice(all_tags, [0, 0], [-1, target_size])

        # Reduce samples depending on num_samples
        # This is not definite if window_size is given
        if self._num_samples is None or self._num_samples == -1:
            num_samples = self._max_samples
        elif self._num_samples > 1:
            num_samples = min(self._num_samples, self._max_samples)
        else:
            raise ValueError('Number of samples must be -1 or > 0')
        difference = self._max_samples - num_samples
        start_red = int(difference / 2.0)
        end_red = self._max_samples - start_red

        if self._sample_depth == 1:
            song = tf.slice(all_song, [0, start_red], [-1, end_red])
        else:
            song = tf.slice(all_song, [0, start_red, 0], [-1, end_red, -1])

        # If window_size is not none check even samples and reduce
        # appropriately.
        if self._window_size is not None:
            # window_size is samples per window
            # Int wil floor the float number
            samples_to_keep = (int(num_samples / self._window_size) * self._window_size)
            song = tf.slice(song, [0, 0], [-1, samples_to_keep])

        # Shuffle
        if self._shuffle:
            song, tags = tf.train.shuffle_batch(
                [song, tags],
                batch_size=self._batch_size,
                capacity=self._batch_size*10,
                num_threads=multiprocessing.cpu_count(),
                min_after_dequeue=10,
                enqueue_many=True)

        # Reshape depending on shape parameter
        if self._data_shape == 'flat':
            if self._sample_depth == 1:
                feats = song
            else:
                feats = tf.reshape(song, [-1, self._sample_depth * num_samples])
        elif self._data_shape == 'image':
            if self._sample_depth == 1:
                feats = tf.expand_dims(song, axis=2)
            else:
                feats = tf.expand_dims(song, axis=3)
        else:
            raise ValueError('Shape not recognized!')


        # Create windows depending on window_size parameter
        if self._window_size is not None:
            num_windows = int(num_samples / self._window_size)
            feats = tf.split(value=feats,
                             num_or_size_splits=num_windows,
                             axis=1,
                             name='window_splitter')

        return feats, tags