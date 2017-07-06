# -*- coding: utf-8 -*-
"""Data providers.
This module provides classes for loading data and iterating over batches of
data points using TensorFlow's queue runners.

It loads the data from the file using the TFRecord reader, decodes it,
manipulates the samples and tags as needed, shuffles, queues and batches.
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
                 num_samples=None,
                 shuffle=True):

        """Class to load the data and provide batches to
        the calling function. Every run a batch is returned.
        Functions include either tfr or csv.

        :param filenames: File name and location of tfrecord
        :param batch_size: The size of batches to be provided
        :param num_epochs: The number of epochs to be returned.
            If None, and indefinite number is provided.
        :param num_tags: The number of tags in the target tensor
        :param num_samples: Number of samples in the feature tensors
        :param shuffle: Boolean whether to shuffle the batches or not
        :return:
        """
        self._batch_size = batch_size
        self._num_tags = num_tags

        self._num_samples = num_samples
        self._shuffle = shuffle

        with file_io.FileIO(metadata_file, 'r') as f:
            metadata = json.load(f)

        self._label_map = metadata['label_map']
        self._max_samples = metadata['max_num_samples']
        self._max_tags = metadata['max_num_tags']
        self._sample_depth = metadata['sample_depth']

        self._filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs)

    def batch_in(self):
        """ Function to provide batch data in.

        Simple function to process raw or 2-dimension (fbanks)
        data and provide it to the user. This is a TensorFlow
        function and hence provides a batch of data each time,
        the function is run in a session.

        :returns: Features and Labels of a batch
        """
        # Load the data from the
        data = self.data_load(self._batch_size)
        loaded_songs, loaded_tags = self.decode(data)

        # Data preparation
        tags = self.tag_prep(loaded_tags)
        songs = self.sample_prep(loaded_songs)
        songs, tags = self.remove_unused(songs, tags)
        songs = self.normalize(songs)

        # Queuing and output
        features, labels = self.batch(songs, tags)
        features = self.set_shape(features)
        return features, labels

    def windows_batch_in(self):
        """ Function to provide batch data where windowed data
        is needed to be batched into windowed song specific
        tensors.

        The data loaded, decoded, and a batch of tensors with
        first dimensionality being window number and second samples
        (or higher for fbanks) is processed and provided  to the
        user. This is a TensorFlow function and hence provides a
        batch of data each time, the function is run in a session.

        :returns: Features and Labels of a batch
        """
        windows_per_song = 12
        # Load data from file and decode data
        # First dimension is the window dimension * self._batch_size
        data = self.data_load(windows_per_song*self._batch_size)
        loaded_songs, loaded_tags = self.decode(data)

        # Merge songs and batch
        # Split into a list of 12 tensors each denoting the corresponding song
        windowed_songs = tf.split(loaded_songs, 12, name='WindowedSongs')
        # Re-stack the data
        songs = tf.stack(windowed_songs, name='StackedSongs')
        # Gather the tags of the corresponding songs
        tags = tf.gather(loaded_tags, list(range(0, 240, 12)))

        # Data preparation
        tags = self.tag_prep(tags)
        songs, tags = self.remove_unused(songs, tags)
        songs = self.normalize(songs)

        # Queuing and output
        features, labels = self.batch(songs, tags)
        features = self.set_shape(features)
        return features, labels

    # Load data
    def data_load(self, read_size):
        """Function to load the data from the record.

        :param read_size: Amount to read from record.
        :return: data (needs to be decoded)
        """
        with tf.name_scope('InputGenerator'):
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read_up_to(self._filename_queue,
                                                      num_records=read_size)
            data = tf.parse_example(
                serialized_example,
                features={
                    'tags': tf.FixedLenFeature([], tf.string),
                    'song': tf.FixedLenFeature([], tf.string)
                }
            )
        return data

    # Decode
    def decode(self, data):
        """Function to decode the data.

        :param data: Data read from the record
        :return: Decoded data depending on type
        """
        with tf.name_scope('Decoding'):
            if self._sample_depth != 1:
                original_songs = tf.cast(tf.decode_raw(data['song'], tf.float64), tf.float32)
                songs = tf.reshape(original_songs, [-1, self._max_samples, self._sample_depth])

            else:
                songs = tf.cast(tf.decode_raw(data['song'], tf.int32), tf.float32)
            tags = tf.cast(tf.decode_raw(data['tags'], tf.int32), tf.float32)
        return songs, tags

    # Reduce samples as needed
    def sample_prep(self, songs):
        """Songs' samples preparation (if needed).

        :param songs: Songs to be potentially clipped
        :return: Potentially clipped songs
        """
        with tf.name_scope('SamplePrep'):
            if self._num_samples is None or self._num_samples == -1:
                num_samples = self._max_samples
            else:
                num_samples = min(self._num_samples, self._max_samples)

            difference = self._max_samples - num_samples
            start_red = int(difference / 2.0)

            if self._sample_depth == 1:
                clipped_songs = tf.slice(songs, [0, start_red], [-1, num_samples])
            else:
                clipped_songs = tf.slice(songs, [0, start_red, 0], [-1, num_samples, -1])
        return clipped_songs

    # Batch songs and queue
    def batch(self, songs, tags):
        """Shuffelling, queuing and batching function.

        :param songs: Songs to be shuffled and queued
        :param tags: Tags to be shuffled and queued
        :return: Batch of songs and tags
        """
        # Batch and enqueue
        with tf.name_scope('Shuffle'):
            if self._shuffle:
                features, labels = tf.train.shuffle_batch(
                    [songs, tags],
                    batch_size=self._batch_size,
                    capacity=self._batch_size*10,
                    num_threads=multiprocessing.cpu_count(),
                    enqueue_many=True,
                    min_after_dequeue=self._batch_size)
            else:
                features, labels = songs, tags
        return features, labels

    # Strip to top 50 tags
    @staticmethod
    def tag_prep(tags):
        """Clip tags to top 50.

        __FUTURE__ clip tags to specified amount

        :param tags: Tags to be clipped
        :return: Top 50 tags
        """
        with tf.name_scope('TagPrep'):
            clipped_tags = tf.slice(tags, [0, 0], [-1, 50])
        return clipped_tags

    # Function to remove all samples that have all zeros in tags
    @staticmethod
    def remove_unused(songs, tags):
        """Function to removed songs from batch with all zero tags

        :param songs: Songs in batch
        :param tags: Tags in batch
        :return: Batch with reduced songs depending on sparsity
        """
        with tf.name_scope('FilterUnused'):
            target_sums = tf.reduce_sum(tags, axis=1, name='target_sum')
            where = tf.not_equal(target_sums, tf.constant(0, dtype=tf.float32))
            indices = tf.squeeze(tf.where(where), axis=1, name='squeeze_indices')
            filtered_songs = tf.gather(songs, indices, name='song_reduction')
            filtered_tags = tf.gather(tags, indices, name='tags_reduction')
        return filtered_songs, filtered_tags

    # Normalize function for features between -1 and 1
    @staticmethod
    def normalize(songs):
        """Function to batch normalize the songs.

        :param songs: Batch of songs to be normalized
        :return: Normalized batch
        """
        with tf.name_scope('InputNormalization'):
            scale_max = tf.reduce_max(songs)
            scale_min = tf.reduce_min(songs)
            factor = tf.maximum(scale_max, scale_min)
            norm_song = tf.div(songs, factor)
        return norm_song

    # Increase dimension at the end for convolution
    @staticmethod
    def set_shape(songs):
        """Function to increase axis for convolutions.

        :param songs: Batch to which to increase dimensionality
        :return: Batch with added last dimension
        """
        with tf.name_scope('SetShape'):
            feats = tf.expand_dims(songs, axis=-1)
        return feats