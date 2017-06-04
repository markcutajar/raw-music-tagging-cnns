# -*- coding: utf-8 -*-
"""Data providers.
This module provides classes for loading datasets and iterating over batches of
data points.
"""

import os
import threading
import numpy as np
import tensorflow as tf
from pydst import DEFAULT_SEED


class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, label_map, num_tags, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        -


class OneOfKDataProvider(DataProvider):
    """1-of-K classification target data provider.

    Transforms integer target labels to binary 1-of-K encoded targets.

    Derived classes must set self.num_classes appropriately.
    """

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(OneOfKDataProvider, self).next()
        return inputs_batch, self.to_one_of_k(targets_batch)

    def to_one_of_k(self, int_targets):
        """Converts integer coded class target to 1-of-K coded targets.

        Args:
            int_targets (ndarray): Array of integer coded class targets (i.e.
                where an integer from 0 to `num_classes` - 1 is used to
                indicate which is the correct class). This should be of shape
                (num_data,).

        Returns:
            Array of 1-of-K coded targets i.e. an array of shape
            (num_data, num_classes) where for each row all elements are equal
            to zero except for the column corresponding to the correct class
            which is equal to one.
        """
        one_of_k_targets = np.zeros((int_targets.shape[0], self.num_classes))
        one_of_k_targets[range(int_targets.shape[0]), int_targets] = 1
        return one_of_k_targets


class MGTAT_DataProviderDepreciated(OneOfKDataProvider):
    # TODO: What does function do?
    """"""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, num_tags=-1, root='magnatagatune/'):
        # TODO: Do the Function docstring
        """
        """

        # Check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or test '
            'Got {}'.format(which_set)
        )

        # Check a valid num_tags is given
        assert num_tags == -1 or num_tags > 0, (
            'Expected num_tags to be either -1 (full) or a positive number '
            'Got {}'.format(num_tags)
        )

        # Set data path for archives
        self.which_set = which_set
        data_path = root + '{}_archive.npz'.format(self.which_set)

        # Check if file required is available
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )

        # load data from compressed numpy file
        loaded = np.load(data_path)
        inputs, targets = loaded['inputs'], loaded['targets']

        # flatten inputs to vectors and upcast to float32
        inputs = inputs.reshape((inputs.shape[0], -1)).astype('float32')
        # label map gives strings corresponding to integer label targets
        label_map = loaded['label_map']

        # pass the loaded data to the parent class __init__
        super(MGTAT_DataProvider, self).__init__(
            inputs, targets, label_map, num_tags, batch_size, max_num_batches, shuffle_order, rng)
        
class MGTAT_DataProvider(DataProvider):
    
    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, num_tags=-1, root='magnatagatune/', win_val=10):
        
        self.load_size = batch_size
        
        # Check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or test '
            'Got {}'.format(which_set)
        )

        # Check a valid num_tags is given
        assert num_tags == -1 or num_tags > 0, (
            'Expected num_tags to be either -1 (full) or a positive number '
            'Got {}'.format(num_tags)
        )
        
        # Check batch size
        
        self.target_size = num_tags
        self.batch_size = batch_size
        
        # Check win_val
        SAMPLE_NO_WIN = 460000
        MAX_WINDOWING = 100
        
        if win_val < 1: # find and fix number
            self.win_val = 1
            self.sample_size = SAMPLE_NO_WIN
            
        elif win_val > max_windowing:
            self.win_val = MAX_WINDOWING
            self.sample_size = np.ceil(SAMPLE_NO_WIN / MAX_WINDOWING)
            
        else:
            self.win_val = win_val
            self.sample_size = np.ceil(SAMPLE_NO_WIN / win_val)
        
        
    
        # Set data path for archives
        self.which_set = which_set
        self.data_dir = root + 'data' + str(self.win_val) + '/' + which_set + '/'
        self.metad_path = root + 'data' + str(self.win_val) + '/' + 'metad.npy'
        
        assert os.path.isfile(metad_path), (
            'Metadata file does not exist at expected path: ' + metad_path
        )
        assert os.path.isdir(data_dir), (
            'Data directory does not exist at expected path: ' + data_dir
        )
        
        q_din = tf.placeholder(tf.float32, shape=[self.load_size, self.sample_size])
        q_tin = tf.placeholder(tf.float32, shape=[self.load_size, self.target_size])
        
        
        
        
        