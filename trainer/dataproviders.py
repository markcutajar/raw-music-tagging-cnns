# -*- coding: utf-8 -*-
"""Data providers.
This module provides classes for loading datasets and iterating over batches of
data points.

TODO: Logger and tags and docstrings
"""

import os
import threading
import numpy as np
import tensorflow as tf
DEFAULT_SEED = 12345


class DataProvider(object):
    """Data Provider from files and metadata"""
    
    def __init__(self, which_set, batch_size=20, target_size=-1, num_samples=131072, max_samples=465984, data_depth=1,
                shuffle_order=True, rng=None, root='../magnatagatune/dataset/raw/', shape='flat'):
        
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
            
            TODO: FINISH DOCSTRING
        """
        
        MAX_TAGS = 188
        
        # Check valid argument data
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or test. '
            'Got {}'.format(which_set))

        # Set data directory paths
        self.data_dir = os.path.join(root, 'tracks/')
        self.metad_path = os.path.join(root, (which_set + '_metadata.npz'))
        assert os.path.isfile(self.metad_path), (
            'Metadata file does not exist at expected path: ' + self.metad_path)
        assert os.path.isdir(self.data_dir), (
            'Data directory does not exist at expected path: ' + self.data_dir)
        
        # Load targets and tids
        metad = np.load(self.metad_path)
        self.targets = metad['targets']
        self.tids = metad['tids']
        
        if target_size < 1 and target_size != -1:
            raise ValueError('target_size must be -1 or > 0')  
        # Set number of tags
        if target_size == -1:
            self._target_size = MAX_TAGS
        else:
            self._target_size = min(target_size, MAX_TAGS)
     
    
        # Set batch information
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        self._max_num_batches = int(np.floor(self.tids.shape[0] / self._batch_size))
        self._maxq = self._batch_size * 5
        
        
        # Set shape of output
        assert shape in ['flat', 'image'], (
            'Expected shape to be either flat or image. '
            'Got {}.'.format(shape))
          
        samples_to_remove = max_samples - num_samples
        if samples_to_remove < 0:
            raise ValueError('num_samples {} is larger than max_samples {}'.format(num_samples, max_samples))
        
        if shape == 'flat':
            self.data_out_dim = [self._batch_size, num_samples * data_depth]
            self.queue_shape = [num_samples * data_depth]    
        elif shape == 'image' and data_depth == 1:
            self.data_out_dim = [self._batch_size, num_samples, 1]
            self.queue_shape = [num_samples, 1]
        else:
            self.data_out_dim = [self._batch_size, num_samples, data_depth, 1]
            self.queue_shape = [num_samples, data_depth, 1]
        
        self._right_snip = int(np.round((samples_to_remove)/ 2.0))
        self._left_snip = (samples_to_remove) - self._right_snip
        self._data_depth = data_depth

        # Shuffle if needed
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng             
        if shuffle_order:
            self.shuffle()  
        
        # Initialize queue
        self.init_queue()

        
    def init_queue(self):
        """Initialises queue using tensorflow FIFOQueue"""
        self.q_din = tf.placeholder(tf.float32, shape=self.data_out_dim)
        self.q_tin = tf.placeholder(tf.float32, shape=[self._batch_size, self._target_size])

            # Set queue with FIFOQueue or RandomShuffleQueue
        self.q = tf.FIFOQueue(shapes=[self.queue_shape, [self._target_size]], 
                                                              capacity=self._maxq, dtypes=[tf.float32, tf.float32])        

        self.enqop = self.q.enqueue_many([self.q_din, self.q_tin])
        
        
    def get_data(self):
        """Function to dequeue and return batches"""
        data_batch, targets_batch= tf.train.batch(self.q.dequeue(), batch_size=self._batch_size, capacity=self._maxq)
        return data_batch, targets_batch
    
    
    def enable(self, sess, num_threads=1):
        threads = []
        self.shuffle()
        for idx in range(num_threads):
            thread = threading.Thread(target=self.load_q, args=(sess,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        return threads
        
    
    def disable(self, sess):
            sess.run(self.q.close(cancel_pending_enqueues=True))
            
    def load_q(self, sess):
        """Function to enqueue data in the object queue
        
        The data needs to be loaded using the tids.
        """
        maxp = self.mtids.shape[0]
        lowp = 0
        topp = 0
        
        while True:
            while topp != None:

                topp = lowp + self._batch_size
                if topp == maxp:
                    topp = None

                toload = self.mtids[lowp:topp]
                ctargets = self.mtargets[lowp:topp]
                if (self._target_size-self.mtargets.shape[1]) < 0:
                    ctargets = np.delete(ctargets, np.s_[(self._target_size-self.mtargets.shape[1])::], axis=1)

                for idx, tid in enumerate(toload):
                    loaded = np.load(self.data_dir + str(tid) + '.npy')
                    
                    if len(self.data_out_dim) == 2:
                        loaded = np.expand_dims(loaded.reshape((-1)), axis=0)
                        if self._left_snip != 0:
                            loaded = np.delete(loaded, np.s_[:(self._left_snip * self._data_depth)], axis=1)
                        if self._right_snip != 0:
                            loaded = np.delete(loaded, np.s_[(-self._data_depth * self._right_snip):], axis=1) 
                    elif len(self.data_out_dim) == 3:
                        loaded = np.expand_dims(loaded.reshape((-1, 1)), axis=0)
                        if self._left_snip != 0:
                            loaded = np.delete(loaded, np.s_[:(self._left_snip)], axis=1)
                        if self._right_snip != 0:
                            loaded = np.delete(loaded, np.s_[(-self._right_snip):], axis=1)
                    else:
                        loaded = np.expand_dims(np.expand_dims(loaded, axis=0), axis=3)
                        if self._left_snip != 0:
                            loaded = np.delete(loaded, np.s_[:(self._left_snip)], axis=1)
                        if self._right_snip != 0:
                            loaded = np.delete(loaded, np.s_[(-self._right_snip):], axis=1)
                        
                    if idx == 0:
                        cdata = loaded
                    else:
                        cdata = np.concatenate((cdata, loaded), axis=0)

                lowp = topp
                
                try:
                    sess.run(self.enqop, feed_dict={self.q_din: cdata, self.q_tin: ctargets})
                except tf.errors.CancelledError:
                    return
            
            self.shuffle()
            lowp = 0
            topp = 0
            
            
    def shuffle(self):
        """Randomly shuffles order of data."""          
        perm = self.rng.permutation(self.tids.shape[0])
        self.tids = self.tids[perm]
        self.targets = self.targets[perm, :]
        
        self.mtids = self.tids[0:self._max_num_batches * self._batch_size]
        self.mtargets = self.targets[0:self._max_num_batches * self._batch_size] 
        
            
    def batch_size(self):
        """Returns the _batch_size."""
        return self._batch_size
    
    
    def target_size(self):
        """Returns the _target_size."""
        return self._target_size
    
    
    def max_num_batches(self):
        """Returns the _max_num_batches."""
        return self._max_num_batches

    