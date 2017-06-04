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
    """MTAT data provider."""

    def __init__(self, which_set, batch_size=100, num_tags=-1, win_val=10, max_num_batches=-1, 
                 shuffle_order=True, rng=None, root='mockdataset/'):
    
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
        MAX_TAGS = 9
        MAX_SAMPLES = 4
        MAX_WINDOWING = 100
        
        # Check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or test '
            'Got {}'.format(which_set))
       
        # Check win_val and set sample_size, this can be varied.
        if win_val not in [1, 10, 40, 70, 100]:
            raise ValueError('window value must be one of 1, 10, 40, 70, 100')
        self._win_val = win_val
        self._sample_size = np.ceil(MAX_SAMPLES / win_val) 
        
        # Set data path for archives
        self._which_set = which_set
        self._data_dir = root + 'data' + str(self._win_val) + '/' + self._which_set + '/'
        self._metad_path = root + 'data' + str(self._win_val) + '/' + self._which_set + '_metadata.npy'
        assert os.path.isfile(self._metad_path), (
            'Metadata file does not exist at expected path: ' + self.metad_path)
        assert os.path.isdir(self._data_dir), (
            'Data directory does not exist at expected path: ' + self.data_dir)
        metad = np.load(self.metad_path).item()
        self.notrim_targets = metad['targets']
        self.notrim_tids = metad['tids']
        
        # Check num_tags and set target_size
        self._max_tags = MAX_TAGS
        self.num_tags(num_tags)
             
        # Check max batches    
        if max_num_batches < 1 and max_num_batches != -1:
            raise ValueError('max_batches must be >= 1 or -1')
        self._max_num_batches = max_num_batches
        
        # Check batch size
        self.batch_size(batch_size)
        self._maxq = self._batch_size * 5
        
        # Shuffle if needed
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng             
        if shuffle_order:
            self.shuffle()  
        
        # Initialize queue
        self.init_q()

        
    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size
                 
    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self.update_num_batches()

    @property
    def num_batches(self):
        """Number of batches available"""
        return self._num_batches

    @property
    def num_tags(self):
        """Number of target tags"""
        return self.target_size
                 
    @num_tags.setter
    def num_tags(self, value):
        if value < 1 and value != -1:
            raise ValueError('num_tags must be -1 or > 0')
        if value == -1:
            self._target_size = self._max_tags
        else:
            self._target_size = min(value, self._max_tags)

    def update_num_batches(self):
        """Update the number of batches in one epoch"""
        self._num_batches = int(np.floor(self.notrim_tids.shape[0] / self._batch_size))
        
    
    def shuffle(self):
        """Randomly shuffles order of data."""          
        perm = self.rng.permutation(self.notrim_tids.shape[0])
        self.notrim_tids = self.notrim_tids[perm]
        self.notrim_targets = self.notrim_targets[perm, :]
        self.trim()
        
    def trim(self):
        """Trim extra samples"""
        self.tids = self.notrim_tids[0:self.batch_num*self.batch_size]
        self.targets = self.notrim_targets[0:self.batch_num*self.batch_size] 
        
    def init_q(self):
        """Initialises queue using tensorflow FIFOQueue"""      
        
        self.q_din = tf.placeholder(tf.float32, shape=[self._batch_size, self._sample_size])
        self.q_tin = tf.placeholder(tf.float32, shape=[self_.batch_size, self._target_size])
        
        
        self.q = tf.FIFOQueue(shapes=[[self.sample_size],[self.target_size]], 
                                                          capacity=self.maxq, dtypes=[tf.float32, tf.float32])
        
        self.enqop = self.q.enqueue_many([self.q_din, self.q_tin])
        self.deqop = self.q.dequeue()

        self.input_batch, self.target_batch = tf.train.batch(self.deqop, batch_size=self.batch_size, capacity=self.maxq)
                           

    def load_q(self, sess):
        """Function for the loading thread"""
        maxp = self.tids.shape[0]
        lowp = 0

        while sess.run(self.q.size()) < self.maxq:
            topp = lowp + self.batch_size

            toload = self.tids[lowp:topp]
            ctargets = np.delete(self.targets[lowp:topp], np.s_[(self.target_size-self.targets.shape[1])::], axis=1)

            for idx, tid in enumerate(toload):
                    
                loaded = np.load(self.data_dir + str(tid) + '.npy')
                if idx == 0:
                    cdata = loaded
                else:
                    cdata=np.vstack((cdata, loaded))
                    
            if topp >= maxp-1:
                lowp = topp
            else:
                lowp = 0
            
            try:
                sess.run(self.enqop, feed_dict={self.q_din: cdata, self.q_tin: ctargets})
            except tf.errors.CancelledError:
                return
                             
    
    def start(self, sess):
        """Set loading queue thread and start it"""
        self.enq_thread = threading.Thread(target=self.load_q, args=[sess])
        self.enq_thread.isDaemon()
        self.enq_thread.start()

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=sess)
    
    def request(self, sess):
        """Request data by calling the run method"""
        run_options = tf.RunOptions(timeout_in_ms=4000)
        cdata_batch, ctarget_batch = sess.run([self.input_batch, self.target_batch], options=run_options)
        return  cdata_batch, ctarget_batch
    

    def stop(self, sess):
        sess.run(self.q.close(cancel_pending_enqueues=True))
        self.coord.request_stop()
        self.coord.join(self.threads)
        sess.close()
    
    
    def reset(self, sess, shuffle=False, seed=None):
        sess.run(self.q.close(cancel_pending_enqueues=True))
        self.coord.request_stop()
        self.coord.join(self.threads)
        sess.close()
        
        if shuffle==True:
            self.shuffle()
        
        self.enq_thread = threading.Thread(target=self.load_q, args=[sess])
        self.enq_thread.isDaemon()
        self.enq_thread.start()
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=sess)
    
    