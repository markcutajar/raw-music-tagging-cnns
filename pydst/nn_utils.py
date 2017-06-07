# -*- coding: utf-8 -*-
""" Neural Network Utilities.

This module provides utilities classes that are handy in neural network construction.
"""

import tensorflow as tf
import numpy as np
       
            
def residual_layer(self, residual_layer_name, name, activation=None):
    """Function to add a residual layer to the graph.
        
    :param inputs: 
          
    """   
    inputs = self.layer_outs[self.last_name]
    residues = self.layer_out[residual_layer_name]
        
    if inputs.shape != residues.shape:
        raise ValueError('Inputs {} and residues {} must have the same dimension for residual layer'
                        .format(inputs.shape, residues.shape))
       
    with self.graph.as_default():  
        with tf.name_scope(name):
                
            output = inputs + residues # Tensors do not add
            if activation is not None:
                output = activation(output)
                
        self.layer_outs[name] = output
        self.last_name = name


def save_model(graph, sess, path):
    with graph.as_default():
        with sess:
            model_saver = tf.train.Saver()
            model_saver.save(sess, path)        
            model_saver.export_meta_graph(path + '.meta')


def load_model(graph, sess, path):
    with graph.as_default():
        with sess:
            model_loader = tf.train.import_meta_graph(path + '.meta')
            model_loader.restore(sess, path)
            
            
def train(graph, sess, max_num_batches, error, accuracy, epochs=100):
    """Function to train the graph.
      
    Please note that this function gets information by an iterable.
      
    :param inputs
    """
    measured_errors = []
    measured_accs = []
    
    epoch_iterations = max_num_batched
    
    with graph.as_default():
      
        for epoch in range(epochs): 

                run_error = 0.
                run_acc = 0.

                for idx in range(epoch_iterations):
                    
                    [_, batch_error, accuracy_error] = sess.run(
                        [train_step, errors[error_name], accuracies[accuracy_name]])

                    run_error += batch_error
                    run_acc += batch_acc

                run_error /= provider.batch_size
                run_acc /= provider.batch_size

                measured_errors.append(run_error)
                measured_accs.append(run_acc)
        
        return measured_errors, measured_accs
    
    
    
    
    
    
    
    
    
    
    
    
    
    
