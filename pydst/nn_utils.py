# -*- coding: utf-8 -*-
""" Neural Network Utilities.

This module provides utilities classes that are handy in neural network construction.
"""

import os
import csv
import time
import datetime
import tensorflow as tf
import numpy as np 

def gloroSTD(params):
    """Function to return the STD according to the glorot and Bengio (2014)"""
    return (2./(params[0]+params[1]))**0.5

## TODO: transform function that szparams is only the output parameter
# Fully connected layer function
def fully_connected_layer(inputs, szparams, nonlin=tf.nn.relu, batchNorm=False, keepProb=1):
    """Function to construct a single fully connected layer.
    
    Args:
        inputs: Input tensor.
        szparams: Input and output sizes in a tuple.
        nonlin: Non-linearity to be used. Default is RELU.
        BatchNorm: Boolean to apply batch normalization
        keepProb: Probability to keep a certain weight to be 
            used in the dropout function.
        
    Returns:
        output: Output tensor of layer with transformations.
            -- Affine, (batch), non-linearity, (Dropout) --
    """
    
    weights = tf.Variable(tf.truncated_normal([szparams[0], szparams[1]], stddev=gloroSTD(szparams)), name='weights')             
    biases = tf.Variable(tf.zeros([szparams[1]]), name='biases')
    
    affine_output = tf.matmul(inputs, weights) + biases
    
    if batchNorm:
        
        epsilon = 0.001
        gamma = tf.Variable(tf.ones([layerParams[1]]), name='scale')
        beta = tf.Variable(tf.zeros([layerParams[1]]), name='shift')
        
        mean, var = tf.nn.moments(affine_output, axes=[0], name='moments')
        
        affine_output = tf.nn.batch_normalization(affine_output, mean, var, beta, gamma, epsilon, name='batch_normalization')

    output = nonlin(affine_output)
    
    if dropProb!=1:
        output = tf.nn.dropout(output, keepProb)
    return output


def multi_fcLayers(inputs, szparams, nonlin=tf.nn.relu, batchNorm=False, keepProb=1):
    """Function to construct numerous feed forward layers.
    
    Given arrays of sizes, non-linearities, batch normalization
    booleans and keep wieght probabilities; numerous layers are
    constructed.
    
    Args:
        inputs: Input tensor.
        szparams: List with the inputs and outputs of the layers.
        nonlin: List of the non-linearities of the different
            layers. If a single non-linearity is given, this is 
            transformed into a list.
        batchNorm: List of batch normalizaton booleans. If a
            single boolean is given this is transformed into a
            list.
        keepProb: List of keep probabilities. If a single boolean
            is given this is transformed into a list.
        
    Return:
        layer_outs: A list with the outputs of each of the layers.
    """
    
    layer_outs = []
    
    if not isinstance(nonlin, list):
        nonlin = [nonlin]*len(szparams)
    if not isinstance(batchNorm, list):
        batchNorm = [batchNorm]*len(szparams)
    if not isinstance(keepProb, list):
        keepProb = [keepProb]*len(szparams)
    
    din = inputs
    for lidx, sz in enumerate(szparams):
        if lidx != 0:
            din = layer_outs[-1]
            
        with tf.name_scope('fc-layer-{0}'.format(lidx)):
            lout = fully_connected_layer(data_in, sz, nonlin=nonlin[lidx], batchNorm=batchNorm, keepProb=keepProb)
            layer_outs.append(lout)
            
    return layer_outs

def define_summaries(scalar_sd={}, histogram_sd={}, image_sd={}):
    """Function to define summaries
    
    Summaries are provided in dictionaries where the key is used
    for the name of the summary and the value is used as the num
    to be stored.
    
    Args:
        scalar_sd: Scalar values to be saved as summaries.
        histogram_sd: Values to be stored as histograms.
        image_sd: Values to be stored as images.
    Returns:
        merged_summary: The merged summary of saved items.
    """
    for key, value in scalar_sd.items():
        tf.summary.scalar(key, value)
    for key, value in histogram_sd.items():
        tf.summary.histogram(key, value)
    for key, value in image_sd.items():
        tf.summary.image(key, value)
    summary_merged = tf.summary.merge_all()
    return summary_merged

