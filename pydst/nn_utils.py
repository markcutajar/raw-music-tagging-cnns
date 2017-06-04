# -*- coding: utf-8 -*-
""" Neural Network Utilities.

This module provides utilities classes that are handy in neural network construction.
"""

import tensorflow as tf


def glorot_std(params):
    """Function to return the STD    
    :param params: The input and output layer sizes.
    :returns Standard deviation according to Glorot and Bengio.
    """
    return (2./(params[0]+params[1]))**0.5


def fully_connected_layer(inputs, szparams, nonlin=tf.nn.relu, batchNorm=False, keepProb=1):
    """Function to construct a single fully connected layer.
    
    :param inputs: Input tensor.
    :param szparams: Input and output sizes in a tuple.
    :param nonlin: Non-linearity to be used. Default is RELU.
    :param batchNorm: Boolean to apply batch normalization
    :param keepProb: Probability to keep a certain weight to be 
            used in the dropout function.
    :returns output: Output tensor of layer with transformations.
            -- Affine, (batch), non-linearity, (Dropout) --
    """
    # TODO: transform function that szparams is only the output parameter
    weights = tf.Variable(tf.truncated_normal([szparams[0], szparams[1]], stddev=glorot_std(szparams)), name='weights')
    biases = tf.Variable(tf.zeros([szparams[1]]), name='biases')
    
    affine_output = tf.matmul(inputs, weights) + biases
    
    if batchNorm:
        
        epsilon = 0.001
        gamma = tf.Variable(tf.ones([szparams[1]]), name='scale')
        beta = tf.Variable(tf.zeros([szparams[1]]), name='shift')
        mean, var = tf.nn.moments(affine_output, axes=[0], name='moments')
        affine_output = tf.nn.batch_normalization(affine_output, mean, var, beta, gamma,
                                                  epsilon, name='batch_normalization')

    output = nonlin(affine_output)
    
    if keepProb != 1:
        output = tf.nn.dropout(output, keepProb)
    return output


def fc_layer_batch(inputs, szparams, nonlin=tf.nn.relu, batchNorm=False, keepProb=1):
    """Function to construct numerous feed forward layers.
    
    Given arrays of sizes, non-linearities, batch normalization
    booleans and keep wieght probabilities; numerous layers are
    constructed.
    
    :param inputs: Input tensor.
    :param szparams: List with the inputs and outputs of the layers.
    :param nonlin: List of the non-linearities of the different
            layers. If a single non-linearity is given, this is 
            transformed into a list.
    :param batchNorm: List of batch normalizaton booleans. If a
            single boolean is given this is transformed into a
            list.
    :param keepProb: List of keep probabilities. If a single boolean
            is given this is transformed into a list.
        :returns layer_outs: A list with the outputs of each of the layers.
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
            lout = fully_connected_layer(din, sz, nonlin=nonlin[lidx], batchNorm=batchNorm, keepProb=keepProb)
            layer_outs.append(lout)
            
    return layer_outs


def define_summaries(scalar_sd = None, histogram_sd = None, image_sd = None ):



# TODO: Check function on the tensorflow API documentation
def maxpool_layer(inputs, div=(2,2)):
    """Function to perform max-pooling on the provided inputs
    
    :param inputs: Input tensor.
    :param div: Check on the tensorflow API documentation to understand the difference between ksize and strides 
    :return: Max-pooled layer.
    """
    return tf.nn.max_pool(inputs, ksize=[1, div[0], div[1], 1], strides=[1, div[0], div[1], 1], padding='SAME')





class tfgraph(object):

    def __init__(self, graph):
        self.graph=graph
        self.layer_outs = []


    def add_ffl(self, inputs, outsize, nonlin=tf.nn.relu, batchNorm=False, keepProb):
        with self.graph:
            output = ff_layer(inputs, outsize, nonlin, batchNorm, keepProb)
        self.layer_outs.append(output)

    def add_conv1l(self, inputs, outsize, kersize, nonlin=tf.nn.relu, batchNorm=False, keepProb=1, padding, dilution):
        with self.graph:
            output = conv1_layer(inputs, outsize, kersize, nonlin, stride, batchNorm, keepProb, padding, dilution)   
        self.layer_outs.append(output)


    def add_conv2l(self, inputs, outsize, kersize, nonlin=tf.nn.relu, batchNorm=False, keepProb=1, padding, dilution):
        with self.graph:
            output = conv2_layer(inputs, outsize, kersize, nonlin, stride, batchNorm, keepProb, padding, dilution)
        self.layer_outs.append(output)


    def add_maxpool(self, inputs, div, padding="SAME"):
        with self.graph:
            output = maxpool(inputs, div, padding)
        self.layer_outs.append(output)


    def add_avgpool(self, inputs, div, padding="SAME"):	
        with self.graph:
            output = avgpool(inputs, div, padding)
        self.layer_outs.append(output)


    def add_batchnorm(self):
        with self.graph:
            output = batchNorm(inputs):
        self.layer_outs.append(output)


    def initialise(self):
        with self.graph:
            # initialise

    
    def add_summary(self, name, value, stype):
        """Function to define summary
    
        A name together with a value and type of summary is
        provided to the function. The summary is added to the
        graph.

        :param scalar_sd: Scalar values to be saved as summaries.
        :param histogram_sd: Values to be stored as histograms.
        :param image_sd: Values to be stored as images.
        :returns merged_summary: The merged summary of saved items.
        TODO: Add other summaries such as distributions, etc.
        """
        assert stype in ['scalar', 'histogram', 'image'], (
            'Expected type to be either scalar, histogram or image. '
            'Got {0}'.format(stype))

        with self.graph:
            if stype='scalar:
                tf.summary.scalar(name, value)
            elif stype='histogram':
                tf.summary.histogram(name, value)
            else: 
                tf.summary.image(name, value)

    
    def merge_summaries(self):
        with self.graph:
            merged = tf.summary.merge_all()
        return merged

    def train(self, train_set, eval_set, ):
        # train function


    def save_model(self):
        # save model


    def load_model(self):
        # load model

    
    def save_layers_wav(self):
        # save layers wave

    def sound_layer(self, layer_num):
        # convert layer to wav
        # play wav file

    def test_graph(self, test_set_provider, results=['confusion'])
        # test graph with unseen dataset