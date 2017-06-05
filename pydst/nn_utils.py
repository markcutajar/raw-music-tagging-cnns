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


def ff_layer(inputs, szparams, nonlin=tf.nn.relu, batchNorm=False, keepProb=1):
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
    dense(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None
)
    return output


def add_conv1l(inputs, filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=tf.nn.relu, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regulizer=None, bias_regularizer=None, activity_regularizer=None, trainable=True, name=None, reuse=None, batch_norm=False, ):
    
def add_conv2l():



# TODO: Check function on the tensorflow API documentation
def maxpool_layer(inputs, div=(2,2)):
    """Function to perform max-pooling on the provided inputs
    
    :param inputs: Input tensor.
    :param div: Check on the tensorflow API documentation to understand the difference between ksize and strides 
    :return: Max-pooled layer.
    """
    return tf.nn.max_pool(inputs, ksize=[1, div[0], div[1], 1], strides=[1, div[0], div[1], 1], padding='SAME')

def avgpool_layer():

def batch_norm():
    

    
    
class tfgraph(object):

    def __init__(self, graph):
        self.graph=graph
        self.layer_outs = []


    def add_ffl(self,
                inputs,
                units,
                activation=None,
                use_bias=True,
                kernel_initializer=None,
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name=None,
                reuse=None
               ):
        """Function to add a single fully connected layer to the graph.

        :param inputs: 
               
        """
        with self.graph:
            output = tf.layers.dense(inputs,
                                     units,
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=None,
                                     bias_initializer=tf.zeros_initializer(),
                                     kernel_regularizer=None,
                                     bias_regularizer=None,
                                     activity_regularizer=None,
                                     trainable=True,
                                     name=None,
                                     reuse=None
                                    )
            self.layer_outs.append(output)

            
    def add_conv1l(self,
                   inputs,
                   filters,
                   kernel_size,
                   strides=1,
                   padding='valid',
                   data_format='channels_last',
                   dilation_rate=1,
                   activation=None,
                   use_bias=True,
                   kernel_initializer=None,
                   bias_initializer=tf.zeros_initializer(),
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   activity_regularizer=None,
                   trainable=True,
                   name=None,
                   reuse=None
                  ):
        """Function to add a conv1d layer to the graph.

        :param inputs: 
               
        """       
        with self.graph:
            output = tf.layers.conv1d(inputs,
                                      filters,
                                      kernel_size,
                                      strides=1,
                                      padding='valid',
                                      data_format='channels_last',
                                      dilation_rate=1,
                                      activation=None,
                                      use_bias=True,
                                      kernel_initializer=None,
                                      bias_initializer=tf.zeros_initializer(),
                                      kernel_regularizer=None,
                                      bias_regularizer=None,
                                      activity_regularizer=None,
                                      trainable=True,
                                      name=None,
                                      reuse=None
                                     )  
            self.layer_outs.append(output)


    def add_conv2l(self,  
                   inputs,
                   filters,
                   kernel_size,
                   strides=(1, 1),
                   padding='valid',
                   data_format='channels_last',
                   dilation_rate=(1, 1),
                   activation=None,
                   use_bias=True,
                   kernel_initializer=None,
                   bias_initializer=tf.zeros_initializer(),
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   activity_regularizer=None,
                   trainable=True,
                   name=None,
                   reuse=None
                  ):
        """Function to add a conv2d layer to the graph.

        :param inputs: 
               
        """    
        with self.graph:
            output = tf.layers.conv2d(inputs,
                                      filters,
                                      kernel_size,
                                      strides=(1, 1),
                                      padding='valid',
                                      data_format='channels_last',
                                      dilation_rate=(1, 1),
                                      activation=None,
                                      use_bias=True,
                                      kernel_initializer=None,
                                      bias_initializer=tf.zeros_initializer(),
                                      kernel_regularizer=None,
                                      bias_regularizer=None,
                                      activity_regularizer=None,
                                      trainable=True,
                                      name=None,
                                      reuse=None
                                     )
            self.layer_outs.append(output)


    def add_maxpool1(self, 
                    inputs,
                    pool_size,
                    strides,
                    padding='valid',
                    data_format='channels_last',
                    name=None
                   ):
        """Function to add a maxpool layer to the graph.

        :param inputs: 
               
        """
        with self.graph:
            output = tf.layers.max_pooling1d(inputs,
                                         pool_size,
                                         strides,
                                         padding='valid',
                                         data_format='channels_last',
                                         name=None
                                        )
            self.layer_outs.append(output)

            
    def add_avgpool1(self,
                     inputs,
                     pool_size,
                     strides,
                     padding='valid',
                     data_format='channels_last',
                     name=None
                    ):
        """Function to add a avg pool layer to the graph.

        :param inputs: 
               
        """
        with self.graph:
            output = tf.layers.average_pooling1d(inputs,
                                                 pool_size,
                                                 strides,
                                                 padding='valid',
                                                 data_format='channels_last',
                                                 name=None
                                                )
            self.layer_outs.append(output)


    def add_batchnorm(self,
                      inputs,
                      axis=-1,
                      momentum=0.99,
                      epsilon=0.001,
                      center=True,
                      scale=True,
                      beta_initializer=tf.zeros_initializer(),
                      gamma_initializer=tf.ones_initializer(),
                      moving_mean_initializer=tf.zeros_initializer(),
                      moving_variance_initializer=tf.ones_initializer(),
                      beta_regularizer=None,
                      gamma_regularizer=None,
                      training=False,
                      trainable=True,
                      name=None,
                      reuse=None
                     ):
        """Function to add a batch normalization layer to the graph.

        :param inputs: 
               
        """
        with self.graph:
            output = tf.layers.batch_normalization(inputs,
                                                   axis=-1,
                                                   momentum=0.99,
                                                   epsilon=0.001,
                                                   center=True,
                                                   scale=True,
                                                   beta_initializer=tf.zeros_initializer(),
                                                   gamma_initializer=tf.ones_initializer(),
                                                   moving_mean_initializer=tf.zeros_initializer(),
                                                   moving_variance_initializer=tf.ones_initializer(),
                                                   beta_regularizer=None,
                                                   gamma_regularizer=None,
                                                   training=False,
                                                   trainable=True,
                                                   name=None,
                                                   reuse=None
                                                  ):
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
            lout = ff_layer(din, sz, nonlin=nonlin[lidx], batchNorm=batchNorm, keepProb=keepProb)
            layer_outs.append(lout)
            
    return layer_outs