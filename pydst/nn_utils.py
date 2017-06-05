# -*- coding: utf-8 -*-
""" Neural Network Utilities.

This module provides utilities classes that are handy in neural network construction.
"""

import tensorflow as tf
import numpy as np

    
class tfgraph(object):

    def __init__(self, graph, sess, inputs_size, targets_size, input_layer_name, targets_name):
        """Create new tfgraph object.
        
        Explain
        :param
        """
        self.graph = graph
        with self.graph:
            with tf.name_scope('inputs and targets placeholders'):
                
                inputs = tf.placeholder(tf.float32, inputs_size, 'inputs')
                targets = tf.placeholder(tf.float32, targets_size, 'targets')
            
            self.layer_outs = {}
            self.layer_outs[input_layer_name] = inputs
            self.targets = targets
            
            self.last_name = input_layer_name

            self.errors = {}
            self.accuracies = {}
        
            self.session = sess
def glorot_std(params):
    """Function to return the STD    
    :param params: The input and output layer sizes.
    :returns Standard deviation according to Glorot and Bengio.
    """
    return (2./(params[0]+params[1]))**0.5
    
    
    def add_ffl(self,
                units,
                activation=None,
                use_bias=True,
                kernel_initializer=None,
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name,
                reuse=None
               ):
        """Function to add a single fully connected layer to the graph.

        :param inputs: 
               
        """
        with self.graph:
            with tf.name_scope(name):
                output = tf.layers.dense(inputs=self.layer_outs[self.last_name],
                                         units=units,
                                         activation=activation,
                                         use_bias=use_bias,
                                         kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer,
                                         bias_regularizer=bias_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         trainable=trainable,
                                         name=name,
                                         reuse=reuse
                                        )
            self.layer_outs[name] = output
            self.last_name = name
            
            
    def add_conv1l(self,
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
                   name,
                   reuse=None
                  ):
        """Function to add a conv1d layer to the graph.

        :param inputs: 
               
        """       
        with self.graph:
            with tf.name_scope(name):
                output = tf.layers.conv1d(inputs=self.layer_outs[self.last_name],
                                          filters=filters,
                                          kernel_size=kernel_size,
                                          strides=strides,
                                          padding=padding,
                                          data_format=data_format,
                                          dilation_rate=dilation_rate,
                                          activation=activation,
                                          use_bias=use_bias,
                                          kernel_initializer=kernel_initializer,
                                          bias_initializer=bias_initializer,
                                          kernel_regularizer=kernel_regularizer,
                                          bias_regularizer=bias_regularizer,
                                          activity_regularizer=activity_regularizer,
                                          trainable=trainable,
                                          name=name,
                                          reuse=reuse
                                         )  
            self.layer_outs[name] = output
            self.last_name = name
            

    def add_conv2l(self,
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
                   name,
                   reuse=None
                  ):
        """Function to add a conv2d layer to the graph.

        :param inputs: 
               
        """    
        with self.graph:
            with tf.name_scope(name):
                output = tf.layers.conv2d(inputs=self.layer_outs[self.last_name],
                                          filters=filters,
                                          kernel_size=kernel_size,
                                          strides=strides,
                                          padding=padding,
                                          data_format=data_format,
                                          dilation_rate=dilation_rate,
                                          activation=activation,
                                          use_bias=use_bias,
                                          kernel_initializer=kernel_initializer,
                                          bias_initializer=bias_initializer,
                                          kernel_regularizer=kernel_regularizer,
                                          bias_regularizer=bias_regularizer,
                                          activity_regularizer=activity_regularizer,
                                          trainable=trainable,
                                          name=name,
                                          reuse=reuse
                                         )
            self.layer_outs[name] = output
            self.last_name = name
            

    def add_maxpool1(self,
                     pool_size,
                     strides,
                     padding='valid',
                     data_format='channels_last',
                     name
                    ):
        """Function to add a maxpool layer to the graph.

        :param inputs: 
               
        """
        with self.graph:
            with tf.name_scope(name):
                output = tf.layers.max_pooling1d(inputs=self.layer_outs[self.last_name],
                                                 pool_size=pool_size,
                                                 strides=strides,
                                                 padding=padding,
                                                 data_format=data_format,
                                                 name=name
                                                )
            self.layer_outs[name] = output
            self.last_name = name
            
            
    def add_avgpool1(self,
                     pool_size,
                     strides,
                     padding='valid',
                     data_format='channels_last',
                     name
                    ):
        """Function to add a avg pool layer to the graph.

        :param inputs: 
               
        """
        with self.graph:
            with tf.name_scope(name):
                output = tf.layers.average_pooling1d(inputs=self.layer_outs[self.last_name],
                                                     pool_size=pool_size,
                                                     strides=strides,
                                                     padding=padding,
                                                     data_format=data_format,
                                                     name=name
                                                    )
            self.layer_outs[name] = output
            self.last_name = name
            

    def add_batchnorm(self,
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
                      name,
                      reuse=None
                     ):
        """Function to add a batch normalization layer to the graph.

        :param inputs: 
               
        """
        with self.graph:
            with tf.name_scope(name):
                output = tf.layers.batch_normalization(inputs=self.layer_outs[self.last_name]
                                                       axis=axis,
                                                       momentum=momentum
                                                       epsilon=epsilon
                                                       center=center,
                                                       scale=scale,
                                                       beta_initializer=beta_initializer,
                                                       gamma_initializer=gamma_initializer,
                                                       moving_mean_initializer=moving_mean_initializer,
                                                       moving_variance_initializer=moving_variance_initializer,
                                                       beta_regularizer=beta_regularizer,
                                                       gamma_regularizer=gamma_regularizer,
                                                       training=training,
                                                       trainable=trainable,
                                                       name=name
                                                       reuse=reuse
                                                      )
            self.layer_outs[name] = output
            self.last_name = name
            

    def add_dropout(rate=0.5,
                    noise_shape=None,
                    seed=None,
                    training=False,
                    name
                   ):
        """Function to add a dropout layer to the graph.

        :param inputs: 
               
        """
        with self.graph:
            with tf.name_scope(name):
                output = tf.layers.dropout(inputs=self.layer_outs[self.last_name]
                                           rate=rate
                                           noise_shape=noise_shape,
                                           seed=seed,
                                           training=training,
                                           name=name
                                          )
            self.layer_outs[name] = output
            self.last_name = name
            
            
    def residual_layer(self, residual_layer_name, activation=None, name):
        """Function to add a residual layer to the graph.
        
        :param inputs: 
               
        """   
        inputs = self.layer_outs[self.last_name]
        residues = self.layer_out[residual_layer_name]
        
        if inputs.shape != residues.shape:
            raise ValueError('Inputs {} and residues {} must have the same dimension for residual layer'
                            .format(inputs.shape, residues.shape))
       
        with self.graph:  
            with tf.name_scope(name):
                
                output = inputs + residues # Tensors do not add
                if activation is not None:
                    output = activation(output)
                
            self.layer_outs[name] = output
            self.last_name = name
            
    
    def accuracy(self, output_layer_name, name):
        """Function to calculate accuracy of certain output.

        :param inputs: 
               
        """
        with self.graph():
            with tf.name_scope(name):
                accuracy = tf.metrics.accuracy(labels=self.targets,
                                               predictions=self.layer_outs[output_layer_name],
                                               weights=None,
                                               metrics_collections=None,
                                               updates_collections=None,
                                               name=name
                                              )
            self.accuracies[name] = accuracy
        
        
    def get_accuracy(self, name):
        """Return an accuracy with a particular name."""
        return self.accuracies[name]
    
    
    def error_softmax(self, output_layer_name, name):
        """Function to calculate softmax error of certain output.

        :param inputs: 
               
        """
        with self.graph:
            with tf.name_scope(name):
                error = tf.losses.softmax_cross_entropy(onehot_labels=self.targets,
                                                        logits=self.layer_outs[output_layer_name],
                                                        weights=1.0,
                                                        label_smoothing=0,
                                                        scope=None,
                                                        loss_collection=tf.GraphKeys.LOSSES
                                                       )
            self.errors[name] = error
        
        
    def error_sigmoid(self, output_layer_name, name):
        """Function to calculate sigmoid error of certain output.

        :param inputs: 
               
        """
        with self.graph:
            with tf.name_scope(name):
                error = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.targets,
                                                        logits=self.logits,
                                                        weights=1.0,
                                                        label_smoothing=0,
                                                        scope=None,
                                                        loss_collection=tf.GraphKeys.LOSSES
                                                       )
            self.errors[name] = error
    
    
    def get_error(self, name):
        """Return an error with a particular name."""
        return self.errors[name]
    
    
    def set_train(self, optimizer=tf.train.AdamOptimizer(), name_error):
        """Set train function."""
        with self.graph:
            with tf.name_scope('train-' + name_error):
                train_step = optimizer.minimize(self.error)
        
        
    def initialise(self):
        """Initialise graph variables."""
        with self.graph:
            self.session.run(tf.global_variables_initializer())
            
            
    def train(self, data_provider, pre_processor=None, error_name, accuracy_name, epochs):
        """Function to train the graph.
        
        Please note that this function gets information by an iterable.
        
        :param inputs
        """
        measured_errors = []
        measured_accs = []
        
        for epoch in range(epochs): 
            
            run_error = 0.
            run_acc = 0.
            
            for input_batch, target_batch in data_provider:
                
                if preprocessor is not None:
                    input_batch = preprocessor(input_batch)
                    
                [train_steps, batch_error, accuracy_error] = sess.run(
                    [_, self.errors[error_name], self.accuracies[accuracy_name]],
                    feed_dict={inputs: input_batch, targets: target_batch})
                
                run_error += batch_error
                run_acc += batch_acc
                
            run_error /= provider.batch_size
            run_acc /= provider.batch_size
            
            measured_errors.append(run_error)
            measured_accs.append(run_acc)
        
        return measured_errors, measured_accs
    
    
    def train_eval(self, train_data, valid_data, pre_processor=None, error_name, accuracy_name, epochs=100, valid_step=10):
        """Function to train and evaluate the graph.
        
        Please note that this function gets information by an iterable.
        
        :param inputs
        """
        train_errors = []
        train_accs = []
        valid_errors = []
        valid_accs = []
        
        for epoch in range(epochs): 
            
            run_error = 0.
            run_acc = 0.
            
            for input_batch, target_batch in train_data:
                
                if preprocessor is not None:
                    input_batch = preprocessor(input_batch)
                    
                [_, batch_error, accuracy_error] = sess.run(
                    [train_step, self.errors[error_name], self.accuracies[accuracy_name]],
                    feed_dict={inputs: input_batch, targets: target_batch})
                
                run_error += batch_error
                run_acc += batch_acc
                
            run_error /= provider.batch_size
            run_acc /= provider.batch_size
            
            train_errors.append(run_error)
            train_accs.append(run_acc)
            
            if (epoch + 1) % valid_step == 0:
                
                run_error = 0.
                run_acc = 0.

                for input_batch, target_batch in valid_data:

                    if preprocessor is not None:
                        input_batch = preprocessor(input_batch)

                    [batch_error, accuracy_error] = sess.run(
                        [self.errors[error_name], self.accuracies[accuracy_name]],
                        feed_dict={inputs: input_batch, targets: target_batch})

                    run_error += batch_error
                    run_acc += batch_acc

                run_error /= provider.batch_size
                run_acc /= provider.batch_size

                valid_errors.append(run_error)
                valid_accs.append(run_acc)

        return train_errors, train_accs, valid_errors, valid_accs
    
    
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

    
    def save_model(self, path):
        with self.session:
            model_saver = tf.train.Saver()
            model_saver.save(self.session, path)        
            model_saver.export_meta_graph(path + '.meta')


    def load_model(self, path):
        with self.session:
            model_loader = tf.train.import_meta_graph(path + '.meta')
            model_loader.restore(self.session, path)

    
    def single_test(self, single_sample, sample_targets):
        """Function test on a single instance and return different layer outputs.

        :param inputs: 
               
        """ 
        name_list, value_list = []
        with self.graph:
            for key, value in self.layer_outs.iteritems():
                name_list.append(key)
                value_list.append(value)
            
            for key, value in self.errors.iteritems():
                name_list.append(key)
                value_list.append(value)
                
            for key, value in self.accuracies.iteritems():
                name_list.append(key)
                value_list.append(value)
            
            values = self.session.run(value_list, feed_dict={inputs: single_sample, targets: sample_targets})
            layer_out_dict = dict(zip(name_list, values))
        
        return layer_out_dict