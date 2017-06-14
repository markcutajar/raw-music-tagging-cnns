"""Graph functions to test the raw music tagging."""

import tensorflow as tf
from tensorflow.python.ops import string_ops



def basic_conv(data_batch,
               targets_batch,
               learning_rate=0.1):

    """
    
    Args:
        features (Tensor): Matrix input tensor
        labels (Tensor): Class label tensor
        learning_rate (float): Learning rate for AdaDeltaOptimizer
        
    Returns:
        summary_op: summary to be found
        train_step: Train step
        global_step: Global step training
    """
    
    outputs = {}
    errors = {}
    accs = {}
    
    name = 'conv1'
    outputs[name] = tf.layers.conv1d(data_batch, 4, 200, strides=50, activation=tf.nn.elu, 
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)
        
    name = 'conv2'
    outputs[name] = tf.layers.conv1d(outputs['conv1'], 6, 100, strides=20, activation=tf.nn.elu, 
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)    
        
    name = 'conv3'
    outputs[name] = tf.layers.conv1d(outputs['conv2'], 8, 10, strides=5, activation=tf.nn.elu, 
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name) 
        
    name = 'layer-1'
    outputs[name] = tf.layers.dense(tf.reshape(outputs['conv3'], [int(outputs['conv3'].shape[0]), -1]), 
                                    200, activation=tf.nn.elu)
        
    name = 'layer-2'
    outputs[name] = tf.layers.dense(outputs['layer-1'], 
                                    50, activation=tf.nn.elu)
                                        
    last_output = outputs['layer-2']
    
    global_step = tf.contrib.framework.get_or_create_global_step()
    
    name = 'error'
    with tf.name_scope(name):
        errors[name] = tf.losses.sigmoid_cross_entropy(multi_class_labels=targets_batch, logits=last_output)
    tf.summary.scalar(name, errors[name])
    
    train_step = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(errors[name], global_step=global_step)
       
        
    name = 'accuracy'
    with tf.name_scope(name):
        accs[name] = tf.contrib.metrics.streaming_accuracy(last_output, targets_batch)
    tf.summary.scalar(name, accs[name])
    
    name = 'auroc'
    with tf.name_scope(name):
        accs[name] = tf.contrib.metrics.streaming_auc(last_output, targets_batch)
    tf.summary.scalar(name, accs[name])

    summary_op = tf.summary.merge_all()
    
    return summary_op, train_step, global_step