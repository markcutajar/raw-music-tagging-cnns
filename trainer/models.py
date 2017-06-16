"""Graph functions to test the raw music tagging."""

import tensorflow as tf

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'


def convbasic(mode,
              data_batch,
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

    name = 'conv1'
    outputs[name] = tf.layers.conv1d(data_batch, 4, 100, strides=20, activation=tf.nn.elu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)
    name = 'conv2'
    outputs[name] = tf.layers.conv1d(outputs['conv1'], 6, 100, strides=5, activation=tf.nn.elu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)
    name = 'conv3'
    outputs[name] = tf.layers.conv1d(outputs['conv2'], 8, 10, strides=1, activation=tf.nn.elu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)
    name = 'layer-1'
    outputs[name] = tf.layers.dense(tf.reshape(outputs['conv3'], [int(outputs['conv3'].shape[0]), -1]), 
                                    200, activation=tf.nn.elu)
    name = 'layer-2'
    outputs[name] = tf.layers.dense(outputs['layer-1'], 
                                    50, activation=tf.nn.elu)
    last_output = outputs['layer-2']

    if mode in (TRAIN, EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()

        name = 'error'
        with tf.name_scope(name):
            errors[name] = tf.losses.sigmoid_cross_entropy(multi_class_labels=targets_batch,
                                                           logits=last_output)
    if mode == TRAIN:
        train_step = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(errors[name],
                                                                                      global_step=global_step)
        return train_step, global_step

    if mode == EVAL:
        return {
            'false_negatives': tf.contrib.metrics.streaming_false_negatives(
                last_output, targets_batch),
            'false_positives': tf.contrib.metrics.streaming_false_positives(
                last_output, targets_batch),
            'true_positives': tf.contrib.metrics.streaming_true_positives(
                last_output, targets_batch),
            'true_negatives': tf.contrib.metrics.streaming_true_negatives(
                last_output, targets_batch),
            'precision': tf.contrib.metrics.streaming_precision(
                last_output, targets_batch)
        }


def mlpone(mode,
           data_batch,
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
    tf.logging.info('Starting layer definitions')
    name = 'layer-1'
    outputs[name] = tf.layers.dense(data_batch, 5000, activation=tf.nn.elu)

    name = 'layer-2'
    outputs[name] = tf.layers.dense(outputs['layer-1'], 1000, activation=tf.nn.elu)

    name = 'layer-3'
    outputs[name] = tf.layers.dense(outputs['layer-2'], 50, activation=tf.nn.elu)
    last_output = outputs['layer-3']


    if mode in (TRAIN, EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()

        name = 'error'
        with tf.name_scope(name):
            errors[name] = tf.losses.sigmoid_cross_entropy(multi_class_labels=targets_batch,
                                                           logits=last_output)

    if mode == TRAIN:
        train_step = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(errors[name],
                                                                                      global_step=global_step)
        return train_step, global_step

    if mode == EVAL:
        return {
            'false_negatives': tf.contrib.metrics.streaming_false_negatives(
                last_output, targets_batch),
            'false_positives': tf.contrib.metrics.streaming_false_positives(
                last_output, targets_batch),
            'true_positives': tf.contrib.metrics.streaming_true_positives(
                last_output, targets_batch),
            'true_negatives': tf.contrib.metrics.streaming_true_negatives(
                last_output, targets_batch),
            'precision': tf.contrib.metrics.streaming_precision(
                last_output, targets_batch)
        }