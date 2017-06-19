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

    name = 'conv-1'
    outputs[name] = tf.layers.conv1d(data_batch, 10, 700, strides=10, activation=tf.nn.relu,
                                     kernel_initializer=tf.truncated_normal_initializer,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     bias_initializer=tf.truncated_normal_initializer,
                                     name=name)

    name = 'maxp-1'
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-1'], pool_size=3, strides=3, name=name)

    name = 'conv-2'
    outputs[name] = tf.layers.conv1d(outputs['maxp-1'], 20, 300, strides=5, activation=tf.nn.relu,
                                     kernel_initializer=tf.truncated_normal_initializer,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     bias_initializer=tf.truncated_normal_initializer,
                                     name=name)

    name = 'maxp-2'
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-2'], pool_size=3, strides=3, name=name)

    name = 'conv-3'
    outputs[name] = tf.layers.conv1d(outputs['maxp-2'], 30, 100, strides=1, activation=tf.nn.relu,
                                     kernel_initializer=tf.truncated_normal_initializer,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     bias_initializer=tf.truncated_normal_initializer,
                                     name=name)

    name = 'maxp-3'
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-3'], pool_size=3, strides=3, name=name)

    name = 'conv-4'
    outputs[name] = tf.layers.conv1d(outputs['maxp-3'], 40, 30, strides=1, activation=tf.nn.relu,
                                     kernel_initializer=tf.truncated_normal_initializer,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     bias_initializer=tf.truncated_normal_initializer,
                                     name=name)

    name = 'maxp-4'
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-4'], pool_size=3, strides=3, name=name)

    name = 'conv-5'
    outputs[name] = tf.layers.conv1d(outputs['maxp-4'], 50, 10, strides=1, activation=tf.nn.relu,
                                     kernel_initializer=tf.truncated_normal_initializer,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     bias_initializer=tf.truncated_normal_initializer,
                                     name=name)

    name = 'maxp-5'
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-5'], pool_size=3, strides=3, name=name)

    name = 'layer-1'
    outputs[name] = tf.layers.dense(tf.reshape(outputs['maxp-5'], [int(outputs['maxp-5'].shape[0]), -1]),
                                    2000, activation=tf.nn.relu, name=name)

    name = 'layer-2'
    outputs[name] = tf.layers.dense(outputs['layer-1'], 2000, activation=tf.nn.relu, name=name)

    name = 'layer-3'
    outputs[name] = tf.layers.dense(outputs['layer-2'], 1000, activation=tf.nn.relu, name=name)

    name = 'layer-4'
    outputs[name] = tf.layers.dense(outputs['layer-3'], 200, activation=tf.nn.relu, name=name)

    name = 'layer-5'
    outputs[name] = tf.layers.dense(outputs['layer-4'], 50, activation=tf.identity, name=name)
    last_output = outputs[name]

    if mode in (TRAIN, EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()

        name = 'error'
        with tf.name_scope(name):
            errors[name] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets_batch, logits=last_output, name=name
            ))


    if mode == TRAIN:
        train_step = tf.train.AdamOptimizer().minimize(errors[name], global_step=global_step)
        return train_step, global_step, errors[name]

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
                last_output, targets_batch),
            'absolute error': tf.contrib.metrics.streaming_mean_absolute_error(
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