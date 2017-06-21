"""Graph functions to test the raw music tagging."""

import tensorflow as tf

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'


def dielemanschrauwen256(mode,
                         data_batch,
                         targets_batch,
                         learning_rate=0.1):
    
    outputs = {}

    name = 'strided-conv'
    outputs[name] = tf.layers.conv1d(data_batch, 1, 256, strides=256, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'conv-1'
    outputs[name] = tf.layers.conv1d(outputs['strided-conv'], 32, 8, strides=1, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'maxp-1'
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-1'], pool_size=4, strides=1, name=name)

    name = 'conv-2'
    outputs[name] = tf.layers.conv1d(outputs['maxp-1'], 32, 8, strides=1, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'maxp-2'
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-2'], pool_size=4, strides=1, name=name)


    name = 'fcl-1'
    outputs[name] = tf.layers.dense(tf.reshape(outputs['maxp-2'],
                                               [int(outputs['maxp-2'].shape[0]), -1]),
                                    100, activation=tf.nn.relu, name=name)


    name = 'fcl-2'
    outputs[name] = tf.layers.dense(outputs['fcl-1'], 50, activation=tf.identity, name=name)
    logits = outputs[name]

    #prediction_values = tf.nn.sigmoid(logits, name='probabilities')
    predictions = tf.nn.relu(logits, name='prob2tag')

    if mode in (TRAIN, EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()
        name = 'error'
        with tf.name_scope(name):
            error = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=targets_batch, logits=logits)

        if mode in (TRAIN):
            train_step = tf.train.AdamOptimizer().minimize(error, global_step=global_step)
            return train_step, global_step, error

        else:
            streaming_metrics = {
                'false_negatives': tf.contrib.metrics.streaming_false_negatives(
                    predictions, targets_batch, name='false_negatives'),
                'false_positives': tf.contrib.metrics.streaming_false_positives(
                    predictions, targets_batch, name='false_positives'),
                'true_positives': tf.contrib.metrics.streaming_true_positives(
                    predictions, targets_batch, name='true_positives'),
                'true_negatives': tf.contrib.metrics.streaming_true_negatives(
                    predictions, targets_batch, name='true_negatives'),
                'precision': tf.contrib.metrics.streaming_precision(
                    predictions, targets_batch, name='precision'),
                'auc': tf.contrib.metrics.streaming_auc(
                    predictions, targets_batch, name='auc'),
                'accuracy': tf.contrib.metrics.streaming_accuracy(
                    predictions, targets_batch, name='accuracy')
            }
            return streaming_metrics, error

    elif mode in (PREDICT):
        return predictions #, prediction_values
    else:
        raise ValueError('Mode not found!')