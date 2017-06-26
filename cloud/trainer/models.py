"""Graph functions to test the raw music tagging."""

import tensorflow as tf

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'

# ---------------------------------------------------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------------------------------------------------
def controller(function_name,
               mode,
               data_batch,
               targets_batch,
               learning_rate=0.1,
               window=False):

    # Load model
    model = globals()[function_name]

    # Get logits depending if windowed inputs or not
    if window:
        with tf.name_scope('windowed_model'):
            logits_array = tf.map_fn(lambda w: model(w),
                                     elems=data_batch,
                                     back_prop=True,
                                     name='MapModels')

            logits = tf.concat(logits_array, axis=0, name='windowLogits')
            logits = tf.reduce_mean(logits, axis=0, name='averageLogits')
    else:
            # Get logits with one call
            logits = model(data_batch)

    # Get tag predictions
    with tf.name_scope('preds') as scope:
        prediction_values = tf.nn.sigmoid(logits, name='probs')
        predictions = tf.round(prediction_values)

    # Get errors and accuracies
    if mode in (TRAIN, EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()
        name = 'error'
        with tf.name_scope(name):
            error = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=targets_batch, logits=logits)

        if mode in TRAIN:
            with tf.name_scope('train') as scope:
                train_step = tf.train.AdamOptimizer().minimize(error, global_step=global_step)
            return train_step, global_step, error

        else:
            with tf.name_scope('metrics') as scope:
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
                    'aucroc': tf.contrib.metrics.streaming_auc(
                        predictions, targets_batch, name='aucroc'),
                    'aucpr': tf.contrib.metrics.streaming_auc(
                        predictions, targets_batch, curve='PR', name='aucpr')
                }
                scalar_metrics = {
                    'evaluation_error': error
                }

                metrics = {
                    'stream': streaming_metrics,
                    'scalar': scalar_metrics,
                    'perclass': perclass_metrics(predictions, targets_batch)
                }
            return metrics

    elif mode in PREDICT:
        return predictions, prediction_values
    else:
        raise ValueError('Mode not found!')


def perclass_metrics(predictions, targets_batch):

    perclass_dict = {}
    predictions_per_tag_list = tf.unstack(predictions, axis=1)
    targets_per_tag_list = tf.unstack(targets_batch, axis=1)

    
    for idx, pred_tag in enumerate(predictions_per_tag_list):
        perclass_dict[str(idx)+'_false_negatives'] = tf.contrib.metrics.streaming_false_negatives(
            pred_tag, targets_per_tag_list[idx], name='false_negatives')

        perclass_dict[str(idx)+'_false_positives'] = tf.contrib.metrics.streaming_false_positives(
            pred_tag, targets_per_tag_list[idx], name='false_positives')

        perclass_dict[str(idx)+'_precision'] = tf.contrib.metrics.streaming_precision(
            pred_tag, targets_per_tag_list[idx], name='precision')

        perclass_dict[str(idx)+'_aucroc'] = tf.contrib.metrics.streaming_auc(
            pred_tag, targets_per_tag_list[idx], name='aucroc')

        perclass_dict[str(idx)+'_aucpr'] = tf.contrib.metrics.streaming_auc(
            pred_tag, targets_per_tag_list[idx], curve='PR', name='aucpr')

    return perclass_dict

# ---------------------------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------------------------

def ds256ra_t50(data_batch):
    outputs = {}
    names = []
    name = 'strided-conv'
    names.append(name)
    outputs[name] = tf.layers.conv1d(data_batch, 1, 256, strides=256, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'conv-1'
    names.append(name)
    outputs[name] = tf.layers.conv1d(outputs['strided-conv'], 32, 8, strides=1, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'maxp-1'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-1'], pool_size=4, strides=4, name=name)

    name = 'conv-2'
    names.append(name)
    outputs[name] = tf.layers.conv1d(outputs['maxp-1'], 32, 8, strides=1, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'maxp-2'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-2'], pool_size=4, strides=4, name=name)

    name = 'flatten'
    names.append(name)
    outputs[name] = tf.reshape(outputs['maxp-2'], [int(outputs['maxp-2'].shape[0]), -1])

    name = 'fcl-1'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['flatten'], 100, activation=tf.nn.relu, name=name)

    name = 'fcl-2'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['fcl-1'], 50, activation=tf.identity, name=name)
    return outputs[name]

def ds256rb_t50(data_batch):
    outputs = {}
    names = []
    name = 'strided-conv'
    names.append(name)
    outputs[name] = tf.layers.conv1d(data_batch, 1, 64, strides=64, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'conv-1'
    names.append(name)
    outputs[name] = tf.layers.conv1d(outputs['strided-conv'], 32, 8, strides=1, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'maxp-1'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-1'], pool_size=4, strides=4, name=name)

    name = 'conv-2'
    names.append(name)
    outputs[name] = tf.layers.conv1d(outputs['maxp-1'], 32, 8, strides=1, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'maxp-2'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-2'], pool_size=4, strides=4, name=name)

    name = 'flatten'
    names.append(name)
    outputs[name] = tf.reshape(outputs['maxp-2'], [int(outputs['maxp-2'].shape[0]), -1])

    name = 'fcl-1'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['flatten'], 100, activation=tf.nn.relu, name=name)

    name = 'fcl-2'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['fcl-1'], 50, activation=tf.identity, name=name)
    return outputs[name]

def ds256rc_t50(data_batch):
    outputs = {}
    names = []
    name = 'strided-conv'
    names.append(name)
    outputs[name] = tf.layers.conv1d(data_batch, 10, 64, strides=64, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'conv-1'
    names.append(name)
    outputs[name] = tf.layers.conv1d(outputs['strided-conv'], 32, 8, strides=1, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'maxp-1'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-1'], pool_size=4, strides=4, name=name)

    name = 'conv-2'
    names.append(name)
    outputs[name] = tf.layers.conv1d(outputs['maxp-1'], 32, 8, strides=1, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'maxp-2'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-2'], pool_size=4, strides=4, name=name)

    name = 'flatten'
    names.append(name)
    outputs[name] = tf.reshape(outputs['maxp-2'], [int(outputs['maxp-2'].shape[0]), -1])

    name = 'fcl-1'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['flatten'], 100, activation=tf.nn.relu, name=name)

    name = 'fcl-2'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['fcl-1'], 50, activation=tf.identity, name=name)
    return outputs[name]

def ds256rd_t50(data_batch):
    outputs = {}
    names = []
    name = 'strided-conv'
    names.append(name)
    outputs[name] = tf.layers.conv1d(data_batch, 10, 16, strides=16, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'conv-1'
    names.append(name)
    outputs[name] = tf.layers.conv1d(outputs['strided-conv'], 32, 8, strides=1, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'maxp-1'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-1'], pool_size=4, strides=4, name=name)

    name = 'conv-2'
    names.append(name)
    outputs[name] = tf.layers.conv1d(outputs['maxp-1'], 32, 8, strides=1, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'maxp-2'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-2'], pool_size=4, strides=4, name=name)

    name = 'flatten'
    names.append(name)
    outputs[name] = tf.reshape(outputs['maxp-2'], [int(outputs['maxp-2'].shape[0]), -1])

    name = 'fcl-1'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['flatten'], 100, activation=tf.nn.relu, name=name)

    name = 'fcl-2'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['fcl-1'], 50, activation=tf.identity, name=name)
    return outputs[name]

def ds256rd_t29(data_batch):
    outputs = {}
    names = []
    name = 'strided-conv'
    names.append(name)
    outputs[name] = tf.layers.conv1d(data_batch, 10, 16, strides=16, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'conv-1'
    names.append(name)
    outputs[name] = tf.layers.conv1d(outputs['strided-conv'], 32, 8, strides=1, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'maxp-1'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-1'], pool_size=4, strides=4, name=name)

    name = 'conv-2'
    names.append(name)
    outputs[name] = tf.layers.conv1d(outputs['maxp-1'], 32, 8, strides=1, activation=tf.nn.relu,
                                     kernel_regularizer=tf.layers.batch_normalization,
                                     name=name)

    name = 'maxp-2'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-2'], pool_size=4, strides=4, name=name)

    name = 'flatten'
    names.append(name)
    outputs[name] = tf.reshape(outputs['maxp-2'], [int(outputs['maxp-2'].shape[0]), -1])

    name = 'fcl-1'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['flatten'], 100, activation=tf.nn.relu, name=name)

    name = 'fcl-2'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['fcl-1'], 29, activation=tf.identity, name=name)
    return outputs[name]

