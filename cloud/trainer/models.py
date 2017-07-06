"""Graph functions to test the raw music tagging."""

import tensorflow as tf

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'
STME, SPM = 'STME', 'SPM'
# ---------------------------------------------------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------------------------------------------------


def controller(function_name,
               mode,
               data_batch,
               targets_batch,
               learning_rate=0.001,
               window=False):

    # Load model
    model = globals()[function_name]

    if window in STME and mode in EVAL:
        # model with tf.map_fn
        logits_array = tf.map_fn(lambda w: model(w, mode),
                                 elems=data_batch,
                                 back_prop=True,
                                 parallel_iterations=12,
                                 name='MapModels')

        logits = tf.concat(logits_array, axis=0, name='windowLogits')
        logits = tf.reduce_mean(logits, axis=0, name='averageLogits')
    elif window in SPM:
        # super pooled model
        # model with tf.map_fn
        logits_array = tf.map_fn(lambda w: model(w, mode),
                                 elems=data_batch,
                                 back_prop=True,
                                 parallel_iterations=12,
                                 name='MapModels')

        # Use super pooling model
        logits = superpoolA(tf.concat(logits_array, axis=1, name='mergingLogits'))
    else:
        # normal model
        logits = model(data_batch, mode)

    # Get tag predictions
    with tf.name_scope('predictions'):
        prediction_values = tf.nn.sigmoid(logits, name='probs')
        predictions = tf.round(prediction_values)

    # Get errors and accuracies
    if mode in (TRAIN, EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()
        name = 'error'
        with tf.name_scope(name):
            # error = tf.losses.sigmoid_cross_entropy(
            #     multi_class_labels=targets_batch, logits=logits, weights=tf.constant(3))
            class_weights = balancing_weights(50, 'log', 104)
            error = weighted_sigmoid_cross_entropy(logits=logits,
                                                   labels=targets_batch,
                                                   false_negatives_weight=10,
                                                   balancing_weights_vector=class_weights)
            tf.losses.add_loss(error)

        if mode in TRAIN:
            with tf.name_scope('train'):
                #train_step = tf.train.AdamOptimizer(learning_rate).minimize(error, global_step=global_step)
                train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(error, global_step=global_step)
            return train_step, global_step, error

        else:
            with tf.name_scope('metrics'):
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
                    #'aucpr': tf.contrib.metrics.streaming_auc(
                    #    predictions, targets_batch, curve='PR', name='aucpr')
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

        perclass_dict[str(idx) + '_true_positives'] = tf.contrib.metrics.streaming_true_positives(
            pred_tag, targets_per_tag_list[idx], name='true_positives')

        perclass_dict[str(idx) + '_true_negatives'] = tf.contrib.metrics.streaming_true_negatives(
            pred_tag, targets_per_tag_list[idx], name='true_negatives')

        perclass_dict[str(idx)+'_precision'] = tf.contrib.metrics.streaming_precision(
            pred_tag, targets_per_tag_list[idx], name='precision')

        perclass_dict[str(idx)+'_aucroc'] = tf.contrib.metrics.streaming_auc(
            pred_tag, targets_per_tag_list[idx], name='aucroc')

        #perclass_dict[str(idx)+'_aucpr'] = tf.contrib.metrics.streaming_auc(
        #    pred_tag, targets_per_tag_list[idx], curve='PR', name='aucpr')
    return perclass_dict

def weighted_sigmoid_cross_entropy(logits, labels, false_negatives_weight, balancing_weights_vector):

    fnw = tf.constant(false_negatives_weight, dtype=tf.float32)
    weighting = tf.maximum(1.0, tf.multiply(fnw, labels))
    false_p_coefficient = tf.multiply(tf.maximum(logits, tf.constant(0.0)), weighting)
    false_n_coefficient = tf.multiply(weighting, tf.multiply(logits, labels))
    log_exp = tf.log(tf.add(tf.constant(1.0),
                            tf.exp(tf.multiply(tf.constant(-1.0), tf.abs(logits)))))
    error_matrix = tf.add(tf.subtract(false_p_coefficient, false_n_coefficient), log_exp)
    error = tf.reduce_mean(error_matrix, axis=0)
    error = tf.multiply(error, balancing_weights_vector)
    error = tf.reduce_mean(error)
    return error


def balancing_weights(num_classes, function, factor):
    if function == 'log':
        class_weights = tf.constant(list(range(1, num_classes+1)), dtype=tf.float32)
        class_weights = tf.add(tf.multiply(tf.constant(factor, dtype=tf.float32),
                                     tf.log(class_weights)), tf.constant(1.0))
        return class_weights
    else:
        raise NotImplementedError('Function {} not implemented! Only Log!'.format(function))


def superpoolA(data):
    superpool_outputs = {}
    name = 'FCL1'
    superpool_outputs[name] = tf.layers.dense(data, 600, activation=tf.nn.elu, name=name)

    name = 'FCL2'
    superpool_outputs[name] = tf.layers.dense(superpool_outputs['FCL1'], 600, activation=tf.nn.elu, name=name)

    name = 'FCL3'
    superpool_outputs[name] = tf.layers.dense(superpool_outputs['FCL2'], 50, activation=tf.identity, name=name)
    return superpool_outputs[name]

# ---------------------------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------------------------

# Model proposed by Dieleman et al. using Raw data
# First Conv: FL256, FS256, FD1
# Output: 50 Neurons
# Structure: 3 Conv, 2 MLP
def ds256ra(data_batch, mode):
    filt_length = 256
    filt_depth = 1
    stride_length = 256
    output_size = 50
    outputs = {}

    name = 'CL1'
    with tf.variable_scope(name):
        output = tf.layers.conv1d(data_batch,
                                  filt_depth,
                                  filt_length,
                                  strides=stride_length,
                                  activation=None,
                                  name='conv')
        output = tf.layers.batch_normalization(output, name='batchNorm', training=(mode==TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'CL2'
    with tf.variable_scope(name):
        output = tf.layers.conv1d(outputs['CL1'], 32, 8, strides=1, activation=None, name='conv')
        output = tf.layers.batch_normalization(output, name='batchNorm', training=(mode==TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'MP1'
    outputs[name] = tf.layers.max_pooling1d(outputs['CL2'], pool_size=4, strides=4, name=name)

    name = 'CL3'
    with tf.variable_scope(name):
        output = tf.layers.conv1d(outputs['MP1'], 32, 8, strides=1, activation=None, name='conv')
        output = tf.layers.batch_normalization(output, name='batchNorm', training=(mode==TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'MP2'
    outputs[name] = tf.layers.max_pooling1d(outputs['CL3'], pool_size=4, strides=4, name=name)

    name = 'FLTN'
    outputs[name] = tf.reshape(outputs['MP2'], [int(outputs['MP2'].shape[0]), -1], name=name)

    name = 'FCL1'
    outputs[name] = tf.layers.dense(outputs['FLTN'], 1000, activation=tf.nn.elu, name=name)

    name = 'FCL2'
    outputs[name] = tf.layers.dense(outputs['FCL1'], output_size, activation=tf.identity, name=name)
    return outputs[name]

# Model proposed by Dieleman et al. using Raw data
# First Conv: FL256, FS16, FD6
# Output: 50 Neurons
# Structure: 3 Conv, 2 MLP
def ds256rb(data_batch, mode):
    filt_length = 256
    filt_depth = 6
    stride_length = 16
    output_size = 50
    outputs = {}

    name = 'CL1'
    with tf.variable_scope(name):
        output = tf.layers.conv1d(data_batch,
                                  filt_depth,
                                  filt_length,
                                  strides=stride_length,
                                  activation=None,
                                  name='conv')
        output = tf.layers.batch_normalization(output, name='batchNorm', training=(mode == TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'CL2'
    with tf.variable_scope(name):
        output = tf.layers.conv1d(outputs['CL1'], 16, 8, strides=1, activation=None, name='conv')
        output = tf.layers.batch_normalization(output, name='batchNorm', training=(mode == TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'MP1'
    outputs[name] = tf.layers.max_pooling1d(outputs['CL2'], pool_size=4, strides=4, name=name)

    name = 'CL3'
    with tf.variable_scope(name):
        output = tf.layers.conv1d(outputs['MP1'], 16, 8, strides=1, activation=None, name='conv')
        output = tf.layers.batch_normalization(output, name='batchNorm', training=(mode == TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'MP2'
    outputs[name] = tf.layers.max_pooling1d(outputs['CL3'], pool_size=4, strides=4, name=name)

    name = 'FLTN'
    outputs[name] = tf.reshape(outputs['MP2'], [int(outputs['MP2'].shape[0]), -1], name=name)

    name = 'FCL1'
    outputs[name] = tf.layers.dense(outputs['FLTN'], 1000, activation=tf.nn.elu, name=name)

    name = 'FCL2'
    outputs[name] = tf.layers.dense(outputs['FCL1'], output_size, activation=tf.identity, name=name)
    return outputs[name]

# Basic MLP for quick testing
def basic(data_batch, mode):
    output_size = 50
    outputs = {}
    name = 'FLTN'
    outputs[name] = tf.reshape(data_batch, [int(data_batch.shape[0]), -1], name=name)

    name = 'FCL1'
    outputs[name] = tf.layers.dense(outputs['FLTN'], 300, activation=tf.nn.elu, name=name)

    name = 'FCL2'
    outputs[name] = tf.layers.dense(outputs['FCL1'], output_size, activation=tf.identity, name=name)
    return outputs[name]

# Basic CNN for raw data with
# batch normalization.
def mkc_r(data_batch, mode):
    output_size=50
    outputs = {}
    name = 'CL1'
    with tf.variable_scope(name):
        output = tf.layers.conv1d(data_batch, 4, 16, strides=16, activation=None,  name='conv')
        output = tf.layers.batch_normalization(output, name='batchNorm', training=True)#(mode==TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'CL2'
    with tf.variable_scope(name):
        output = tf.layers.conv1d(outputs['CL1'], 8, 8, strides=4, activation=None, name='conv')
        output = tf.layers.batch_normalization(output, name='batchNorm', training=True)#(mode==TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'MP1'
    outputs[name] = tf.layers.max_pooling1d(outputs['CL2'], pool_size=2, strides=2, name=name)

    name = 'CL3'
    with tf.variable_scope(name):
        output = tf.layers.conv1d(outputs['MP1'], 12, 4, strides=1, activation=None, name='conv')
        output = tf.layers.batch_normalization(output, name='batchNorm', training=True)#(mode==TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'MP2'
    outputs[name] = tf.layers.max_pooling1d(outputs['CL3'], pool_size=2, strides=2, name=name)

    name = 'FLTN'
    outputs[name] = tf.reshape(outputs['MP2'], [int(outputs['MP2'].shape[0]), -1], name=name)

    name = 'FCL1'
    outputs[name] = tf.layers.dense(outputs['FLTN'], 1000, activation=tf.nn.elu, name=name)

    name = 'FCL2'
    outputs[name] = tf.layers.dense(outputs['FCL1'], 300, activation=tf.nn.elu, name=name)

    name = 'FCL3'
    outputs[name] = tf.layers.dense(outputs['FCL2'], output_size, activation=tf.identity, name=name)
    return outputs[name]

# Basic CNN for windowed raw data
# with batch normalization.
def mkc_rw(data_batch, mode):
    output_size=50
    outputs = {}
    name = 'CL1'
    with tf.variable_scope(name):
        output = tf.layers.conv1d(data_batch, 4, 64, strides=4, activation=None,  name='conv')
        #output = tf.layers.batch_normalization(output, name='batchNorm', training=(mode==TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'CL2'
    with tf.variable_scope(name):
        output = tf.layers.conv1d(outputs['CL1'], 8, 8, strides=2, activation=None, name='conv')
        #output = tf.layers.batch_normalization(output, name='batchNorm', training=(mode==TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'MP1'
    outputs[name] = tf.layers.max_pooling1d(outputs['CL2'], pool_size=2, strides=2, name=name)

    name = 'CL3'
    with tf.variable_scope(name):
        output = tf.layers.conv1d(outputs['MP1'], 12, 4, strides=1, activation=None, name='conv')
        #output = tf.layers.batch_normalization(output, name='batchNorm', training=(mode==TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'MP2'
    outputs[name] = tf.layers.max_pooling1d(outputs['CL3'], pool_size=2, strides=2, name=name)

    name = 'FLTN'
    outputs[name] = tf.reshape(outputs['MP2'], [int(outputs['MP2'].shape[0]), -1], name=name)

    name = 'FCL1'
    outputs[name] = tf.layers.dense(outputs['FLTN'], 1000, activation=tf.nn.elu, name=name)

    name = 'FCL2'
    outputs[name] = tf.layers.dense(outputs['FCL1'], 300, activation=tf.nn.elu, name=name)

    name = 'FCL3'
    outputs[name] = tf.layers.dense(outputs['FCL2'], output_size, activation=tf.identity, name=name)
    return outputs[name]

# Basic CNN with L2 regularization
# and no batch normalization.
def mkc_r_l2(data_batch, mode):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    actfn = tf.nn.elu
    output_size=50
    outputs = {}
    name = 'CL1'
    with tf.variable_scope(name):
        outputs[name] = tf.layers.conv1d(data_batch, 4, 16, strides=16, activation=actfn,  name='conv',
                                  kernel_regularizer=regularizer)

    name = 'CL2'
    with tf.variable_scope(name):
        outputs[name] = tf.layers.conv1d(outputs['CL1'], 8, 8, strides=4, activation=actfn, name='conv',
                                  kernel_regularizer=regularizer)

    name = 'MP1'
    outputs[name] = tf.layers.max_pooling1d(outputs['CL2'], pool_size=2, strides=2, name=name)

    name = 'CL3'
    with tf.variable_scope(name):
        outputs[name] = tf.layers.conv1d(outputs['MP1'], 12, 4, strides=1, activation=actfn, name='conv',
                                  kernel_regularizer=regularizer)

    name = 'MP2'
    outputs[name] = tf.layers.max_pooling1d(outputs['CL3'], pool_size=2, strides=2, name=name)

    name = 'FLTN'
    outputs[name] = tf.reshape(outputs['MP2'], [int(outputs['MP2'].shape[0]), -1], name=name)

    name = 'FCL1'
    outputs[name] = tf.layers.dense(outputs['FLTN'], 6000, activation=tf.nn.elu, name=name)

    name = 'FCL2'
    outputs[name] = tf.layers.dense(outputs['FCL1'], 2000, activation=tf.nn.elu, name=name)

    name = 'FCL3'
    outputs[name] = tf.layers.dense(outputs['FCL2'], output_size, activation=tf.identity, name=name)
    return outputs[name]

# Basic CNN for fbanks data with
# batch normalization.
def mkc_f(data_batch, mode):
    output_size=50
    outputs = {}
    name = 'CL1'
    with tf.variable_scope(name):
        output = tf.layers.conv2d(data_batch, 4, [16, 1], strides=[16, 1], activation=None,  name='conv')
        output = tf.layers.batch_normalization(output, name='batchNorm', training=True)#(mode==TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'CL2'
    with tf.variable_scope(name):
        output = tf.layers.conv2d(outputs['CL1'], 8, [8, 1], strides=[4, 1], activation=None, name='conv')
        output = tf.layers.batch_normalization(output, name='batchNorm', training=True)#(mode==TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'MP1'
    outputs[name] = tf.layers.max_pooling2d(outputs['CL2'], pool_size=[2, 1], strides=[2, 1], name=name)

    name = 'CL3'
    with tf.variable_scope(name):
        output = tf.layers.conv2d(outputs['MP1'], 12, [4, 1], strides=[1, 1], activation=None, name='conv')
        output = tf.layers.batch_normalization(output, name='batchNorm', training=True)#(mode==TRAIN))
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'MP2'
    outputs[name] = tf.layers.max_pooling2d(outputs['CL3'], pool_size=[2, 1], strides=[2, 1], name=name)

    name = 'FLTN'
    outputs[name] = tf.reshape(outputs['MP2'], [int(outputs['MP2'].shape[0]), -1], name=name)

    name = 'FCL1'
    outputs[name] = tf.layers.dense(outputs['FLTN'], 3000, activation=tf.nn.elu, name=name)

    name = 'FCL2'
    outputs[name] = tf.layers.dense(outputs['FCL1'], 1000, activation=tf.nn.elu, name=name)

    name = 'FCL3'
    outputs[name] = tf.layers.dense(outputs['FCL2'], output_size, activation=tf.identity, name=name)
    return outputs[name]


# ---------------------------------------------------------------------------------------------------------------------
# In Development
# ---------------------------------------------------------------------------------------------------------------------

def ds256fa(data_batch):
    OZ = 50
    outputs = {}
    names = []

    name = 'conv-1'
    names.append(name)
    output = tf.layers.conv2d(data_batch, 32, (8, 1), strides=(8, 1), activation=None,
                              name=name)
    output = tf.layers.batch_normalization(output, name=name + 'bn')
    outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'maxp-1'
    names.append(name)
    outputs[name] = tf.layers.max_pooling2d(outputs['conv-1'], pool_size=(4, 1), strides=(4, 1), name=name)

    name = 'conv-2'
    names.append(name)
    output = tf.layers.conv2d(outputs['maxp-1'], 32, (8, 1), strides=(8, 1), activation=None,
                              name=name)
    output = tf.layers.batch_normalization(output, name=name + 'bn')
    outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'maxp-2'
    names.append(name)
    outputs[name] = tf.layers.max_pooling2d(outputs['conv-2'], pool_size=(4, 1), strides=(4, 1), name=name)

    name = 'flatten'
    names.append(name)
    outputs[name] = tf.reshape(outputs['maxp-2'], [int(outputs['maxp-2'].shape[0]), -1])

    name = 'fcl-1'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['flatten'], 100, activation=tf.nn.elu, name=name)

    name = 'fcl-2'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['fcl-1'], OZ, activation=tf.identity, name=name)
    return outputs[name]
def ds256ra_t3(data_batch):
    FL = 256
    FD = 1
    SL = 256
    OZ = 3
    outputs = {}
    names = []
    name = 'strided-conv'
    names.append(name)
    with tf.name_scope(name):
        output = tf.layers.conv1d(data_batch, FD, FL, strides=SL, activation=None, name=name)
        output=tf.layers.batch_normalization(output, name=name + 'bn')
        outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'conv-1'
    names.append(name)
    with tf.name_scope(name):
        output = tf.layers.conv1d(outputs['strided-conv'], 32, 8, strides=1, activation=None, name=name)
        output=tf.layers.batch_normalization(output, name=name + 'bn')
        outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'maxp-1'
    names.append(name)
    with tf.name_scope(name):
        outputs[name] = tf.layers.max_pooling1d(outputs['conv-1'], pool_size=4, strides=4, name=name)

    name = 'conv-2'
    names.append(name)
    with tf.name_scope(name):
        output = tf.layers.conv1d(outputs['maxp-1'], 32, 8, strides=1, activation=None, name=name)
        output=tf.layers.batch_normalization(output, name=name + 'bn')
        outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'maxp-2'
    names.append(name)
    with tf.name_scope(name):
        outputs[name] = tf.layers.max_pooling1d(outputs['conv-2'], pool_size=4, strides=4, name=name)

    name = 'flatten'
    names.append(name)
    with tf.name_scope(name):
        outputs[name] = tf.reshape(outputs['maxp-2'], [int(outputs['maxp-2'].shape[0]), -1])

    name = 'fcl-1'
    names.append(name)
    with tf.name_scope(name):
        outputs[name] = tf.layers.dense(outputs['flatten'], 100, activation=tf.nn.elu, name=name)

    name = 'fcl-2'
    names.append(name)
    with tf.name_scope(name):
        outputs[name] = tf.layers.dense(outputs['fcl-1'], OZ, activation=tf.identity, name=name)
    return outputs[name]
def ds256rc(data_batch):
    FL = 16
    FD = 16
    SL = 16
    OZ = 50
    outputs = {}
    names = []
    name = 'strided-conv'
    names.append(name)
    output = tf.layers.conv1d(data_batch, FD, FL, strides=SL, activation=None, name=name)
    output = tf.layers.batch_normalization(output, name=name + 'bn')
    outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'conv-1'
    names.append(name)
    output = tf.layers.conv1d(outputs['strided-conv'], 32, 8, strides=1, activation=None, name=name)
    output = tf.layers.batch_normalization(output, name=name + 'bn')
    outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'maxp-1'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-1'], pool_size=4, strides=4, name=name)

    name = 'conv-2'
    names.append(name)
    output = tf.layers.conv1d(outputs['maxp-1'], 32, 8, strides=1, activation=None, name=name)
    output = tf.layers.batch_normalization(output, name=name + 'bn')
    outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'maxp-2'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-2'], pool_size=4, strides=4, name=name)

    name = 'flatten'
    names.append(name)
    outputs[name] = tf.reshape(outputs['maxp-2'], [int(outputs['maxp-2'].shape[0]), -1])

    name = 'fcl-1'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['flatten'], 100, activation=tf.nn.elu, name=name)

    name = 'fcl-2'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['fcl-1'], OZ, activation=tf.identity, name=name)
    return outputs[name]
def ds256rd(data_batch):
    FL = 256
    FD = 16
    SL = 16
    OZ = 50
    outputs = {}
    names = []
    name = 'strided-conv'
    names.append(name)
    output = tf.layers.conv1d(data_batch, FD, FL, strides=SL, activation=None, name=name)
    output = tf.layers.batch_normalization(output, name=name + 'bn')
    outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'conv-1'
    names.append(name)
    output = tf.layers.conv1d(outputs['strided-conv'], 32, 8, strides=1, activation=None, name=name)
    output = tf.layers.batch_normalization(output, name=name + 'bn')
    outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'maxp-1'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-1'], pool_size=4, strides=4, name=name)

    name = 'conv-2'
    names.append(name)
    output = tf.layers.conv1d(outputs['maxp-1'], 32, 8, strides=1, activation=None, name=name)
    output = tf.layers.batch_normalization(output, name=name + 'bn')
    outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'maxp-2'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-2'], pool_size=4, strides=4, name=name)

    name = 'flatten'
    names.append(name)
    outputs[name] = tf.reshape(outputs['maxp-2'], [int(outputs['maxp-2'].shape[0]), -1])

    name = 'fcl-1'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['flatten'], 100, activation=tf.nn.elu, name=name)

    name = 'fcl-2'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['fcl-1'], OZ, activation=tf.identity, name=name)
    return outputs[name]
def ds256re(data_batch):
    FL = 16
    FD = 16
    SL = 1
    OZ = 50
    outputs = {}
    names = []
    name = 'strided-conv'
    names.append(name)
    output = tf.layers.conv1d(data_batch, FD, FL, strides=SL, dilation_rate=2, activation=None, name=name)
    output = tf.layers.batch_normalization(output, name=name + 'bn')
    outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'conv-1'
    names.append(name)
    output = tf.layers.conv1d(outputs['strided-conv'], 32, 8, strides=1, dilation_rate=2, activation=None, name=name)
    output = tf.layers.batch_normalization(output, name=name + 'bn')
    outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'maxp-1'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-1'], pool_size=4, strides=4, name=name)

    name = 'conv-2'
    names.append(name)
    output = tf.layers.conv1d(outputs['maxp-1'], 32, 8, strides=1, dilation_rate=2, activation=None, name=name)
    output = tf.layers.batch_normalization(output, name=name + 'bn')
    outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'maxp-2'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-2'], pool_size=4, strides=4, name=name)

    name = 'flatten'
    names.append(name)
    outputs[name] = tf.reshape(outputs['maxp-2'], [int(outputs['maxp-2'].shape[0]), -1])

    name = 'fcl-1'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['flatten'], 100, activation=tf.nn.elu, name=name)

    name = 'fcl-2'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['fcl-1'], OZ, activation=tf.identity, name=name)
    return outputs[name]
def ds256rf(data_batch):
    FL = 16
    FD = 16
    SL = 1
    OZ = 50
    outputs = {}
    names = []
    name = 'strided-conv'
    names.append(name)
    output = tf.layers.conv1d(data_batch, FD, FL, strides=SL, dilation_rate=4, activation=None, name=name)
    output = tf.layers.batch_normalization(output, name=name + 'bn')
    outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'conv-1'
    names.append(name)
    output = tf.layers.conv1d(outputs['strided-conv'], 32, 8, strides=1, dilation_rate=4, activation=None, name=name)
    output = tf.layers.batch_normalization(output, name=name + 'bn')
    outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'maxp-1'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-1'], pool_size=4, strides=4, name=name)

    name = 'conv-2'
    names.append(name)
    output = tf.layers.conv1d(outputs['maxp-1'], 32, 8, strides=1, dilation_rate=4, activation=None, name=name)
    output = tf.layers.batch_normalization(output, name=name + 'bn')
    outputs[name] = tf.nn.elu(output, name=name + 'act')

    name = 'maxp-2'
    names.append(name)
    outputs[name] = tf.layers.max_pooling1d(outputs['conv-2'], pool_size=4, strides=4, name=name)

    name = 'flatten'
    names.append(name)
    outputs[name] = tf.reshape(outputs['maxp-2'], [int(outputs['maxp-2'].shape[0]), -1])

    name = 'fcl-1'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['flatten'], 100, activation=tf.nn.elu, name=name)

    name = 'fcl-2'
    names.append(name)
    outputs[name] = tf.layers.dense(outputs['fcl-1'], OZ, activation=tf.identity, name=name)
    return outputs[name]
