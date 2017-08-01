"""Graph functions to test the raw music tagging."""

import tensorflow as tf

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'
STME, SPM = 'STME', 'SPM'

TRUE_POSITIVE_FACTOR = 10
TAG_BALANCING_FACTOR = 0
# ---------------------------------------------------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------------------------------------------------


def controller(function_name,
               mode,
               data_super_batch,
               targets_super_batch,
               gpus,
               learning_rate=0.001,
               window=None):

    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    global_step = tf.contrib.framework.get_or_create_global_step()
    model = globals()[function_name]

    # Gradients array for different towers having
    # the different GPUs.
    tower_grads = []
    tower_losses = []
    tower_predictions = []
    with tf.variable_scope(tf.get_variable_scope()):
        for GPU_IDX, GPU in enumerate(gpus):

            data_batch = data_super_batch[GPU_IDX]
            targets_batch = targets_super_batch[GPU_IDX]

            with tf.device(GPU):
                with tf.name_scope(GPU.replace('/', '').replace(':', '_')) as scope:
                    logits = get_logits(model, data_batch, window, mode)

                    with tf.name_scope('tower_loss'):
                        class_weights = balancing_weights(50, 'log', TAG_BALANCING_FACTOR)
                        loss = weighted_sigmoid_cross_entropy(logits=logits,
                                                              labels=targets_batch,
                                                              false_negatives_weight=TRUE_POSITIVE_FACTOR,
                                                              balancing_weights_vector=class_weights)
                        tower_losses.append(loss)

                    tf.get_variable_scope().reuse_variables()
                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    if mode in TRAIN:
                        with tf.name_scope('tower_gradients'):
                            gradients = optimizer.compute_gradients(loss)
                        tower_grads.append(gradients)

                    elif mode in EVAL:
                        # Get tag predictions
                        with tf.name_scope('tower_predictions'):
                            prediction_values = tf.nn.sigmoid(logits, name='probs')
                            tower_predictions.append(tf.round(prediction_values))

    # Get merged loss
    with tf.name_scope('loss'):
        losses = tf.stack(values=tower_losses)
        merged_loss = tf.reduce_mean(losses)
        tf.losses.add_loss(merged_loss)

    # Compute train_op if training
    if mode in TRAIN:
        with tf.name_scope('train'):
            all_gradients = average_gradients(tower_grads)
            train_op = optimizer.apply_gradients(all_gradients, global_step=global_step)

        # Add histograms for gradients.
        for grad, var in all_gradients:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Add merged error_loss
        summaries.append(tf.summary.scalar('training_error', merged_loss))
        return train_op, global_step

    elif mode in EVAL:
        # Get tag predictions
        with tf.name_scope('predictions'):
            predictions = tf.concat(tower_predictions, axis=0)
            targets = tf.concat(targets_super_batch, axis=0)

        with tf.name_scope('metrics'):
            return {
                'stream': general_metrics(predictions, targets),
                'scalar': {'evaluation_error': merged_loss},
                'perclass': perclass_metrics(predictions, targets)
            }
    else:
        raise ValueError('Mode {} not found'.format(mode))


def average_gradients(tower_grads):

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_logits(model, data_batch, window, mode):
    if window is None:
        # normal model
        logits = model(data_batch, mode)
    elif window in STME and mode in TRAIN:
        # normal model
        logits = model(data_batch, mode)
    elif window in STME and mode in EVAL:
        # model with tf.map_fn
        logits_array = tf.map_fn(lambda w: model(w, mode),
                                 elems=data_batch,
                                 back_prop=True,
                                 parallel_iterations=12,
                                 swap_memory=True,
                                 name='MapModels')
        logits = tf.reduce_mean(logits_array, axis=0, name='averageLogits')
    elif window in SPM:
        # super pooled model
        # model with tf.map_fn
        logits_array = tf.map_fn(lambda w: model(w, mode),
                                 elems=data_batch,
                                 back_prop=True,
                                 parallel_iterations=12,
                                 swap_memory=True,
                                 name='MapModels')

        logits = superpool_c(tf.stack(tf.unstack(logits_array), axis=1, name='mergingLogits'))

    else:
        raise ValueError('Window type {} not recognized'.format(window))
    return logits


def general_metrics(predictions, targets_batch):
    return {
        'false_negatives': tf.contrib.metrics.streaming_false_negatives(
            predictions, targets_batch, name='false_negatives'),
        'false_positives': tf.contrib.metrics.streaming_false_positives(
            predictions, targets_batch, name='false_positives'),
        'true_positives': tf.contrib.metrics.streaming_true_positives(
            predictions, targets_batch, name='true_positives'),
        'true_negatives': tf.contrib.metrics.streaming_true_negatives(
            predictions, targets_batch, name='true_negatives'),
        'aucroc': tf.contrib.metrics.streaming_auc(
            predictions, targets_batch, name='aucroc')
    }


def perclass_metrics(predictions, targets_batch):

    perclass_dict = {}
    predictions_per_tag_list = tf.unstack(predictions, axis=1)
    targets_per_tag_list = tf.unstack(targets_batch, axis=1)

    for idx, pred_tag in enumerate(predictions_per_tag_list):
        """perclass_dict[str(idx)+'_false_negatives'] = tf.contrib.metrics.streaming_false_negatives(
            pred_tag, targets_per_tag_list[idx], name='false_negatives')

        perclass_dict[str(idx)+'_false_positives'] = tf.contrib.metrics.streaming_false_positives(
            pred_tag, targets_per_tag_list[idx], name='false_positives')

        perclass_dict[str(idx) + '_true_positives'] = tf.contrib.metrics.streaming_true_positives(
            pred_tag, targets_per_tag_list[idx], name='true_positives')

        perclass_dict[str(idx) + '_true_negatives'] = tf.contrib.metrics.streaming_true_negatives(
            pred_tag, targets_per_tag_list[idx], name='true_negatives')"""

        perclass_dict[str(idx)+'_aucroc'] = tf.contrib.metrics.streaming_auc(
            pred_tag, targets_per_tag_list[idx], name='aucroc')

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


def balancing_weights(num_classes, function_type, factor):
    if function_type == 'log':
        class_weights = tf.constant(list(range(1, num_classes+1)), dtype=tf.float32)
        class_weights = tf.add(tf.multiply(tf.constant(factor, dtype=tf.float32),
                                           tf.log(class_weights)), tf.constant(1.0))
        return class_weights
    else:
        raise NotImplementedError('Function {} not implemented! Only Log!'.format(function_type))


def conv_max_layers_1d(in_layer,
                       depth,
                       filter_length,
                       filter_stride,
                       conv_name,
                       pool_length,
                       pool_stride,
                       pool_name,
                       dil_rate=1):

    with tf.variable_scope(conv_name):
        out_cl = tf.contrib.layers.conv2d(in_layer, depth, [1, filter_length],
                                          stride=[1, filter_stride],
                                          scope='conv',
                                          rate=dil_rate,
                                          activation_fn=tf.nn.elu,
                                          normalizer_fn=tf.contrib.layers.batch_norm)

    out_mp = tf.contrib.layers.max_pool2d(out_cl, [1, pool_length], stride=[1, pool_stride], scope=pool_name)
    return out_mp


def conv_avg_layers_1d(in_layer,
                       depth,
                       filter_length,
                       filter_stride,
                       conv_name,
                       pool_length,
                       pool_stride,
                       pool_name,
                       dil_rate=1):

    with tf.variable_scope(conv_name):
        out_cl = tf.contrib.layers.conv2d(in_layer, depth, [1, filter_length],
                                          stride=[1, filter_stride],
                                          scope='conv',
                                          rate=dil_rate,
                                          activation_fn=tf.nn.elu,
                                          normalizer_fn=tf.contrib.layers.batch_norm)

    out_mp = tf.contrib.layers.avg_pool2d(out_cl, [1, pool_length], stride=[1, pool_stride], scope=pool_name)
    return out_mp


def conv_layers_1d(in_layer,
                   depth,
                   filter_length,
                   filter_stride,
                   conv_name,
                   dil_rate=1):

    with tf.variable_scope(conv_name):
        return tf.contrib.layers.conv2d(in_layer, depth, [1, filter_length],
                                        stride=[1, filter_stride],
                                        scope='conv',
                                        rate=dil_rate,
                                        activation_fn=tf.nn.elu,
                                        normalizer_fn=tf.contrib.layers.batch_norm)


# ---------------------------------------------------------------------------------------------------------------------
# Super pooling models
# ---------------------------------------------------------------------------------------------------------------------


def superpool_a(data):
    superpool_outputs = {}
    name = 'FCSL1'
    superpool_outputs[name] = tf.layers.dense(data, 600, activation=tf.nn.elu, name=name)

    name = 'FCSL2'
    superpool_outputs[name] = tf.layers.dense(superpool_outputs['FCSL1'], 600, activation=tf.nn.elu, name=name)

    name = 'FCSL3'
    superpool_outputs[name] = tf.layers.dense(superpool_outputs['FCSL2'], 50, activation=tf.identity, name=name)
    return superpool_outputs[name]


def superpool_b(data):
    superpool_outputs = {}
    name = 'FCSL1'
    output = tf.layers.dense(data, 600, activation=tf.nn.elu, name=name)
    superpool_outputs[name] = tf.layers.dropout(output, training=True)

    name = 'FCSL2'
    output = tf.layers.dense(superpool_outputs['FCSL1'], 600, activation=tf.nn.elu, name=name)
    superpool_outputs[name] = tf.layers.dropout(output, training=True)

    name = 'FCSL3'
    superpool_outputs[name] = tf.layers.dense(superpool_outputs['FCSL2'], 50, activation=tf.identity, name=name)
    return superpool_outputs[name]


def superpool_c(data):
    with tf.variable_scope('CL1_SP'):
        out_cl1_sp = tf.layers.conv1d(data, 32, 3, strides=1, activation=None,
                                      name='conv', data_format='channels_first')
        out_cl1_sp = tf.layers.batch_normalization(out_cl1_sp, name='batchNorm', training=True, axis=1)
        out_cl1_sp = tf.nn.elu(out_cl1_sp, name='nonLin')

    out_fltn_sp = tf.reshape(out_cl1_sp, [int(out_cl1_sp.shape[0]), -1], name='FLTN_SP')

    out_fcl2_sp = tf.layers.dense(out_fltn_sp, 1500, activation=tf.nn.elu, name='FCSL2')
    out_fcl2_sp = tf.layers.dropout(out_fcl2_sp, training=True)

    output_final = tf.layers.dense(out_fcl2_sp, 50, activation=tf.identity, name='FCSL3')
    return output_final


# ---------------------------------------------------------------------------------------------------------------------
# Baseline Models
# ---------------------------------------------------------------------------------------------------------------------


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
# DS256 Models
# ---------------------------------------------------------------------------------------------------------------------


# Model proposed by Dieleman et al. using Raw data
# First Conv: FL256, FS256, FD1
# Output: 50 Neurons
# Structure: 3 Conv, 2 MLP
def ds256ra(data_batch, mode):
    filt_length = 256
    filt_depth = 128
    stride_length = 256
    output_size = 50

    name = 'CL1'
    with tf.variable_scope(name):
        out_cl1 = tf.layers.conv1d(data_batch,
                                   filt_depth,
                                   filt_length,
                                   strides=stride_length,
                                   activation=None,
                                   name='conv')
        out_cl1 = tf.layers.batch_normalization(out_cl1, name='batchNorm', training=True)
        out_cl1 = tf.nn.elu(out_cl1, name='nonLin')

    name = 'CL2'
    with tf.variable_scope(name):
        out_cl2 = tf.layers.conv1d(out_cl1, 32, 8, strides=1, activation=None, name='conv')
        out_cl2 = tf.layers.batch_normalization(out_cl2, name='batchNorm', training=True)
        out_cl2 = tf.nn.elu(out_cl2, name='nonLin')

    name = 'MP1'
    out_mp1 = tf.layers.max_pooling1d(out_cl2, pool_size=4, strides=4, name=name)

    name = 'CL3'
    with tf.variable_scope(name):
        out_cl3 = tf.layers.conv1d(out_mp1, 32, 8, strides=1, activation=None, name='conv')
        out_cl3 = tf.layers.batch_normalization(out_cl3, name='batchNorm', training=True)
        out_cl3 = tf.nn.elu(out_cl3, name='nonLin')

    name = 'MP2'
    out_mp2 = tf.layers.max_pooling1d(out_cl3, pool_size=4, strides=4, name=name)

    name = 'FLTN'
    out_fltn = tf.reshape(out_mp2, [int(out_mp2.shape[0]), -1], name=name)

    name = 'FCL1'
    out_fcl1 = tf.layers.dense(out_fltn, 1000, activation=tf.nn.elu, name=name)

    name = 'FCL2'
    out_fcl2 = tf.layers.dense(out_fcl1, output_size, activation=tf.identity, name=name)
    return out_fcl2


# Model proposed by Dieleman et al. using FBanks data
# Structure: 2 Conv, 2 MLP
def ds256fa(data_batch, mode):
    output_size = 50
    outputs = {}

    name = 'CL1'
    with tf.variable_scope(name):
        output = tf.layers.conv2d(data_batch, 32, (8, 1), strides=(1, 1), activation=None,  name='conv')
        output = tf.layers.batch_normalization(output, name='batchNorm', training=True)
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'MP1'
    outputs[name] = tf.layers.max_pooling2d(outputs['CL1'], pool_size=(4, 1), strides=(4, 1), name=name)

    name = 'CL2'
    with tf.variable_scope(name):
        output = tf.layers.conv2d(outputs['MP1'], 32, (8, 1), strides=(1, 1), activation=None, name='conv')
        output = tf.layers.batch_normalization(output, name='batchNorm', training=True)
        outputs[name] = tf.nn.elu(output, name='nonLin')

    name = 'MP2'
    outputs[name] = tf.layers.max_pooling2d(outputs['CL2'], pool_size=(4, 1), strides=(4, 1), name=name)

    name = 'FLTN'
    outputs[name] = tf.reshape(outputs['MP2'], [int(outputs['MP2'].shape[0]), -1], name=name)

    name = 'FCL1'
    outputs[name] = tf.layers.dense(outputs['FLTN'], 1000, activation=tf.nn.elu, name=name)

    name = 'FCL2'
    outputs[name] = tf.layers.dense(outputs['FCL1'], output_size, activation=tf.identity, name=name)
    return outputs[name]


# Model proposed by Dieleman et al. using Raw data
# First Conv: FL256, FS256, FD1
# Output: 200 neurons
# Structure: 3 Conv, 2 MLP
def ds256rb(data_batch, mode):
    filt_length = 256
    filt_depth = 128
    stride_length = 256
    output_size = 50
    outputs = {}

    name = 'CL1'
    with tf.variable_scope(name):
        out_CL1 = tf.layers.conv1d(data_batch,
                                   filt_depth,
                                   filt_length,
                                   strides=stride_length,
                                   activation=None,
                                   name='conv')
        out_CL1 = tf.layers.batch_normalization(out_CL1, name='batchNorm', training=True)
        out_CL1 = tf.nn.elu(out_CL1, name='nonLin')

    name = 'CL2'
    with tf.variable_scope(name):
        out_CL2 = tf.layers.conv1d(out_CL1, 32, 8, strides=1, activation=None, name='conv')
        out_CL2 = tf.layers.batch_normalization(out_CL2, name='batchNorm', training=True)
        out_CL2 = tf.nn.elu(out_CL2, name='nonLin')

    name = 'MP1'
    out_MP1 = tf.layers.max_pooling1d(out_CL2, pool_size=4, strides=4, name=name)

    name = 'CL3'
    with tf.variable_scope(name):
        out_CL3 = tf.layers.conv1d(out_MP1, 32, 8, strides=1, activation=None, name='conv')
        out_CL3 = tf.layers.batch_normalization(out_CL3, name='batchNorm', training=True)
        out_CL3 = tf.nn.elu(out_CL3, name='nonLin')

    name = 'MP2'
    out_MP2 = tf.layers.max_pooling1d(out_CL3, pool_size=4, strides=4, name=name)

    name = 'FLTN'
    out_FLTN = tf.reshape(out_MP2, [int(out_MP2.shape[0]), -1], name=name)

    name = 'FCL1'
    out_FCL1 = tf.layers.dense(out_FLTN, 1000, activation=tf.nn.elu, name=name)
    out_FCL1 = tf.layers.dropout(out_FCL1, training=True)

    name = 'FCL2'
    out_FCL2 = tf.layers.dense(out_FCL1, 200, activation=tf.identity, name=name)
    out_FCL2 = tf.layers.dropout(out_FCL2, training=True)
    return out_FCL2


# Model similar to one proposed by Dieleman et al.
# First Conv: FL256, FS256, FD1
# Output: 3x1x32
# Structure: 4 Conv
def ds256rc(data_batch, mode):
    filt_length = 256
    filt_depth = 128
    stride_length = 256
    output_size = 50
    outputs = {}

    name = 'CL1'
    with tf.variable_scope(name):
        out_CL1 = tf.layers.conv1d(data_batch,
                                   filt_depth,
                                   filt_length,
                                   strides=stride_length,
                                   activation=None,
                                   name='conv')
        out_CL1 = tf.layers.batch_normalization(out_CL1, name='batchNorm', training=True)
        out_CL1 = tf.nn.elu(out_CL1, name='nonLin')

    name = 'CL2'
    with tf.variable_scope(name):
        out_CL2 = tf.layers.conv1d(out_CL1, 32, 8, strides=1, activation=None, name='conv')
        out_CL2 = tf.layers.batch_normalization(out_CL2, name='batchNorm', training=True)
        out_CL2 = tf.nn.elu(out_CL2, name='nonLin')

    name = 'MP1'
    out_MP1 = tf.layers.max_pooling1d(out_CL2, pool_size=4, strides=4, name=name)

    name = 'CL3'
    with tf.variable_scope(name):
        out_CL3 = tf.layers.conv1d(out_MP1, 32, 8, strides=1, activation=None, name='conv')
        out_CL3 = tf.layers.batch_normalization(out_CL3, name='batchNorm', training=True)
        out_CL3 = tf.nn.elu(out_CL3, name='nonLin')
    return out_CL3


# ---------------------------------------------------------------------------------------------------------------------
# DM Models
# ---------------------------------------------------------------------------------------------------------------------


# Deeper Model similar to proposed by Lee et al.
# Raw data
# Output: 50 Neurons
# Structure: 7 Conv, 1 MLP
def dm8_ra(data_batch, mode):

    data = tf.expand_dims(data_batch, axis=1)
    with tf.variable_scope('CL1'):
        out_l1 = tf.contrib.layers.conv2d(data, 64, [1, 8],
                                          stride=[1, 8],
                                          scope='conv',
                                          activation_fn=tf.nn.elu,
                                          normalizer_fn=tf.contrib.layers.batch_norm)  # 4854

    out_l2 = conv_max_layers_1d(out_l1, 64, 4, 1, 'CL2', 4, 4, 'MP1')  # 1212
    out_l3 = conv_max_layers_1d(out_l2, 128, 4, 1, 'CL3', 4, 4, 'MP2')  # 302
    out_l4 = conv_max_layers_1d(out_l3, 128, 4, 1, 'CL4', 4, 4, 'MP3')  # 74
    out_l5 = conv_max_layers_1d(out_l4, 256, 4, 1, 'CL5', 4, 4, 'MP4')  # 17
    out_l6 = conv_max_layers_1d(out_l5, 256, 4, 1, 'CL6', 2, 2, 'MP5')  # 6
    out_l7 = conv_max_layers_1d(out_l6, 512, 4, 1, 'CL7', 2, 2, 'MP6')  # 1

    out_drop = tf.contrib.layers.dropout(out_l7)
    out_fltn = tf.reshape(out_drop, [int(out_drop.shape[0]), -1], name='FLTN')
    out_fcl = tf.layers.dense(out_fltn, 50, activation=tf.nn.sigmoid, name='FCL')
    return out_fcl


# Deeper Model similar to proposed by Lee et al.
# Raw data
# Output: 50 Neurons
# Structure: 7 Conv, 1 MLP
def dm16_ra(data_batch, mode):

    data = tf.expand_dims(data_batch, axis=1)
    with tf.variable_scope('CL1'):
        out_l1 = tf.contrib.layers.conv2d(data, 64, [1, 16],
                                          stride=[1, 16],
                                          scope='conv',
                                          activation_fn=tf.nn.elu,
                                          normalizer_fn=tf.contrib.layers.batch_norm)  # 2427

    out_l2 = conv_max_layers_1d(out_l1, 64, 4, 1, 'CL2', 4, 4, 'MP1')  # 605
    out_l3 = conv_max_layers_1d(out_l2, 128, 4, 1, 'CL3', 4, 4, 'MP2')  # 150
    out_l4 = conv_max_layers_1d(out_l3, 128, 4, 1, 'CL4', 4, 4, 'MP3')  # 39
    out_l5 = conv_max_layers_1d(out_l4, 256, 4, 1, 'CL5', 2, 2, 'MP4')  # 17
    out_l6 = conv_max_layers_1d(out_l5, 256, 4, 1, 'CL6', 2, 2, 'MP5')  # 6
    out_l7 = conv_max_layers_1d(out_l6, 512, 4, 1, 'CL7', 2, 2, 'MP6')  # 1

    out_drop = tf.contrib.layers.dropout(out_l7)
    out_fltn = tf.reshape(out_drop, [int(out_drop.shape[0]), -1], name='FLTN')
    out_fcl = tf.layers.dense(out_fltn, 50, activation=tf.nn.sigmoid, name='FCL')
    return out_fcl


# Deeper Model similar to proposed by Lee et al.
# Raw data
# Output: 50 Neurons
# Structure: 7 Conv, 1 MLP
def dm16_rb(data_batch, mode):

    data = tf.expand_dims(data_batch, axis=1)
    with tf.variable_scope('CL1'):
        out_l1 = tf.contrib.layers.conv2d(data, 64, [1, 16],
                                          stride=[1, 16],
                                          scope='conv',
                                          activation_fn=tf.nn.elu,
                                          normalizer_fn=tf.contrib.layers.batch_norm)  # 2427

    out_l2 = conv_max_layers_1d(out_l1, 64, 4, 1, 'CL2', 4, 4, 'MP1')  # 605
    out_l3 = conv_max_layers_1d(out_l2, 128, 4, 1, 'CL3', 4, 4, 'MP2')  # 150
    out_l4 = conv_max_layers_1d(out_l3, 128, 4, 1, 'CL4', 4, 4, 'MP3')  # 39
    out_l5 = conv_max_layers_1d(out_l4, 256, 4, 1, 'CL5', 2, 2, 'MP4')  # 17
    out_l6 = conv_max_layers_1d(out_l5, 256, 4, 1, 'CL6', 2, 2, 'MP5')  # 6
    out_l7 = conv_max_layers_1d(out_l6, 512, 4, 1, 'CL7', 2, 2, 'MP6')  # 1

    out_drop = tf.contrib.layers.dropout(out_l7)
    out_fltn = tf.reshape(out_drop, [int(out_drop.shape[0]), -1], name='FLTN')
    out_fcl = tf.layers.dense(out_fltn, 50, activation=tf.identity, name='FCL')
    return out_fcl


# Deeper Model similar to proposed by Lee et al.
# Raw data
# Output: 50 Neurons
# Structure: 7 Conv, 1 MLP
def dm16_rc(data_batch, mode):

    data = tf.expand_dims(data_batch, axis=1)
    with tf.variable_scope('CL1'):
        out_l1 = tf.contrib.layers.conv2d(data, 64, [1, 16],
                                          stride=[1, 16],
                                          scope='conv',
                                          activation_fn=tf.nn.elu,
                                          normalizer_fn=tf.contrib.layers.batch_norm)  # 29124

    out_l2 = conv_max_layers_1d(out_l1, 64, 4, 1, 'CL2', 8, 8, 'MP1')  # 3640
    out_l3 = conv_max_layers_1d(out_l2, 128, 4, 1, 'CL3', 8, 8, 'MP2')  # 454
    out_l4 = conv_max_layers_1d(out_l3, 128, 4, 1, 'CL4', 4, 4, 'MP3')  # 113
    out_l5 = conv_max_layers_1d(out_l4, 256, 4, 1, 'CL5', 4, 4, 'MP4')  # 27
    out_l6 = conv_max_layers_1d(out_l5, 256, 4, 1, 'CL6', 4, 4, 'MP5')  # 5

    with tf.variable_scope('CL7'):
        out_l7 = tf.contrib.layers.conv2d(out_l6, 512, [1, 4],
                                          stride=[1, 4],
                                          scope='conv',
                                          activation_fn=tf.nn.elu,
                                          normalizer_fn=tf.contrib.layers.batch_norm)

    out_drop = tf.contrib.layers.dropout(out_l7)
    out_fltn = tf.reshape(out_drop, [int(out_drop.shape[0]), -1], name='FLTN')
    out_fcl = tf.layers.dense(out_fltn, 50, activation=tf.nn.sigmoid, name='FCL')
    return out_fcl


# Deeper Model similar to proposed by Lee et al.
# Raw data
# Output: 50 Neurons
# Structure: 7 Conv, 1 MLP
def dm64_ra(data_batch, mode):

    data = tf.expand_dims(data_batch, axis=1)
    with tf.variable_scope('CL1'):
        out_l1 = tf.contrib.layers.conv2d(data, 64, [1, 64],
                                          stride=[1, 64],
                                          scope='conv',
                                          activation_fn=tf.nn.elu,
                                          normalizer_fn=tf.contrib.layers.batch_norm)  # 606

    out_l2 = conv_max_layers_1d(out_l1, 64, 4, 1, 'CL2', 4, 4, 'MP1')  # 150
    out_l3 = conv_max_layers_1d(out_l2, 128, 4, 1, 'CL3', 2, 2, 'MP2')  # 73
    out_l4 = conv_max_layers_1d(out_l3, 128, 4, 1, 'CL4', 2, 2, 'MP3')  # 34
    out_l5 = conv_max_layers_1d(out_l4, 256, 4, 1, 'CL5', 2, 2, 'MP4')  # 15
    out_l6 = conv_max_layers_1d(out_l5, 256, 4, 1, 'CL6', 2, 2, 'MP5')  # 5

    with tf.variable_scope('CL7'):
        out_l7 = tf.contrib.layers.conv2d(out_l6, 512, [1, 4],
                                          stride=[1, 4],
                                          scope='conv',
                                          activation_fn=tf.nn.elu,
                                          normalizer_fn=tf.contrib.layers.batch_norm)  # 1

    out_drop = tf.contrib.layers.dropout(out_l7)
    out_fltn = tf.reshape(out_drop, [int(out_drop.shape[0]), -1], name='FLTN')
    out_fcl = tf.layers.dense(out_fltn, 50, activation=tf.nn.sigmoid, name='FCL')
    return out_fcl


# Deeper Model similar to proposed by Lee et al.
# Raw data
# Output: 50 Neurons
# Structure: 7 Conv, 1 MLP
def dm128_ra(data_batch, mode):

    data = tf.expand_dims(data_batch, axis=1)
    with tf.variable_scope('CL1'):
        out_l1 = tf.contrib.layers.conv2d(data, 64, [1, 128],
                                          stride=[1, 128],
                                          scope='conv',
                                          activation_fn=tf.nn.elu,
                                          normalizer_fn=tf.contrib.layers.batch_norm)  # 303

    out_l2 = conv_max_layers_1d(out_l1, 64, 4, 1, 'CL2', 2, 2, 'MP1')  # 149
    out_l3 = conv_max_layers_1d(out_l2, 128, 4, 1, 'CL3', 2, 2, 'MP2')  # 72
    out_l4 = conv_max_layers_1d(out_l3, 128, 4, 1, 'CL4', 2, 2, 'MP3')  # 34
    out_l5 = conv_max_layers_1d(out_l4, 256, 4, 1, 'CL5', 2, 2, 'MP4')  # 15
    out_l6 = conv_max_layers_1d(out_l5, 256, 4, 1, 'CL6', 2, 2, 'MP5')  # 5

    with tf.variable_scope('CL7'):
        out_l7 = tf.contrib.layers.conv2d(out_l6, 512, [1, 4],
                                          stride=[1, 4],
                                          scope='conv',
                                          activation_fn=tf.nn.elu,
                                          normalizer_fn=tf.contrib.layers.batch_norm)  # 1

    out_drop = tf.contrib.layers.dropout(out_l7)
    out_fltn = tf.reshape(out_drop, [int(out_drop.shape[0]), -1], name='FLTN')
    out_fcl = tf.layers.dense(out_fltn, 50, activation=tf.nn.sigmoid, name='FCL')
    return out_fcl


# ---------------------------------------------------------------------------------------------------------------------
# DL Models
# ---------------------------------------------------------------------------------------------------------------------


# Deeper Model similar to proposed by Lee et al.
# Raw data
# Output: 50 Neurons
# Structure: 7 Conv, 1 MLP
def dl16_16_ra(data_batch, mode):

    data = tf.expand_dims(data_batch, axis=1)
    with tf.variable_scope('CL1'):
        out_l1 = tf.contrib.layers.conv2d(data, 64, [1, 16],
                                          stride=[1, 16],
                                          scope='conv',
                                          activation_fn=tf.nn.elu,
                                          normalizer_fn=tf.contrib.layers.batch_norm)  # 2427

    out_l2 = conv_max_layers_1d(out_l1, 64, 4, 1, 'CL2', 4, 4, 'MP1', dil_rate=16)  # 605
    out_l3 = conv_max_layers_1d(out_l2, 128, 4, 1, 'CL3', 4, 4, 'MP2', dil_rate=8)  # 150
    out_l4 = conv_max_layers_1d(out_l3, 128, 4, 1, 'CL4', 4, 4, 'MP3', dil_rate=4)  # 39
    out_l5 = conv_max_layers_1d(out_l4, 256, 4, 1, 'CL5', 2, 2, 'MP4', dil_rate=2)  # 17
    out_l6 = conv_max_layers_1d(out_l5, 256, 4, 1, 'CL6', 2, 2, 'MP5')  # 6
    out_l7 = conv_max_layers_1d(out_l6, 512, 4, 1, 'CL7', 2, 2, 'MP6')  # 1

    out_drop = tf.contrib.layers.dropout(out_l7)
    out_fltn = tf.reshape(out_drop, [int(out_drop.shape[0]), -1], name='FLTN')
    out_fcl = tf.layers.dense(out_fltn, 50, activation=tf.nn.sigmoid, name='FCL')
    return out_fcl


# Deeper Model similar to proposed by Lee et al.
# Raw data
# Output: 50 Neurons
# Structure: 7 Conv, 1 MLP
def dl16_8_ra(data_batch, mode):

    data = tf.expand_dims(data_batch, axis=1)
    with tf.variable_scope('CL1'):
        out_l1 = tf.contrib.layers.conv2d(data, 64, [1, 16],
                                          stride=[1, 16],
                                          scope='conv',
                                          activation_fn=tf.nn.elu,
                                          normalizer_fn=tf.contrib.layers.batch_norm)  # 2427

    out_l2 = conv_max_layers_1d(out_l1, 64, 4, 1, 'CL2', 4, 4, 'MP1', dil_rate=8)  # 605
    out_l3 = conv_max_layers_1d(out_l2, 128, 4, 1, 'CL3', 4, 4, 'MP2', dil_rate=4)  # 150
    out_l4 = conv_max_layers_1d(out_l3, 128, 4, 1, 'CL4', 4, 4, 'MP3', dil_rate=2)  # 39
    out_l5 = conv_max_layers_1d(out_l4, 256, 4, 1, 'CL5', 2, 2, 'MP4')  # 17
    out_l6 = conv_max_layers_1d(out_l5, 256, 4, 1, 'CL6', 2, 2, 'MP5')  # 6
    out_l7 = conv_max_layers_1d(out_l6, 512, 4, 1, 'CL7', 2, 2, 'MP6')  # 1

    out_drop = tf.contrib.layers.dropout(out_l7)
    out_fltn = tf.reshape(out_drop, [int(out_drop.shape[0]), -1], name='FLTN')
    out_fcl = tf.layers.dense(out_fltn, 50, activation=tf.nn.sigmoid, name='FCL')
    return out_fcl


# Deeper Model similar to proposed by Lee et al.
# Raw data
# Output: 50 Neurons
# Structure: 7 Conv, 1 MLP
def dl16_4_ra(data_batch, mode):

    data = tf.expand_dims(data_batch, axis=1)
    with tf.variable_scope('CL1'):
        out_l1 = tf.contrib.layers.conv2d(data, 64, [1, 16],
                                          stride=[1, 16],
                                          scope='conv',
                                          activation_fn=tf.nn.elu,
                                          normalizer_fn=tf.contrib.layers.batch_norm)  # 2427

    out_l2 = conv_max_layers_1d(out_l1, 64, 4, 1, 'CL2', 4, 4, 'MP1', dil_rate=4)  # 605
    out_l3 = conv_max_layers_1d(out_l2, 128, 4, 1, 'CL3', 4, 4, 'MP2', dil_rate=2)  # 150
    out_l4 = conv_max_layers_1d(out_l3, 128, 4, 1, 'CL4', 4, 4, 'MP3')  # 39
    out_l5 = conv_max_layers_1d(out_l4, 256, 4, 1, 'CL5', 2, 2, 'MP4')  # 17
    out_l6 = conv_max_layers_1d(out_l5, 256, 4, 1, 'CL6', 2, 2, 'MP5')  # 6
    out_l7 = conv_max_layers_1d(out_l6, 512, 4, 1, 'CL7', 2, 2, 'MP6')  # 1

    out_drop = tf.contrib.layers.dropout(out_l7)
    out_fltn = tf.reshape(out_drop, [int(out_drop.shape[0]), -1], name='FLTN')
    out_fcl = tf.layers.dense(out_fltn, 50, activation=tf.nn.sigmoid, name='FCL')
    return out_fcl


# ---------------------------------------------------------------------------------------------------------------------
# MUL Models
# ---------------------------------------------------------------------------------------------------------------------


# Deeper Model similar to proposed by Lee et al.
# Raw data
# Output: 50 Neurons
# Structure: 7 Conv, 1 MLP
def mul16a(data_batch, mode):

    data = tf.expand_dims(data_batch, axis=1)
    out_l1 = conv_layers_1d(data, 32, 16, 16, 'CL1')  # 2427

    out_l2a = conv_max_layers_1d(out_l1, 32, 4, 1, 'CL2a', 4, 4, 'MP1a')  # 606
    out_l2b = conv_avg_layers_1d(out_l1, 32, 16, 1, 'CL2b', 4, 4, 'MP1b')  # 606

    out_l2 = tf.concat([out_l2a, out_l2b], axis=-1)

    out_l3a = conv_max_layers_1d(out_l2, 64, 4, 1, 'CL3a', 4, 4, 'MP2a')  # 151
    out_l3b = conv_avg_layers_1d(out_l2, 64, 16, 1, 'CL3b', 4, 4, 'MP2b')  # 151

    out_l3 = tf.concat([out_l3a, out_l3b], axis=-1)

    out_l4 = conv_max_layers_1d(out_l3, 128, 4, 1, 'CL4', 4, 4, 'MP3')  # 39
    out_l5 = conv_max_layers_1d(out_l4, 256, 4, 1, 'CL5', 4, 4, 'MP4')  # 9
    out_l6 = conv_max_layers_1d(out_l5, 256, 4, 1, 'CL6', 4, 4, 'MP5')  # 2
    out_l7 = conv_max_layers_1d(out_l6, 512, 4, 1, 'CL7', 2, 2, 'MP6')  # 1

    out_drop = tf.contrib.layers.dropout(out_l7)
    out_fltn = tf.reshape(out_drop, [int(out_drop.shape[0]), -1], name='FLTN')
    out_fcl = tf.layers.dense(out_fltn, 50, activation=tf.nn.sigmoid, name='FCL')
    return out_fcl


# Deeper Model similar to proposed by Lee et al.
# Raw data
# Output: 50 Neurons
# Structure: 7 Conv, 1 MLP
def mul16b(data_batch, mode):

    data = tf.expand_dims(data_batch, axis=1)
    out_l1 = conv_layers_1d(data, 32, 16, 16, 'CL1')  # 2427

    out_l2a = conv_max_layers_1d(out_l1, 32, 4, 1, 'CL2a', 4, 4, 'MP1a')  # 606

    out_l2ba = conv_layers_1d(out_l1, 32, 8, 1, 'CL2ba', dil_rate=8)
    out_l2bb = conv_layers_1d(out_l2ba, 32, 8, 1, 'CL2bb', dil_rate=4)
    out_l2b = conv_avg_layers_1d(out_l2bb, 32, 8, 1, 'CL2b', 4, 4, 'MP1b', dil_rate=2)

    out_l2 = tf.concat([out_l2a, out_l2b], axis=-1)

    out_l3a = conv_max_layers_1d(out_l2, 64, 4, 1, 'CL3a', 4, 4, 'MP2a')  # 151

    out_l3ba = conv_layers_1d(out_l2, 32, 8, 1, 'CL3ba', dil_rate=8)
    out_l3bb = conv_layers_1d(out_l3ba, 32, 8, 1, 'CL3bb', dil_rate=4)
    out_l3b = conv_avg_layers_1d(out_l3bb, 32, 8, 1, 'CL3b', 4, 4, 'MP2b', dil_rate=2)

    out_l3 = tf.concat([out_l3a, out_l3b], axis=-1)

    out_l4 = conv_max_layers_1d(out_l3, 128, 4, 1, 'CL4', 4, 4, 'MP3')  # 39
    out_l5 = conv_max_layers_1d(out_l4, 256, 4, 1, 'CL5', 4, 4, 'MP4')  # 9
    out_l6 = conv_max_layers_1d(out_l5, 256, 4, 1, 'CL6', 4, 4, 'MP5')  # 2
    out_l7 = conv_max_layers_1d(out_l6, 512, 4, 1, 'CL7', 2, 2, 'MP6')  # 1

    out_drop = tf.contrib.layers.dropout(out_l7)
    out_fltn = tf.reshape(out_drop, [int(out_drop.shape[0]), -1], name='FLTN')
    out_fcl = tf.layers.dense(out_fltn, 50, activation=tf.nn.sigmoid, name='FCL')
    return out_fcl


# Deeper Model similar to proposed by Lee et al.
# Raw data
# Output: 50 Neurons
# Structure: 7 Conv, 1 MLP
def mul16c(data_batch, mode):

    data = tf.expand_dims(data_batch, axis=1)
    out_l1 = conv_layers_1d(data, 32, 16, 16, 'CL1')  # 2427

    out_l2a = conv_max_layers_1d(out_l1, 32, 4, 1, 'CL2a', 4, 4, 'MP1a')  # 606
    out_l2ba = conv_layers_1d(out_l1, 32, 8, 1, 'CL2ba', dil_rate=8)
    out_l2bb = conv_layers_1d(out_l2ba, 32, 8, 1, 'CL2bb', dil_rate=4)
    out_l2b = conv_avg_layers_1d(out_l2bb, 32, 8, 1, 'CL2b', 4, 4, 'MP1b', dil_rate=2)
    out_l2c = conv_avg_layers_1d(out_l1, 32, 16, 1, 'CL2c', 4, 4, 'MP1c')
    out_l2 = tf.concat([out_l2a, out_l2b, out_l2c], axis=-1)

    out_l3a = conv_max_layers_1d(out_l2, 64, 4, 1, 'CL3a', 4, 4, 'MP2a')  # 151
    out_l3ba = conv_layers_1d(out_l2, 32, 8, 1, 'CL3ba', dil_rate=8)
    out_l3bb = conv_layers_1d(out_l3ba, 32, 8, 1, 'CL3bb', dil_rate=4)
    out_l3b = conv_avg_layers_1d(out_l3bb, 32, 8, 1, 'CL3b', 4, 4, 'MP2b', dil_rate=2)
    out_l3c = conv_avg_layers_1d(out_l2, 64, 16, 1, 'CL3c', 4, 4, 'MP2c')
    out_l3 = tf.concat([out_l3a, out_l3b, out_l3c], axis=-1)

    out_l4 = conv_max_layers_1d(out_l3, 128, 4, 1, 'CL4', 4, 4, 'MP3')  # 39
    out_l5 = conv_max_layers_1d(out_l4, 256, 4, 1, 'CL5', 4, 4, 'MP4')  # 9
    out_l6 = conv_max_layers_1d(out_l5, 256, 4, 1, 'CL6', 4, 4, 'MP5')  # 2
    out_l7 = conv_max_layers_1d(out_l6, 512, 4, 1, 'CL7', 2, 2, 'MP6')  # 1

    out_drop = tf.contrib.layers.dropout(out_l7)
    out_fltn = tf.reshape(out_drop, [int(out_drop.shape[0]), -1], name='FLTN')
    out_fcl = tf.layers.dense(out_fltn, 50, activation=tf.nn.sigmoid, name='FCL')
    return out_fcl


# Deeper Model similar to proposed by Lee et al.
# Raw data
# Output: 50 Neurons
# Structure: 7 Conv, 1 MLP
def mul16d(data_batch, mode):

    data = tf.expand_dims(data_batch, axis=1)
    out_l1a = conv_layers_1d(data, 32, 16, 16, 'CL1a')
    out_l1b = conv_layers_1d(data, 64, 64, 16, 'CL1b')
    out_l1 = tf.concat([out_l1a, out_l1b], axis=-1)

    out_l2a = conv_max_layers_1d(out_l1, 32, 4, 1, 'CL2a', 4, 4, 'MP1a')  # 606
    out_l2b = conv_avg_layers_1d(out_l1, 32, 16, 1, 'CL2b', 4, 4, 'MP1b')  # 606

    out_l2 = tf.concat([out_l2a, out_l2b], axis=-1)

    out_l3a = conv_max_layers_1d(out_l2, 64, 4, 1, 'CL3a', 4, 4, 'MP2a')  # 151
    out_l3b = conv_avg_layers_1d(out_l2, 64, 16, 1, 'CL3b', 4, 4, 'MP2b')  # 151

    out_l3 = tf.concat([out_l3a, out_l3b], axis=-1)

    out_l4 = conv_max_layers_1d(out_l3, 128, 4, 1, 'CL4', 4, 4, 'MP3')  # 39
    out_l5 = conv_max_layers_1d(out_l4, 256, 4, 1, 'CL5', 4, 4, 'MP4')  # 9
    out_l6 = conv_max_layers_1d(out_l5, 256, 4, 1, 'CL6', 4, 4, 'MP5')  # 2
    out_l7 = conv_max_layers_1d(out_l6, 512, 4, 1, 'CL7', 2, 2, 'MP6')  # 1

    out_drop = tf.contrib.layers.dropout(out_l7)
    out_fltn = tf.reshape(out_drop, [int(out_drop.shape[0]), -1], name='FLTN')
    out_fcl = tf.layers.dense(out_fltn, 50, activation=tf.nn.sigmoid, name='FCL')
    return out_fcl
