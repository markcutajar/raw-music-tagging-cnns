# -*- coding: utf-8 -*-
"""This code trains a predefined CNN model to automatically tag musical songs.
    It is designed to make use of Google's distributed cloud machine learning
    engine.

    Please note that some of the code is courtesy of the Google TensorFlow,
    Authors, and the census samples project found at:

    https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census
"""

import argparse
import json
import os
from . import models as models
import threading
import tensorflow as tf
from .dataproviders import DataProvider

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_CHECKPOINT = 60
TRAIN_SUMMARIES = 60
CHECKPOINT_PER_EVAL = 4  #2


# ---------------------------------------------------------------------------------------------------------------------
# Hook to run by the Monitored Session
# ---------------------------------------------------------------------------------------------------------------------


class EvaluationRunHook(tf.train.SessionRunHook):
    """EvaluationRunHook performs continuous evaluation of the model.

    Args:
        checkpoint_dir (string): Dir to store model checkpoints
        metric_dir (string): Dir to store metrics like accuracy and auroc
        graph (tf.Graph): Evaluation graph
        eval_frequency (int): Frequency of evaluation every n train steps
        eval_steps (int): Evaluation steps to be performed
    """

    def __init__(self,
                 checkpoint_dir,
                 metrics,
                 graph,
                 eval_frequency,
                 eval_steps=None,
                 **kwargs):

        self._eval_steps = eval_steps
        self._checkpoint_dir = checkpoint_dir
        self._kwargs = kwargs
        # self._eval_every = eval_frequency
        self._eval_every = CHECKPOINT_PER_EVAL
        self._latest_checkpoint = None
        self._checkpoints_since_eval = 0
        self._graph = graph

        with graph.as_default():
            stream_value_dict, stream_update_dict = tf.contrib.metrics.aggregate_metric_map(metrics['stream'])
            stream_metrics = [
                tf.summary.scalar(name, value_op)
                for name, value_op in stream_value_dict.items()
            ]

            perclass_value_dict, perclass_update_dict = tf.contrib.metrics.aggregate_metric_map(metrics['perclass'])
            perclass_metrics = [
                tf.summary.scalar(name, value_op)
                for name, value_op in perclass_value_dict.items()
            ]

            other_scalars = [
                tf.summary.scalar(name, value_op)
                for name, value_op in metrics['scalar'].items()
            ]

            variables = []
            for var in self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                variables.append(tf.summary.histogram(var.name.replace(':', '_'), var))

            self._summary_metrics = tf.summary.merge_all()

            # Saver class add ops to save and restore
            # variables to and from checkpoint
            self._saver = tf.train.Saver()

            # Creates a global step to contain a counter for
            # the global training step
            self._gs = tf.contrib.framework.get_or_create_global_step()

            self._final_ops_dict = [stream_value_dict, perclass_value_dict]
            self._eval_ops = [stream_update_dict.values(), perclass_update_dict.values()]
            # self._final_ops_dict = stream_value_dict
            # self._eval_ops = stream_update_dict.values()

        # MonitoredTrainingSession runs hooks in background threads
        # and it doesn't wait for the thread from the last session.run()
        # call to terminate to invoke the next hook, hence locks.
        self._eval_lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()
        self._file_writer = tf.summary.FileWriter(
            os.path.join(checkpoint_dir, 'eval'), graph=graph)

    def after_run(self, run_context, run_values):
        # Always check for new checkpoints in case a single evaluation
        # takes longer than checkpoint frequency and _eval_every is >1
        self._update_latest_checkpoint()

        if self._eval_lock.acquire(False):
            try:
                if self._checkpoints_since_eval >= self._eval_every:
                    tf.logging.info('running eval after run')
                    self._checkpoints_since_eval = 0
                    self._run_eval()
            finally:
                self._eval_lock.release()

    def _update_latest_checkpoint(self):
        """Update the latest checkpoint file created in the output dir."""
        if self._checkpoint_lock.acquire(False):
            try:
                latest = tf.train.latest_checkpoint(self._checkpoint_dir)
                if not latest == self._latest_checkpoint:
                    self._checkpoints_since_eval += 1
                    self._latest_checkpoint = latest
            finally:
                self._checkpoint_lock.release()

    def end(self, session):
        """Called at then end of session to make sure we always evaluate."""
        self._update_latest_checkpoint()
        # with self._eval_lock:
        #    self._run_eval()

    def _run_eval(self):
        """Run model evaluation and generate summaries."""
        coord = tf.train.Coordinator(clean_stop_exception_types=(
            tf.errors.CancelledError, tf.errors.OutOfRangeError))
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Session(graph=self._graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            # Restores previously saved variables from latest checkpoint
            self._saver.restore(session, self._latest_checkpoint)

            session.run([
                tf.tables_initializer(),
                tf.local_variables_initializer()
            ])

            tf.train.start_queue_runners(coord=coord, sess=session)
            train_step = session.run(self._gs)

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            tf.logging.info('Starting Evaluation For Step: {}'.format(train_step))
            with coord.stop_on_exception():
                eval_step = 0
                while self._eval_steps is None or eval_step < self._eval_steps:
                    results, final_values, _ = session.run([
                        self._summary_metrics,
                        self._final_ops_dict,
                        self._eval_ops
                    ], options=run_options, run_metadata=run_metadata)

                    if eval_step % 100 == 0:
                        tf.logging.info("On Evaluation Step: {}".format(eval_step))
                    eval_step += 1

            # Write the summaries, save results and log
            self._file_writer.add_summary(results, global_step=train_step)
            self._file_writer.add_run_metadata(run_metadata, 'run_metadata_{}'.format(train_step),
                                               global_step=train_step)
            self._file_writer.flush()
            tf.logging.info('Eval complete. Step: {}'.format(eval_step))


# ---------------------------------------------------------------------------------------------------------------------
# Run method for each worker/master node
# ---------------------------------------------------------------------------------------------------------------------


def run(target,
        cluster,
        is_chief,
        job_dir,
        train_files,
        eval_files,
        metadata_files,
        train_steps,
        eval_steps,
        train_batch_size,
        eval_batch_size,
        learning_rate,
        eval_frequency,
        model_function,
        eval_num_epochs,
        num_epochs,
        target_size,
        selective_tags,
        num_song_samples,
        windowing_type):
    """Run the training and evaluation graph.

    Args:
        target (string): TensorFlow server target
        is_chief (bool): Boolean flag to specify a chief server
        cluster: Cluster_spec of the custer being run for replica_device_setter
        train_files (string): File for training
        eval_files (string): File for evaluation
        metadata_files (string): File containing dataset metadata
        job_dir (string): Output dir for checkpoint and summary
        train_steps (int): Maximum number of training steps
        eval_steps (int): Number of steps to run evaluation for at each checkpoint
        train_batch_size (int): Batch size for training
        eval_batch_size (int): Batch size for evaluation
        learning_rate (float): Learning rate for Gradient Descent
        eval_frequency (int): Run evaluation frequency every n training steps.
            Do not evaluate too frequently otherwise you will
            pay for performance and do not evaluate too in-frequently
            otherwise you will not know how soon to stop training.
            Use default values to start with
        model_function (str): Function name to be loaded and used to build the graph
        eval_num_epochs (int): Number of epochs during evaluation
        num_epochs (int): Maximum number of training data epochs on which to train
        target_size (int): The number of tags being use as an output
        selective_tags (filename): Filename for selective tags in json
        num_song_samples (int): Samples from the songs to be used for training
        windowing_type (str): The type of windowing to be used.
            None: No windowing
            STME: Seperate training and merged evaluation
            SPM: Super-pooled model
    """

    # If the server is chief which is `master`
    # In between graph replication Chief is one node in
    # the cluster with extra responsibility and by default
    # is worker task zero.
    #
    # The duties of the chief are, being a worker, saving
    # checkpoints, running evaluation and restoring if a
    # crash happens.

    if is_chief:
        # Construct evaluation graph
        # tf.logging.info('Learning Rate: {}'.format(learning_rate))
        evaluation_graph = tf.Graph()
        with evaluation_graph.as_default():
            # Evaluation data provider
            eval_data = DataProvider(
                [eval_files],
                metadata_files,
                batch_size=eval_batch_size,
                num_epochs=eval_num_epochs,
                num_tags=target_size,
                num_samples=num_song_samples
            )

            # Features and label tensors
            if windowing_type is None:
                features, labels = eval_data.batch_in()
            else:
                features, labels = eval_data.windows_batch_in()

            # Model for evaluation
            metrics = models.controller(
                model_function,
                models.EVAL,
                features,
                labels,
                learning_rate=learning_rate,
                window=windowing_type
            )

        # Hook for monitored training session
        hooks = [EvaluationRunHook(
            job_dir,
            metrics,
            evaluation_graph,
            eval_frequency,
            eval_steps=eval_steps
        )]

    else:
        hooks = []

    # Create a new graph and specify that as default
    training_graph = tf.Graph()
    with training_graph.as_default():

        with tf.device(tf.train.replica_device_setter()):  # cluster=cluster)):
            # Training data provider
            train_data = DataProvider(
                [train_files],
                metadata_files,
                batch_size=train_batch_size,
                num_epochs=num_epochs,
                num_tags=target_size,
                num_samples=num_song_samples
            )

            # Features and label tensors
            if windowing_type is None:
                features, labels = train_data.batch_in()
            elif windowing_type is 'STME':
                features, labels = train_data.batch_in()
            else:
                features, labels = train_data.windows_batch_in()

            # Model for training
            [train_op, global_step_tensor, train_error] = models.controller(
                model_function,
                models.TRAIN,
                features,
                labels,
                learning_rate=learning_rate,
                window=windowing_type
            )

        # Summary for training error
        error_summary = tf.summary.merge([
            tf.summary.scalar('training_error', train_error)
        ])

        # GPU option to limit usage of GPU memory per process.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

        # File writer for metadata of master:replica
        if is_chief:
            meta_writer = tf.summary.FileWriter(os.path.join(job_dir, 'train'), graph=training_graph)

        # Creates a MonitoredSession for training
        # MonitoredSession is a Session-like object that handles
        # initialization, recovery and hooks
        tf.logging.info('Starting session')
        with tf.train.MonitoredTrainingSession(master=target,
                                               is_chief=is_chief,
                                               checkpoint_dir=job_dir,
                                               hooks=hooks,
                                               save_checkpoint_secs=TRAIN_CHECKPOINT,
                                               save_summaries_steps=TRAIN_SUMMARIES,
                                               config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            # Command for logging variable and ops device placement: log_device_placement = True

            # Tuple of exceptions that should cause a clean stop of the coordinator
            coord = tf.train.Coordinator(clean_stop_exception_types=(
                tf.errors.CancelledError, tf.errors.OutOfRangeError))
            # Important to start all queue runners so that data is available
            # for reading.
            # Initialize the input_fn thread to load the queue runner.
            tf.train.start_queue_runners(coord=coord, sess=session)

            # Global step to keep track of global number of steps particularly in
            # distributed setting
            step = global_step_tensor.eval(session=session)

            # Set up run options and metadata
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # Run the training graph which returns the step number as tracked by
            # the global step tensor.
            # When train epochs is reached, coord.should_stop() will be true.
            with coord.stop_on_exception():
                while (train_steps is None or step < train_steps) and not coord.should_stop():
                    step, _, error = session.run([global_step_tensor, train_op, error_summary],
                                                 options=run_options, run_metadata=run_metadata)
                    if is_chief:
                        if step == 10 or step % 2000 == 0:
                            meta_writer.add_run_metadata(run_metadata,
                                                         'run_metadata_{}'.format(step),
                                                         global_step=step)


# ---------------------------------------------------------------------------------------------------------------------
# Cluster Creator
# ---------------------------------------------------------------------------------------------------------------------


def dispatch(*args, **kwargs):
    """Parse TF_CONFIG to cluster_spec and call run() method
    """
    tf.logging.info('Setting up the server')
    tf_config = os.environ.get('TF_CONFIG')

    # If TF_CONFIG is not available run local
    if not tf_config:
        return run('', None, True, *args, **kwargs)

    tf_config_json = json.loads(tf_config)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    # If cluster information is empty run local
    if job_name is None or task_index is None:
        return run('', None, True, *args, **kwargs)

    cluster_spec = tf.train.ClusterSpec(cluster)
    server = tf.train.Server(cluster_spec,
                             job_name=job_name,
                             task_index=task_index)

    # Wait for incoming connections forever
    # Worker ships the graph to the ps server
    # The ps server manages the parameters of the model.
    if job_name == 'ps':
        server.join()
        return
    elif job_name in ['master', 'worker']:
        return run(server.target, cluster_spec, job_name == 'master', *args, **kwargs)


# ---------------------------------------------------------------------------------------------------------------------
# Input Parsing
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-files',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')

    parser.add_argument('--eval-files',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')

    parser.add_argument('--metadata-files',
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')

    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')

    parser.add_argument('--model-function',
                        required=True,
                        type=str)

    parser.add_argument('--train-steps',
                        type=int,
                        help="""\
                        Maximum number of training steps to perform
                        Training steps are in the units of training-batch-size.
                        So if train-steps is 500 and train-batch-size if 100 then
                        at most 500 * 100 training instances will be used to train.
                        """)

    parser.add_argument('--eval-steps',
                        help='Number of steps to run evaluation for at each checkpoint',
                        default=106,
                        type=int)

    parser.add_argument('--train-batch-size',
                        type=int,
                        default=20,
                        help='Batch size for training steps')

    parser.add_argument('--eval-batch-size',
                        type=int,
                        default=20,
                        help='Batch size for evaluation steps')

    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.001,
                        help='Learning rate for Optimizer')

    parser.add_argument('--eval-frequency',
                        default=1,
                        help='Perform one evaluation per n steps')

    parser.add_argument('--eval-num-epochs',
                        type=int,
                        default=1,
                        help='Number of epochs during evaluation')

    parser.add_argument('--num-epochs',
                        type=int,
                        help='Maximum number of epochs on which to train')

    parser.add_argument('--target-size',
                        type=int,
                        default=50,
                        help='Number of tags to be used as an output')

    parser.add_argument('--selective-tags',
                        type=str,
                        help='Path to selective tag file')

    parser.add_argument('--num-song-samples',
                        type=int,
                        default=-1,
                        help="""\
                        Samples of the songs to be used in the training,
                        evaluation and prediction process.
                        """)

    parser.add_argument('--windowing-type',
                        type=str,
                        default=None,
                        help="""\
                            The type of windowing to be used by the function.
                            None: No windowing.
                            stme: Seperate training and merged evaluation.
                            spm: Super-pooled output layer model
                            """)

    parse_args, unknown = parser.parse_known_args()

    # If unknown arguments found, warn them on the console
    tf.logging.warn('Unknown arguments: {}'.format(unknown))
    dispatch(**parse_args.__dict__)
