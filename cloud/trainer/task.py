# -*- coding: utf-8 -*-
"""This code trains a predefined CNN model to automatically tag musical songs.
    It is designed to make use of Google's distributed cloud machine learning
    engine.
    
    Please note that some of the code is courtosy of the Google Tensorflow,
    Authurs, and the census samples project found at:

    https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census
"""

import argparse
import json
import os
import trainer.models as models
import threading
import tensorflow as tf
from trainer.dataproviders import DataProvider

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.INFO)


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
        #self._eval_every = eval_frequency
        self._eval_every = 10
        self._latest_checkpoint = None
        self._checkpoints_since_eval = 0
        self._graph = graph

        with graph.as_default():
            value_dict, update_dict = tf.contrib.metrics.aggregate_metric_map(metrics[0])

            # Creates a Summary protocol buffer by merging summaries
            self._summary_streaming = tf.summary.merge([
                tf.summary.scalar(name, value_op)
                for name, value_op in value_dict.iteritems()
            ])

            self._summary_error = tf.summary.merge([
                tf.summary.scalar('evaluation_error', metrics[1])
            ])

            # Saver class add ops to save and restore
            # variables to and from checkpoint
            self._saver = tf.train.Saver()

            # Creates a global step to contain a counter for
            # the global training step
            self._gs = tf.contrib.framework.get_or_create_global_step()

            self._final_ops_dict = value_dict
            self._eval_ops = update_dict.values()

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

        with self._eval_lock:
            self._run_eval()

    def _run_eval(self):
        """Run model evaluation and generate summaries."""
        coord = tf.train.Coordinator(clean_stop_exception_types=(
            tf.errors.CancelledError, tf.errors.OutOfRangeError))

        with tf.Session(graph=self._graph) as session:
            # Restores previously saved variables from latest checkpoint
            self._saver.restore(session, self._latest_checkpoint)

            session.run([
                tf.tables_initializer(),
                tf.local_variables_initializer()
            ])

            tf.train.start_queue_runners(coord=coord, sess=session)
            train_step = session.run(self._gs)

            tf.logging.info('Starting Evaluation For Step: {}'.format(train_step))
            with coord.stop_on_exception():
                eval_step = 0
                while self._eval_steps is None or eval_step < self._eval_steps:
                    error_streaming, error_sce, final_values, _ = session.run([
                        self._summary_streaming,
                        self._summary_error,
                        self._final_ops_dict,
                        self._eval_ops
                    ])
                    if eval_step % 100 == 0 or eval_step % 0 == 0:
                        tf.logging.info("On Evaluation Step: {}".format(eval_step))
                    eval_step += 1

            # Write the summaries
            self._file_writer.add_summary(error_streaming, global_step=train_step)
            self._file_writer.add_summary(error_sce, global_step=train_step)
            self._file_writer.flush()
            tf.logging.info(final_values)


def run(target,
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
        num_song_samples,
        data_shape):

    """Run the training and evaluation graph.

    Args:
        target (string): TensorFlow server target
        is_chief (bool): Boolean flag to specify a chief server
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
        num_song_samples (int): Samples from the songs to be used for training
        data_shape (string): Shape of data - depending on the model used
    """

    model = getattr(models, model_function)
    
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
        evaluation_graph = tf.Graph()
        with evaluation_graph.as_default():

            eval_data = DataProvider(
                [eval_files],
                metadata_files,
                batch_size=eval_batch_size,
                num_epochs=eval_num_epochs,
                num_tags=target_size,
                num_samples=num_song_samples,
                data_shape=data_shape
            )
            
            # Features and label tensors
            features, labels = eval_data.raw_input_fn()
            
            # Metric dictionary of evaluation
            metric_dict, error = model(
                models.EVAL,
                features,
                labels,
                learning_rate=learning_rate
            )

        hooks = [EvaluationRunHook(
            job_dir,
            [metric_dict, error],
            evaluation_graph,
            eval_frequency,
            eval_steps=eval_steps
        )]
        
    else:
        hooks = []
  
    # Create a new graph and specify that as default
    training_graph = tf.Graph()
    with training_graph.as_default():
    
        with tf.device(tf.train.replica_device_setter()):
            train_data = DataProvider(
                [train_files],
                metadata_files,
                batch_size=train_batch_size,
                num_epochs=num_epochs,
                num_tags=target_size,
                num_samples=num_song_samples,
                data_shape=data_shape
            )
            
            # Features and label tensors
            features, labels = train_data.raw_input_fn()
            
            # Metric 
            [train_op, global_step_tensor, train_error] = model(
                models.TRAIN,
                features,
                labels,
                learning_rate=learning_rate            
            )

        error_summary = tf.summary.merge([
            tf.summary.scalar('training_error', train_error)
        ])

        if is_chief:
            train_file_writer = tf.summary.FileWriter(
                os.path.join(job_dir, 'train'), graph=training_graph)

        # Creates a MonitoredSession for training
        # MonitoredSession is a Session-like object that handles
        # initialization, recovery and hooks
        tf.logging.info('Starting session')
        with tf.train.MonitoredTrainingSession(master=target, 
                                               is_chief=is_chief,
                                               checkpoint_dir=job_dir,
                                               hooks=hooks,
                                               save_checkpoint_secs=30,
                                               save_summaries_steps=30) as session:

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

            # Run the training graph which returns the step number as tracked by
            # the global step tensor.
            # When train epochs is reached, coord.should_stop() will be true.
            with coord.stop_on_exception():
                while (train_steps is None or step < train_steps) and not coord.should_stop():
                    step, _, error = session.run([global_step_tensor, train_op, error_summary])

                    if is_chief:
                        train_file_writer.add_summary(error, global_step=step)
                        train_file_writer.flush()



def dispatch(*args, **kwargs):
    """Parse TF_CONFIG to cluster_spec and call run() method
    """
    tf.logging.info('Setting up the server')
    tf_config = os.environ.get('TF_CONFIG')

    # If TF_CONFIG is not available run local
    if not tf_config:
        return run('', True, *args, **kwargs)

    tf_config_json = json.loads(tf_config)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    # If cluster information is empty run local
    if job_name is None or task_index is None:
        return run('', True, *args, **kwargs)

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
        return run(server.target, job_name == 'master', *args, **kwargs)


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
                        default=129,
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
                        default=0.003,
                        help='Learning rate for SGD')
  
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
                        
    parser.add_argument('--num-song-samples',
                        type=int,
                        default=465984,
                        help="""\
                        Samples of the songs to be used in the training,
                        evaluation and prediction process.
                        """)

    parser.add_argument('--data-shape',
                        type=str,
                        default='image',
                        help="""\
                        Shape of the data - flat - or - image -. Depending
                        on if mlp or conv model respectively.
                        """)
                        
    parse_args, unknown = parser.parse_known_args()

    # If unknown arguments found, warn them on the console
    tf.logging.warn('Unknown arguments: {}'.format(unknown))
    dispatch(**parse_args.__dict__)
