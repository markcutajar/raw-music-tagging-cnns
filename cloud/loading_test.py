import time
from trainer.dataproviders import DataProvider
import tensorflow as tf
import numpy as np

start = time.time()
dataBatcher = DataProvider(filenames=['../magnatagatune/train_rawdata.tfrecords'], 
                           metadata_file='../magnatagatune/raw_metadata.json',
                           batch_size = 20,
                           num_tags=50,
                           num_samples=-1)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
features, tags = dataBatcher.batch_in()

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for  i in range(200):
        songs, labels = sess.run([features, tags])

    coord.request_stop()
    coord.join(threads)
    
end = time.time()
print('total: {}'.format(end-start))
print('persec: {}'.format(200.0/(end-start)))