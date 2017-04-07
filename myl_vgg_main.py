from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf
import myl_vgg_train as mytrain
from myl_vgg_net import vgg16


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './train_master',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 5000,
                            """Number of steps to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


if __name__ == '__main__':
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  """Train VGG model."""
  global_step = tf.contrib.framework.get_or_create_global_step()

  # Get images and labels.
  images, labels = mytrain.train_input('train_data', FLAGS.batch_size, 'train.txt')

  # Build a Graph that computes the logits.
  mynet = vgg16(images, train=False)
  print('cnn built.')
  logits = mynet.forward_output()

  # Calculate loss.
  loss = mytrain.loss(logits, labels)

  # Build a Graph that trains the model with one batch of examples and
  # updates the model parameters.
  train_op = mytrain.train(loss, global_step)

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)
    mynet.load_weights('converted_data', sess)
    print('weights loaded!')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print('running...')
    #moniter(train_op, loss)
    try:
      while not coord.should_stop():
        # Run training steps or whatever
        print('running...train_op')
        sess.run(train_op)

    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()