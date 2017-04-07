from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
from six.moves import urllib
import tensorflow as tf
import numpy as np


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './lfw-224', """Path to the training data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")


IMAGE_SIZE = 224
NUM_CLASSES = 62
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 3000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 5      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.


def train_input(dir, batch_size, namelist):
  print('here is input...')
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  train_data_dir = os.path.join(FLAGS.data_dir, dir)

  filenames = []
  labels = []
  for line in open(os.path.join(FLAGS.data_dir, namelist)):
    splited = line.split(' ')
    labels.append(int(splited[1]))
    filenames.append(os.path.join(train_data_dir, splited[0]))
  
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  labels = tf.one_hot(labels, NUM_CLASSES)
  filename_label_queue = tf.train.slice_input_producer([filenames, labels])
  value = tf.read_file(filename_label_queue[0])
  ori_img = tf.image.decode_png(value)
  image = tf.image.resize_images(ori_img, [224, 224])
  image.set_shape((224, 224, 3))
  label = filename_label_queue[1]

  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 1000
  capacity = min_after_dequeue + 3 * batch_size

  # Subtract off the mean and divide by the variance of the pixels.
  #float_image = tf.image.per_image_standardization(distorted_image)

  # Generate a batch of images and labels by building up a queue of examples.
  image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

  if FLAGS.use_fp16:
    image_batch = tf.cast(label_batch, tf.float16)
    label_batch = tf.cast(label_batch, tf.float16)

  return image_batch, label_batch



def loss(logits, labels):
  print('Calculating loss!!!')
  # Calculate the average cross entropy loss across the batch.
  print(type(logits))
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  return cross_entropy_mean


def train(total_loss, global_step):
  print('loss: ')
  print(total_loss.shape)
  print('step: ')
  print(global_step)
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)

  # Compute gradients.
  opt = tf.train.AdamOptimizer(lr)
  grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return apply_gradient_op