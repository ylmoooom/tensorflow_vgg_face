from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import random
from six.moves import urllib
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '../dataset/', """Path to the training data directory.""")
tf.app.flags.DEFINE_integer('num_classes', 62, """Number of classes.""")
tf.app.flags.DEFINE_integer('image_size', 224, """Images size of input.""")


NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2742
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 281
# Constants describing the training process.
WEIGHT_DECAY_FACTOR = 0.0005
MOVING_AVERAGE_DECAY = 0.9     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 40      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.


def train_input(dir, namelist):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  train_data_dir = os.path.join(FLAGS.data_dir, dir)

  filenames = []
  labels = []
  for line in open(os.path.join(FLAGS.data_dir, namelist)):
    splited = line.split("\t")
    labels.append(int(splited[1]))
    f = os.path.join(train_data_dir, splited[0])
    filenames.append(f)
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  image_batch = []
  label_batch = []
  for i in range(FLAGS.batch_size):
    findex = random.randint(0,NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN-1)
    image = imread(filenames[findex], mode='RGB')
    image = imresize(image,(224, 224))
    image_batch.append(image)
    # For softmax_cross_entropy_with_logits.
    label = np.zeros(FLAGS.num_classes)
    label[labels[findex]] = 1
    label_batch.append(label)

    # For sparse_softmax_cross_entropy_with_logits.
    #label_batch.append(labels[findex])

  image_batch = np.array(image_batch)
  label_batch = np.array(label_batch)
  #print(image_batch.shape)
  #print(label_batch.shape)
  return image_batch, label_batch



# def batch_input(dir, batch_size, namelist):
#   if not FLAGS.data_dir:
#     raise ValueError('Please supply a data_dir')
#   train_data_dir = os.path.join(FLAGS.data_dir, dir)

#   filenames = []
#   labels = []
#   for line in open(os.path.join(FLAGS.data_dir, namelist)):
#     splited = line.split(' ')
#     labels.append(int(splited[1]))
#     filenames.append(os.path.join(train_data_dir, splited[0]))
  
#   for f in filenames:
#     if not tf.gfile.Exists(f):
#       raise ValueError('Failed to find file: ' + f)

#   labels = tf.one_hot(labels, NUM_CLASSES)
#   filename_label_queue = tf.train.slice_input_producer([filenames, labels])
#   value = tf.read_file(filename_label_queue[0])
#   ori_img = tf.image.decode_png(value)
#   image = tf.image.resize_images(ori_img, [224, 224])
#   image.set_shape((224, 224, 3))
#   label = filename_label_queue[1]

#   # min_after_dequeue defines how big a buffer we will randomly sample
#   #   from -- bigger means better shuffling but slower start up and more
#   #   memory used.
#   # capacity must be larger than min_after_dequeue and the amount larger
#   #   determines the maximum we will prefetch.  Recommendation:
#   #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
#   min_after_dequeue = 500
#   capacity = min_after_dequeue + 3 * batch_size

#   # Subtract off the mean and divide by the variance of the pixels.
#   #float_image = tf.image.per_image_standardization(distorted_image)

#   # Generate a batch of images and labels by building up a queue of examples.
#   image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

#   if FLAGS.use_fp16:
#     image_batch = tf.cast(label_batch, tf.float16)
#     label_batch = tf.cast(label_batch, tf.float16)

#   return image_batch, label_batch


def loss(logits, labels):
  labels = tf.cast(labels, tf.float64)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  #weight_norm = tf.reduce_sum(input_tensor=WEIGHT_DECAY_FACTOR*tf.stack([tf.nn.l2_loss(i) for i in tf.trainable_variables()]))
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    #correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
  with tf.name_scope('total_loss'):
    #total_loss = tf.add_n([weight_norm, cross_entropy_mean])
    tf.summary.scalar('total_loss', cross_entropy_mean)
  return cross_entropy_mean, accuracy


# def _add_loss_summaries(total_loss):
#   # Compute the moving average of all individual losses and the total loss.
#   loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
#   losses = tf.get_collection('losses')
#   loss_averages_op = loss_averages.apply(losses + [total_loss])

#   # Attach a scalar summary to all individual losses and the total loss; do the
#   # same for the averaged version of the losses.
#   for l in losses + [total_loss]:
#     # Name each loss as '(raw)' and name the moving average version of the loss
#     # as the original loss name.
#     tf.summary.scalar(l.op.name + ' (raw)', l)
#     tf.summary.scalar(l.op.name, loss_averages.average(l))

#   return loss_averages_op



def train(total_loss, global_step):
 # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  #loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  #with tf.control_dependencies([loss_averages_op]):
  opt = tf.train.MomentumOptimizer(lr,0.9)
  grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  #variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
  #variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train')
  return train_op

def test_input(dir, namelist):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir for test phase.')
  train_data_dir = os.path.join(FLAGS.data_dir, dir)

  image_batch = []
  label_batch = []
  filenames = []
  labels = []
  for line in open(os.path.join(FLAGS.data_dir, namelist)):
    splited = line.split("\t")
    labels.append(int(splited[1]))
    filename = os.path.join(train_data_dir, splited[0])
    filenames.append(filename)
    if not tf.gfile.Exists(filename):
      raise ValueError('Failed to find file: ' + filename)

  for i in range(FLAGS.batch_size,):
    findex = random.randint(0, NUM_EXAMPLES_PER_EPOCH_FOR_EVAL-1)
    image = imread(filenames[findex], mode='RGB')
    image = imresize(image,(224, 224))
    image_batch.append(image)
    label = np.zeros(FLAGS.num_classes)
    label[labels[findex]] = 1
    label_batch.append(label)

    # For sparse_softmax_cross_entropy_with_logits.
    #label_batch.append(labels[findex])

  image_batch = np.array(image_batch)
  label_batch = np.array(label_batch)

  return image_batch, label_batch