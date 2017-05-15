######################################################################################
# Yunlong Mao,                                                                       #
#                                                                                    #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md   #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow    #
######################################################################################

import tensorflow as tf
import numpy as np

def inference(input_images, train=True):
    # zero-mean input
    with tf.name_scope('preprocess') as scope:
        print('input images size:')
        print(input_images.shape)
        mean, variance = tf.nn.moments(input_images, axes=[0])
        images = tf.nn.batch_normalization(input_images, mean, variance, None, None, 1e-8)
        #mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        #images = input_images-mean

    # conv1_1 with Local_Norm
    with tf.variable_scope('conv1_1') :
        kernel = tf.get_variable('weights', shape=[3, 3, 3, 64], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        out_a = tf.nn.relu(tf.nn.bias_add(conv, biases))
        # local_norm
        #my_module = tf.load_op_library('/home/yl_mao0916/tensor/tensorflow/bazel-bin/tensorflow/core/user_ops/myl_net_layer.so')
        #out_b = my_module.local_norm(out_a)
        out_b = tf.nn.local_response_normalization(out_a)
        #noise = tf.random_normal([64,224,224,64], 0, 0.96)
        #out_c = tf.add(out_b,noise)
        #conv1_1 = tf.convert_to_tensor(out_b, name='conv1_1')
        conv1_1 = tf.convert_to_tensor(out_b, name='conv1_1')

    # conv1_2
    with tf.variable_scope('conv1_2') :
        kernel = tf.get_variable('weights', shape=[3, 3, 64, 64], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        out = tf.nn.bias_add(conv, biases)
        #relu1_2
        conv1_2 = tf.nn.relu(out, name='conv1_2')

    # pool1
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    # conv2_1
    with tf.variable_scope('conv2_1') :
        kernel = tf.get_variable('weights', shape=[3, 3, 64, 128], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        out = tf.nn.bias_add(conv, biases)
        #relu2_1
        conv2_1 = tf.nn.relu(out, name='conv2_1')

    # conv2_2
    with tf.variable_scope('conv2_2') :
        kernel = tf.get_variable('weights', shape=[3, 3, 128, 128], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        out = tf.nn.bias_add(conv, biases)
        #relu2_2
        conv2_2 = tf.nn.relu(out, name='conv2_2')

    # pool2
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    # conv3_1
    with tf.variable_scope('conv3_1') :
        kernel = tf.get_variable('weights', shape=[3, 3, 128, 256], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        out = tf.nn.bias_add(conv, biases)
        #relu3_1
        conv3_1 = tf.nn.relu(out, name='conv3_1')

    # conv3_2
    with tf.variable_scope('conv3_2') :
        kernel = tf.get_variable('weights', shape=[3, 3, 256, 256], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        out = tf.nn.bias_add(conv, biases)
        #relu3_2
        conv3_2 = tf.nn.relu(out, name='conv3_2')

    # conv3_3
    with tf.variable_scope('conv3_3') :
        kernel = tf.get_variable('weights', shape=[3, 3, 256, 256], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        out = tf.nn.bias_add(conv, biases)
        #relu3_3
        conv3_3 = tf.nn.relu(out, name='conv3_3')

    # pool3
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')

    # conv4_1
    with tf.variable_scope('conv4_1') :
        kernel = tf.get_variable('weights', shape=[3, 3, 256, 512], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        out = tf.nn.bias_add(conv, biases)
        #relu4_1
        conv4_1 = tf.nn.relu(out, name='conv4_1')

    # conv4_2
    with tf.variable_scope('conv4_2') :
        kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        out = tf.nn.bias_add(conv, biases)
        #relu4_2
        conv4_2 = tf.nn.relu(out, name='conv4_2')

    # conv4_3
    with tf.variable_scope('conv4_3') :
        kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        out = tf.nn.bias_add(conv, biases)
        #relu4_3
        conv4_3 = tf.nn.relu(out, name='conv4_3')

    # pool4
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool4')

    # conv5_1
    with tf.variable_scope('conv5_1') :
        kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        out = tf.nn.bias_add(conv, biases)
        #relu5_1
        conv5_1 = tf.nn.relu(out, name='conv5_1')

    # conv5_2
    with tf.variable_scope('conv5_2') :
        kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        out = tf.nn.bias_add(conv, biases)
        #relu5_2
        conv5_2 = tf.nn.relu(out, name='conv5_2')

    # conv5_3
    with tf.variable_scope('conv5_3') :
        kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        out = tf.nn.bias_add(conv, biases)
        #relu5_3
        conv5_3 = tf.nn.relu(out, name='conv5_3')

    # pool5
    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

    # fc1
    with tf.variable_scope('fc6') :
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc1w = tf.get_variable('weights', shape=[shape,4096], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        fc1b = tf.get_variable('biases', shape=[4096], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
        #relu6
        if train:
            fc1 = tf.nn.dropout(tf.nn.relu(fc1l), 0.5)
        else:
            fc1 = tf.nn.relu(fc1l)


    # fc2
    with tf.variable_scope('fc7') :
        fc2w = tf.get_variable('weights', shape=[4096,4096], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=train)
        fc2b = tf.get_variable('biases', shape=[4096], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
        fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
        #relu7
        if train:
            fc2 = tf.nn.dropout(tf.nn.relu(fc2l), 0.5)
        else:
            fc2 = tf.nn.relu(fc2l)

    
    # fc3
    with tf.variable_scope('fc8') :
        fc3w = tf.get_variable('weights', shape=[4096,158], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-2), trainable=True)
        fc3b = tf.get_variable('biases', shape=[158], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=True)
        fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)

    return fc3l


def load_weights(weight_file, sess):
    weights = np.load(weight_file).item()
    if 'fc8' in weights:
        del weights['fc8']
        print 'FC layer #8 will not be loaded.'
    keys = sorted(weights.keys())
    for key in keys:
        with tf.variable_scope(key, reuse=True):
            print key, np.shape(weights[key]['weights'])
            sess.run(tf.get_variable('weights').assign(weights[key]['weights']))
            sess.run(tf.get_variable('biases').assign(weights[key]['biases']))
    #imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
