######################################################################################
# Yunlong Mao,                                                                       #
#                                                                                    #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md   #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow    #
######################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize

class vgg16:
    def __init__(self, imgs, train=True):
        self.imgs = imgs
        self.convlayers(train)
        self.fc_layers(train)
        #self.probs = tf.nn.softmax(self.fc3l)


    def convlayers(self, train):
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.variable_scope('conv1_1') :
            kernel = tf.get_variable('weights', shape=[3, 3, 3, 64], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
            out = tf.nn.bias_add(conv, biases)
            #relu1_1
            self.conv1_1 = tf.nn.relu(out, name='conv1_1')

        # conv1_2
        with tf.variable_scope('conv1_2') :
            kernel = tf.get_variable('weights', shape=[3, 3, 64, 64], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
            out = tf.nn.bias_add(conv, biases)
            #relu1_2
            self.conv1_2 = tf.nn.relu(out, name='conv1_2')

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1') :
            kernel = tf.get_variable('weights', shape=[3, 3, 64, 128], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
            out = tf.nn.bias_add(conv, biases)
            #relu2_1
            self.conv2_1 = tf.nn.relu(out, name='conv2_1')

        # conv2_2
        with tf.variable_scope('conv2_2') :
            kernel = tf.get_variable('weights', shape=[3, 3, 128, 128], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
            out = tf.nn.bias_add(conv, biases)
            #relu2_2
            self.conv2_2 = tf.nn.relu(out, name='conv2_2')

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1') :
            kernel = tf.get_variable('weights', shape=[3, 3, 128, 256], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
            out = tf.nn.bias_add(conv, biases)
            #relu3_1
            self.conv3_1 = tf.nn.relu(out, name='conv3_1')

        # conv3_2
        with tf.variable_scope('conv3_2') :
            kernel = tf.get_variable('weights', shape=[3, 3, 256, 256], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
            out = tf.nn.bias_add(conv, biases)
            #relu3_2
            self.conv3_2 = tf.nn.relu(out, name='conv3_2')

        # conv3_3
        with tf.variable_scope('conv3_3') :
            kernel = tf.get_variable('weights', shape=[3, 3, 256, 256], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
            out = tf.nn.bias_add(conv, biases)
            #relu3_3
            self.conv3_3 = tf.nn.relu(out, name='conv3_3')

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1') :
            kernel = tf.get_variable('weights', shape=[3, 3, 256, 512], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
            out = tf.nn.bias_add(conv, biases)
            #relu4_1
            self.conv4_1 = tf.nn.relu(out, name='conv4_1')

        # conv4_2
        with tf.variable_scope('conv4_2') :
            kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
            out = tf.nn.bias_add(conv, biases)
            #relu4_2
            self.conv4_2 = tf.nn.relu(out, name='conv4_2')

        # conv4_3
        with tf.variable_scope('conv4_3') :
            kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
            out = tf.nn.bias_add(conv, biases)
            #relu4_3
            self.conv4_3 = tf.nn.relu(out, name='conv4_3')

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1') :
            kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
            out = tf.nn.bias_add(conv, biases)
            #relu5_1
            self.conv5_1 = tf.nn.relu(out, name='conv5_1')

        # conv5_2
        with tf.variable_scope('conv5_2') :
            kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
            out = tf.nn.bias_add(conv, biases)
            #relu5_2
            self.conv5_2 = tf.nn.relu(out, name='conv5_2')

        # conv5_3
        with tf.variable_scope('conv5_3') :
            kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=train)
            out = tf.nn.bias_add(conv, biases)
            #relu5_3
            self.conv5_3 = tf.nn.relu(out, name='conv5_3')

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

    def fc_layers(self, train):
        # fc1
        with tf.variable_scope('fc6') :
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.get_variable('weights', shape=[shape,4096], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            fc1b = tf.get_variable('biases', shape=[4096], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=train)
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            #relu6
            self.fc1 = tf.nn.relu(fc1l)

        #!!!------DROPOUT ratio:0.5
        #self.fc1 = tf.nn.dropout(self.fc1, 0.5)

        # fc2
        with tf.variable_scope('fc7') :
            fc2w = tf.get_variable('weights', shape=[4096,4096], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            fc2b = tf.get_variable('biases', shape=[4096], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=train)
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            #relu7
            self.fc2 = tf.nn.relu(fc2l)

        #!!!------DROPOUT ratio:0.5
        #self.fc2 = tf.nn.dropout(self.fc2, 0.5)
        
        # fc3
        with tf.variable_scope('fc8') :
            fc3w = tf.get_variable('weights', shape=[4096,62], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=1e-1), trainable=train)
            fc3b = tf.get_variable('biases', shape=[62], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=True)
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)

    def forward_output(self):
        return self.fc3l

    def load_weights(self, weight_file, sess):
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

if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'converted_data', sess)

    img1 = imread('ak.png', mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print class_names[p], prob[p]
