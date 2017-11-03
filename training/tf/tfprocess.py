#!/usr/bin/env python3

import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW',
                        strides=[1, 1, 1, 1], padding='SAME')

class TFProcess:
    def __init__(self):
        self.steps = 0
        self.session = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, 18*19*19])
        self.y_ = tf.placeholder(tf.float, [None, 362])
        self.y_conv = self.construct_net(self.x)

        self.cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                    logits=self.y_conv)
        self.cross_entropy_mean = tf.reduce_mean(self.cross_entropy)
        self.train_step = \
            tf.train.GradientDescentOptimizer(1e-2).minimize(self.cross_entropy)

        self.correct_prediction = \
            tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.correct_prediction = \
            tf.cast(self.correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)

        self.init = tf.global_variabbles_initializer()
        self.session.run(self.init)

    def process(self, batch):
        self.steps += 1
        if self.steps % 100 == 0:
            train_accuracy = self.accuracy.eval(
                feed_dict={self.x: batch[0], self.y_: batch[1]})
            print('step %d, training accuracy %g' % (self.steps, train_accuracy))
        self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1]})

        #if self.steps % 1000 == 0:
        #    print('test accuracy %g' % self.accuracy.eval(feed_dict={
        #        x: mnist.test.images, y_: mnist.test.labels}))

    def construct_net(self, planes):
        # NCHW format
        # batch, 18 channels, 19 x 19
        x_planes = tf.reshape(planes, [-1, 18, 19, 19])

        W_conv1 = weight_variable([3, 3, 18, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x_planes, W_conv1) + b_conv1)
        W_conv2 = weight_variable([3, 3, 64, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv2) + b_conv3)
        W_conv4 = weight_variable([3, 3, 64, 64])
        b_conv4 = bias_variable([64])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv2) + b_conv4)
        W_conv5 = weight_variable([3, 3, 64, 64])
        b_conv5 = bias_variable([64])
        h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv2) + b_conv5)
        W_conv6 = weight_variable([3, 3, 64,  2])
        b_conv6 = bias_variable([64])
        h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv2) + b_conv6)

        W_fc1 = weight_variable([2 * 361, 362])
        b_fc1 = bias_variable([362])

        h_conv6_flat = tf.reshape(h_conv6, [-1, 2*19*19])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv6_flat, W_fc1) + b_fc1)

        return h_fc1
