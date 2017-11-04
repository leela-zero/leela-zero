#!/usr/bin/env python3

import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def relu_bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, trainable=False)

def offset_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable=True)

def scale_variable(shape):
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial, trainable=False)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW',
                        strides=[1, 1, 1, 1], padding='SAME')

class TFProcess:
    def __init__(self):
        self.steps = 0
        self.session = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, 18, 19 * 19])
        self.y_ = tf.placeholder(tf.float32, [None, 362])
        self.y_conv = self.construct_net(self.x)

        self.cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                    logits=self.y_conv)
        self.cross_entropy_mean = tf.reduce_mean(self.cross_entropy)
        self.train_step = \
            tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(self.cross_entropy)

        self.correct_prediction = \
            tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.correct_prediction = \
            tf.cast(self.correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)

        self.init = tf.global_variables_initializer()
        self.session.run(self.init)

    def process(self, batch):
        self.steps += 1
        if self.steps % 100 == 0:
            #train_accuracy = self.accuracy.eval(session=self.session,
            #    feed_dict={self.x: batch[0], self.y_: batch[1]})
            # print('step %d, training accuracy %g' % (self.steps, train_accuracy))
            train_loss = \
                self.cross_entropy_mean.eval(session=self.session,
                                             feed_dict={self.x: batch[0], self.y_: batch[1]})
            print('step %d, loss %g' % (self.steps, train_loss))
        self.train_step.run(session=self.session,
                            feed_dict={self.x: batch[0], self.y_: batch[1]})

        if self.steps % 1000 == 0:
            train_accuracy = \
                self.accuracy.eval(session=self.session,
                                   feed_dict={self.x: batch[0], self.y_: batch[1]})
            print('step %d, training accuracy %g' % (self.steps, train_accuracy))

    def conv_block(self, inputs, input_channels, output_channels):
        W_conv = weight_variable([3, 3, input_channels, output_channels])
        b_conv = relu_bias_variable([output_channels])
        s_conv = scale_variable([output_channels])
        o_conv = offset_variable([output_channels])
        h_bn, bm, vn = tf.nn.fused_batch_norm(tf.nn.bias_add(conv2d(inputs, W_conv),
                                                                    b_conv, data_format='NCHW'),
                                              s_conv, o_conv, None, None, data_format='NCHW')
        h_conv = tf.nn.relu(h_bn)
        return h_conv

    def construct_net(self, planes):
        # NCHW format
        # batch, 18 channels, 19 x 19
        x_planes = tf.reshape(planes, [-1, 18, 19, 19])

        conv1 = self.conv_block(x_planes,  18,  64)
        conv2 = self.conv_block(conv1,     64,  64)
        conv3 = self.conv_block(conv2,     64,  64)
        conv4 = self.conv_block(conv3,     64,   2)

        W_fc1 = weight_variable([2 * 19 * 19, (19 * 19) + 1])
        b_fc1 = bias_variable([(19 * 19) + 1])

        h_conv4_flat = tf.reshape(conv4, [-1, 2*19*19])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)

        return h_fc1
