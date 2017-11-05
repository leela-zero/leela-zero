#!/usr/bin/env python3

import os
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Bias weights for layers not followed by BatchNorm
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

# No point in learning bias weights as they are cancelled
# out by the BatchNorm layers's mean adjustment.
def bn_bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, trainable=False)

# Offsets after BN are at least not immediately cancelled out,
# but empirically they do not help much if anything.
# By default, without offset, about half the ReLU units after a BN fire,
# and this is pretty optimal. Leela doesn't incorporate these right now.
# They could be added if they turn out to be useful anyway.
def offset_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, trainable=False)

# The ReLU non-linearity is actually linear wrt scaling factors,
# so any scaling here could be absorbed into the next layers'
# convolution weights. No point in learning a scale factor.
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
        self.z_ = tf.placeholder(tf.float32, [None, 1])
        self.y_conv, self.z_conv = self.construct_net(self.x)

        self.cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                    logits=self.y_conv)
        self.policy_loss = tf.reduce_mean(self.cross_entropy)

        self.mse_loss = \
            tf.reduce_mean(tf.squared_difference(self.z_, self.z_conv))

        self.loss = 1.0 * self.policy_loss + 1.0 * self.mse_loss

        self.train_step = \
            tf.train.MomentumOptimizer(
                learning_rate=0.1, momentum=0.9, use_nesterov=True).\
                minimize(self.loss)

        self.correct_prediction = \
            tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.correct_prediction = \
            tf.cast(self.correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)
        self.mse = tf.reduce_mean(tf.squared_difference(self.z_, self.z_conv))

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        self.session.run(self.init)

    def process(self, batch):
        self.steps += 1
        if self.steps % 100 == 0:
            train_loss = \
                self.loss.eval(session=self.session,
                               feed_dict={self.x: batch[0],
                                          self.y_: batch[1],
                                          self.z_: batch[2]})
            print("step {0}, batch loss={1}".format(self.steps, train_loss))
        self.train_step.run(session=self.session,
                            feed_dict={self.x: batch[0],
                                       self.y_: batch[1],
                                       self.z_: batch[2]})

        if self.steps % 1000 == 0:
            train_accuracy = \
                self.accuracy.eval(session=self.session,
                                   feed_dict={self.x: batch[0],
                                              self.y_: batch[1],
                                              self.z_: batch[2]})
            train_mse = \
                self.mse.eval(session=self.session,
                              feed_dict={self.x: batch[0],
                                         self.y_: batch[1],
                                         self.z_: batch[2]})
            print("step {0}, training accuracy={1}%, mse={2}".format(
                self.steps, train_accuracy*100.0, train_mse))
            path = os.path.join(os.getcwd(), "model_" + str(self.steps) + ".ckpt")
            save_path = self.saver.save(self.session, path)
            print("Model saved in file: {0}".format(save_path))

    def conv_block(self, inputs, input_channels, output_channels):
        W_conv = weight_variable([3, 3, input_channels, output_channels])
        b_conv = bn_bias_variable([output_channels])
        s_conv = scale_variable([output_channels])
        o_conv = offset_variable([output_channels])
        h_bn, bm, vn = \
            tf.nn.fused_batch_norm(tf.nn.bias_add(conv2d(inputs, W_conv),
                                                  b_conv, data_format='NCHW'),
                                   s_conv, o_conv, None, None, data_format='NCHW')
        h_conv = tf.nn.relu(h_bn)
        return h_conv

    def residual_block(self, inputs, channels):
        # First convnet
        W_conv_1 = weight_variable([3, 3, channels, channels])
        b_conv_1 = bn_bias_variable([channels])
        s_conv_1 = scale_variable([channels])
        o_conv_1 = offset_variable([channels])
        # Second convnet
        W_conv_2 = weight_variable([3, 3, channels, channels])
        b_conv_2 = bn_bias_variable([channels])
        s_conv_2 = scale_variable([channels])
        o_conv_2 = offset_variable([channels])
        h_bn1, bm1, vn1 = \
            tf.nn.fused_batch_norm(tf.nn.bias_add(conv2d(inputs, W_conv_1),
                                                  b_conv_1, data_format='NCHW'),
                                   s_conv_1, o_conv_1, None, None, data_format='NCHW')
        h_out_1 = tf.nn.relu(h_bn1)
        h_bn2, bm2, vn2 = \
            tf.nn.fused_batch_norm(tf.nn.bias_add(conv2d(h_out_1, W_conv_2),
                                                  b_conv_2, data_format='NCHW'),
                                   s_conv_2, o_conv_2, None, None, data_format='NCHW')
        h_out_2 = tf.nn.relu(h_bn2 + inputs)
        return h_out_2

    def construct_net(self, planes):
        # NCHW format
        # batch, 18 channels, 19 x 19
        x_planes = tf.reshape(planes, [-1, 18, 19, 19])

        conv1 = self.conv_block(x_planes, 18, 128)
        conv2 = self.residual_block(conv1, 128)
        conv3 = self.residual_block(conv2, 128)
        conv4 = self.residual_block(conv3, 128)
        conv5 = self.residual_block(conv4, 128)
        conv6 = self.residual_block(conv5, 128)
        conv7 = self.residual_block(conv6, 128)

        # Policy head
        conv8 = self.conv_block(conv7, 128, 2)
        h_conv8_flat = tf.reshape(conv8, [-1, 2*19*19])
        W_fc1 = weight_variable([2 * 19 * 19, (19 * 19) + 1])
        b_fc1 = bias_variable([(19 * 19) + 1])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv8_flat, W_fc1) + b_fc1)

        # Value head
        conv9 = self.conv_block(conv7, 128, 1)
        h_conv9_flat = tf.reshape(conv9, [-1, 19*19])
        W_fc2 = weight_variable([19 * 19, 256])
        b_fc2 = bias_variable([256])
        h_fc2 = tf.nn.relu(tf.matmul(h_conv9_flat, W_fc2) + b_fc2)
        W_fc3 = weight_variable([256, 1])
        b_fc3 = bias_variable([1])
        h_fc3 = tf.nn.tanh(tf.matmul(h_fc2, W_fc3) + b_fc3)

        return h_fc1, h_fc3
