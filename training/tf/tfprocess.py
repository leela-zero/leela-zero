#!/usr/bin/env python3

import os
import numpy as np
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

def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW',
                        strides=[1, 1, 1, 1], padding='SAME')

class TFProcess:
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.session = tf.Session(config=config)

        # For exporting
        self.weights = []

        # TF variables
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.x = tf.placeholder(tf.float32, [None, 18, 19 * 19])
        self.y_ = tf.placeholder(tf.float32, [None, 362])
        self.z_ = tf.placeholder(tf.float32, [None, 1])
        self.training = tf.placeholder(tf.bool)
        self.batch_norm_count = 0
        self.y_conv, self.z_conv = self.construct_net(self.x)

        # Calculate loss on policy head
        cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                    logits=self.y_conv)
        self.policy_loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('policy_loss', self.policy_loss)

        # Loss on value head
        self.mse_loss = \
            tf.reduce_mean(tf.squared_difference(self.z_, self.z_conv))
        tf.summary.scalar('mse_loss', self.mse_loss)

        # Regularizer
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        reg_variables = tf.trainable_variables()
        reg_term = \
            tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        loss = 1.0 * self.policy_loss + 1.0 * self.mse_loss + reg_term

        opt_op = tf.train.MomentumOptimizer(
            learning_rate=0.05, momentum=0.9, use_nesterov=True)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = \
                opt_op.minimize(loss, global_step=self.global_step)

        correct_prediction = \
            tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar('accuracy', self.accuracy)

        self.avg_policy_loss = None
        self.avg_mse_loss = None

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        self.session.run(self.init)

    def restore(self, file):
        print("Restoring from {0}".format(file))
        self.saver.restore(self.session, file)

    def process(self, batch):
        # Run training for this batch
        _, policy_loss, mse_loss = self.session.run(
            [self.train_op, self.policy_loss, self.mse_loss],
            feed_dict={self.x: batch[0],
                       self.y_: batch[1],
                       self.z_: batch[2],
                       self.training: True})
        steps = tf.train.global_step(self.session, self.global_step)
        # Keep running averages
        # XXX: use built-in support like tf.moving_average_variables
        if self.avg_policy_loss:
            self.avg_policy_loss = 0.99 * self.avg_policy_loss + 0.01 * policy_loss
        else:
            self.avg_policy_loss = policy_loss
        if self.avg_mse_loss:
            self.avg_mse_loss = 0.99 * self.avg_mse_loss + 0.01 * mse_loss
        else:
            self.avg_mse_loss = mse_loss
        if steps % 100 == 0:
            print("step {0}, policy loss={1} mse={2}".format(
                steps, self.avg_policy_loss, self.avg_mse_loss))
        # Ideally this would use a seperate dataset and so on...
        if steps % 1000 == 0:
            train_accuracy = \
                self.accuracy.eval(session=self.session,
                                   feed_dict={self.x: batch[0],
                                              self.y_: batch[1],
                                              self.z_: batch[2],
                                              self.training: False})
            train_mse = \
                self.mse_loss.eval(session=self.session,
                                   feed_dict={self.x: batch[0],
                                              self.y_: batch[1],
                                              self.z_: batch[2],
                                              self.training: False})
            print("step {0}, training accuracy={1}%, mse={2}".format(
                steps, train_accuracy*100.0, train_mse))
            path = os.path.join(os.getcwd(), "leelaz-model")
            save_path = self.saver.save(self.session, path, global_step=steps)
            print("Model saved in file: {}".format(save_path))
            leela_path = path + ".txt"
            self.save_leelaz_weights(leela_path)
            print("Leela weights saved to {}".format(leela_path))

    def save_leelaz_weights(self, filename):
        with open(filename, "w") as file:
            # Version tag
            file.write("1")
            for weights in self.weights:
                # Newline unless last line (single bias)
                file.write("\n")
                work_weights = None
                # Keyed batchnorm weights
                if isinstance(weights, str):
                    work_weights = tf.get_default_graph().get_tensor_by_name(weights)
                elif weights.shape.ndims == 4:
                    # Convolution weights need a transpose
                    # TF
                    # [filter_height, filter_width, in_channels, out_channels]
                    # Leela
                    # [output, input, filter_size, filter_size]
                    work_weights = tf.transpose(weights, [3, 2, 0, 1])
                else:
                    # Fully connected layers are [in, out] in both
                    # As are biases etc
                    work_weights = weights
                nparray = work_weights.eval(session=self.session)
                wt_str = [str(wt) for wt in np.ravel(nparray)]
                file.write(" ".join(wt_str))

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def conv_block(self, inputs, input_channels, output_channels):
        W_conv = weight_variable([3, 3, input_channels, output_channels])
        b_conv = bn_bias_variable([output_channels])
        self.weights.append(W_conv)
        self.weights.append(b_conv)
        # The weights are internal to the batchnorm layer, so apply
        # a unique scope that we can store, and use to look them back up
        # later on.
        weight_key = self.get_batchnorm_key()
        self.weights.append(weight_key + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key):
            h_bn = \
                tf.layers.batch_normalization(
                    tf.nn.bias_add(conv2d(inputs, W_conv),
                                          b_conv, data_format='NCHW'),
                                   epsilon=1e-5, axis=1, fused=True,
                                   center=False, scale=False,
                                   training=self.training)
        h_conv = tf.nn.relu(h_bn)
        return h_conv

    def residual_block(self, inputs, channels):
        # First convnet
        W_conv_1 = weight_variable([3, 3, channels, channels])
        b_conv_1 = bn_bias_variable([channels])
        self.weights.append(W_conv_1)
        self.weights.append(b_conv_1)
        weight_key_1 = self.get_batchnorm_key()
        self.weights.append(weight_key_1 + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key_1 + "/batch_normalization/moving_variance:0")

        # Second convnet
        W_conv_2 = weight_variable([3, 3, channels, channels])
        b_conv_2 = bn_bias_variable([channels])
        self.weights.append(W_conv_2)
        self.weights.append(b_conv_2)
        weight_key_2 = self.get_batchnorm_key()
        self.weights.append(weight_key_2 + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key_2 + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key_1):
            h_bn1 = \
                tf.layers.batch_normalization(
                    tf.nn.bias_add(conv2d(inputs, W_conv_1),
                                   b_conv_1, data_format='NCHW'),
                                   epsilon=1e-5, axis=1, fused=True,
                                   center=False, scale=False,
                                   training=self.training)
        h_out_1 = tf.nn.relu(h_bn1)
        with tf.variable_scope(weight_key_2):
            h_bn2 = \
                tf.layers.batch_normalization(
                    tf.nn.bias_add(conv2d(h_out_1, W_conv_2),
                                          b_conv_2, data_format='NCHW'),
                                   epsilon=1e-5, axis=1, fused=True,
                                   center=False, scale=False,
                                   training=self.training)
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
        self.weights.append(W_fc1)
        self.weights.append(b_fc1)
        h_fc1 = tf.nn.relu(tf.matmul(h_conv8_flat, W_fc1) + b_fc1)

        # Value head
        conv9 = self.conv_block(conv7, 128, 1)
        h_conv9_flat = tf.reshape(conv9, [-1, 19*19])
        W_fc2 = weight_variable([19 * 19, 256])
        b_fc2 = bias_variable([256])
        self.weights.append(W_fc2)
        self.weights.append(b_fc2)
        h_fc2 = tf.nn.relu(tf.matmul(h_conv9_flat, W_fc2) + b_fc2)
        W_fc3 = weight_variable([256, 1])
        b_fc3 = bias_variable([1])
        self.weights.append(W_fc3)
        self.weights.append(b_fc3)
        h_fc3 = tf.nn.tanh(tf.matmul(h_fc2, W_fc3) + b_fc3)

        return h_fc1, h_fc3
