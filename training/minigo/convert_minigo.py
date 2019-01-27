#!/usr/bin/env python3
import re
import os
import sys

import numpy as np
import tensorflow as tf

# Hide boring TF log statements
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}


def matches(name, parts):
    return all(part in name for part in parts)

def deduped(names):
    names = [re.sub('_\d+', '', name) for name in names]
    return sorted([(n, names.count(n)) for n in set(names)])

def getMinigoWeightsV1(model):
    """Load and massage Minigo weights to Leela format.

    This version works on older models (v9 or before)
    But was broken when conv bias was removed in v10
    See: https://github.com/tensorflow/minigo/pull/292 and
         https://github.com/gcp/leela-zero/issues/2020
    """
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model+'.meta')
    saver.restore(sess, model)

    trainable_names = []
    for v in tf.trainable_variables():
        trainable_names.append(v.name)

    weights = []
    for v in tf.global_variables():
        if v.name in trainable_names:
            weights.append(v)
        elif 'batch_normalization' in v.name:
            # Moving mean and variance are not trainable, but are needed for the model
            if 'moving_mean' in v.name or 'moving_variance' in v.name:
                weights.append(v)

    # To match the format of V2
    weights_v2_format = []
    for w in weights:
        nparray = w.eval(session=sess)
        weights_v2_format.append((w.name, nparray))
    return weights_v2_format

def getMinigoWeightsV2(model):
    """Load and massage Minigo weights to Leela format.

    This version works on older models (v9 or before)
    But was broken when conv bias was removed in v10
    See: https://github.com/tensorflow/minigo/pull/292 and
         https://github.com/gcp/leela-zero/issues/2020
    """
    var_names = tf.train.load_checkpoint(model).get_variable_to_dtype_map()

    # count() overcounts by 3 from policy/value head and each layer has two convolutions.
    layers = (max([count for n, count in deduped(var_names)]) - 3) // 2
    print (layers, 'layers')

    has_conv_bias = any(matches(name, ('conv2d', 'bias')) for name in var_names.keys())
    if not has_conv_bias:
        print('Did not find conv bias in this model, using all zeros')
    empty_conv_bias = tf.constant([], name='placeholder_for_conv_bias')

    # 2 * layer copies of
    #   6*n + 0: conv2d/kernel:0
    #   6*n + 1: conv2d/bias:0
    #   6*n + 2: batch_normalization/gamma:0
    #   6*n + 3: batch_normalization/beta:0
    #   6*n + 4: batch_normalization/moving_mean:0
    #   6*n + 5: batch_normalization/moving_variance:0
    # at the end 2x
    #   conv2d_39/kernel:0
    #   conv2d_39/bias:0
    #   batch_normalization_39/moving_mean:0
    #   batch_normalization_39/moving_variance:0
    #   dense/kernel:0
    #   dense/bias:0
    # final value dense
    #   dense_2/kernel:0
    #   dense_2/bias:0

    weight_names = []

    def tensor_number(number):
        return '' if number ==0 else '_' + str(number)

    def add_conv(number, with_gamma=True):
        number = tensor_number(number)
        weight_names.append('conv2d{}/kernel:0'.format(number))
        weight_names.append('conv2d{}/bias:0'.format(number))
        if with_gamma:
            weight_names.append('batch_normalization{}/gamma:0'.format(number))
            weight_names.append('batch_normalization{}/beta:0'.format(number))
        weight_names.append('batch_normalization{}/moving_mean:0'.format(number))
        weight_names.append('batch_normalization{}/moving_variance:0'.format(number))

    def add_dense(number):
        number = tensor_number(number)
        weight_names.append('dense{}/kernel:0'.format(number))
        weight_names.append('dense{}/bias:0'.format(number))

    # This blindly builds the correct names for the tensors.
    for l in range(2 * layers + 1):
        add_conv(l)

    add_conv(2 * layers + 1, with_gamma=False)
    add_dense(0)
    add_conv(2 * layers + 2, with_gamma=False)
    add_dense(1)
    add_dense(2)

    # This tries to load the data for each tensors.
    weights = []
    for i, name in enumerate(weight_names):
        if matches(name, ('conv2d', 'bias')) and not has_conv_bias:
            w = np.zeros(weights[-1][1].shape[-1:])
        else:
            w = tf.train.load_variable(model, name)

#        print ("{:45} {} {}".format(name, type(w), w.shape))
        weights.append((name, w))
    return weights

def merge_gammas(weights):
    out_weights = []
    skip = 0
    for e, (name, w) in enumerate(weights):
        if skip > 0:
            skip -= 1
            continue

        if matches(name, ('conv2d', 'kernel')) and 'gamma' in weights[e+2][0]:
            kernel = w
            bias = weights[e+1][1]
            gamma = weights[e+2][1]
            beta = weights[e+3][1]
            mean = weights[e+4][1]
            var = weights[e+5][1]

            new_kernel = kernel * np.reshape(gamma, (1, 1, 1, -1))
            new_bias = gamma * bias + beta * np.sqrt(var + 1e-5)
            new_mean = mean * gamma

            out_weights.append(new_kernel)
            out_weights.append(new_bias)
            out_weights.append(new_mean)
            out_weights.append(var)

            skip = 5

        elif matches(name, ('dense', 'kernel')):
            # Minigo uses channels last order while LZ uses channels first,
            # Do some surgery for the dense layers to make the output match.
            planes = w.shape[0] // 361
            if planes > 0:
                w1 = np.reshape(w, [19, 19, planes, -1])
                w2 = np.transpose(w1, [2, 0, 1, 3])
                new_kernel = np.reshape(w2, [361*planes, -1])
                out_weights.append(new_kernel)
            else:
                out_weights.append(w)
        else:
            out_weights.append(w)

    return out_weights

def save_leelaz_weights(filename, weights):
    with open(filename, 'w') as file:
        # Version tag
        # Minigo outputs winrate from blacks point of view (same as ELF)
        file.write('2')
        for e, w in enumerate(weights):
            # Newline unless last line (single bias)
            file.write('\n')
            work_weights = None
            if len(w.shape) == 4:
                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                work_weights = np.transpose(w, [3, 2, 0, 1])
            elif len(w.shape) == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                work_weights = np.transpose(w, [1, 0])
            else:
                # Biases, batchnorm etc
                work_weights = w
            if e == 0:
                # Fix input planes
                #
                # Add zero weights for white to play input plane
                work_weights = np.pad(work_weights, ((0, 0), (0, 1), (0, 0), (0, 0)), 'constant', constant_values=0)

                # Permutate weights
                p = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 16, 17]

                work_weights = work_weights[:, p, :, :]
            wt_str = ["{:0.8g}".format(wt) for wt in np.ravel(work_weights)]
            file.write(' '.join(wt_str))


if len(sys.argv) < 2:
    print('Model filename without extension needed as an argument.')
    exit()

model = sys.argv[1]

print ('loading ', model)
print ()

# Can be used for v9 or before models.
# weights = getMinigoWeightsV1(model)
weights = getMinigoWeightsV2(model)

if 0:
    for name, variables in [
            ('load_checkpoint', var_names.keys()),
    #        ('trainable_names', trainable_names),
    #        ('global_variable', [v.name for v in tf.global_variables()])
            ]:
        print (name, len(variables))
        print (deduped(variables))
        print ()

save_leelaz_weights(model + '_converted.txt', merge_gammas(weights))
