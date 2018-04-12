#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017 Henrik Forsten
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
import numpy as np
import scipy.signal as signal
from copy import deepcopy

def convolve(w, x , bn=None, bn_epsilon=1e-5):
    x_ch, x_w, x_h = x.shape
    outputs, inputs, _, __ = w.shape
    assert x_ch == inputs
    res = np.zeros((outputs, x_w, x_h))
    for o in range(outputs):
        for c in range(inputs):
            res[o,:,:] += signal.correlate2d(x[c,:,:], w[o,c,:,:], mode='same')
    if bn == None:
        return res
    bn_means = bn[0]
    bn_vars = bn[1]
    for o in range(outputs):
        scale = 1.0 / np.sqrt(bn_epsilon + bn_vars[o])
        v = scale * (res[o,:,:] - bn_means[o])
        res[o,:,:] = np.maximum(v, 0)
    return res

def read_net(filename):
    with open(filename, 'r') as f:
        weights = []
        for e, line in enumerate(f):
            if e == 0:
                print("Version", line.strip())
                if line != '1\n':
                    raise ValueError("Unknown version {}".format(line.strip()))
            else:
                weights.append(list(map(float, line.split(' '))))
            if e == 2:
                channels = len(line.split(' '))
                print("Channels", channels)
        blocks = e - (4 + 14)
        if blocks % 8 != 0:
            raise ValueError("Inconsistent number of weights in the file")
        blocks //= 8
        print("Blocks", blocks)

        return blocks, channels, weights

def conv_bn_wider(weights, next_weights, inputs, channels,
                  new_channels, noise_std=0, last_block=False,
                  rand=None, dir_alpha=None, verify=False):

    if new_channels == 0:
        return weights, next_weights

    if rand == None:
        rand = list(range(channels))
        rand.extend(np.random.randint(0, channels, new_channels))
    rep_factor = np.bincount(rand)

    factor = np.zeros(len(rand))

    #In the net2net paper every input weight was weighted equally,
    #but in general we can have unequal division of the weights
    if dir_alpha == None:
        #Equal division
        for i in range(len(rand)):
            factor[i] = 1.0/rep_factor[rand[i]]
    else:
        #Unequal input weighting determined by dirichlet distribution
        for i in range(channels):
            x = np.random.dirichlet([dir_alpha]*rep_factor[i])
            e = 0
            for j in range(channels + new_channels):
                if rand[j] == i:
                    factor[j] = x[e]
                    e += 1

    #Widen the current layer
    w_conv_new = np.array(weights[0]).reshape(channels, inputs, 3, 3)[rand, :, :, :]
    bias = np.array(weights[1])[rand]
    w_bn_means = np.array(weights[2])[rand]
    w_bn_vars = np.array(weights[3])[rand]

    #Widen the next layer inputs
    if not last_block:
        w_filter = 3
    else:
        w_filter = 1

    next_weights_new = []
    for j in range(len(next_weights)):
        n = np.array(next_weights[j]).reshape(-1, channels, w_filter, w_filter)
        next_weights_new.append(n[:, rand, :, :])

        for i in range(len(rand)):
            noise = 0
            if i >= channels:
                noise = np.random.normal(0, noise_std)
            next_weights_new[j][:, i, :, :] *= (1.0 + noise)*factor[i]

    if noise_std == 0 and verify:
        x = np.random.random((inputs, 19, 19))
        old1 = convolve(np.array(weights[0]).reshape(channels, inputs, 3, 3), x, bn=[weights[2], weights[3]])
        old2 = convolve(np.array(next_weights[0]).reshape(-1, channels, w_filter, w_filter), old1)
        new1 = convolve(np.array(w_conv_new).reshape(channels + new_channels, inputs, 3, 3), x, bn=[w_bn_means, w_bn_vars])
        new2 = convolve(np.array(next_weights_new[0]).reshape(-1, channels + new_channels, w_filter, w_filter), new1)
        assert (np.abs(old2 - new2) < 1e-6).all()

    w_conv_new = w_conv_new.flatten()
    for j in range(len(next_weights)):
        next_weights_new[j] = next_weights_new[j].flatten()

    w_new = [w_conv_new, bias, w_bn_means, w_bn_vars]

    return w_new, next_weights_new

def write_layer(weights, out_file):
    for w in weights:
        out_file.write(' '.join(map(str,w)) + '\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add filters/blocks to existing network such that the output is preserved.')
    parser.add_argument("blocks", help="Residual blocks to add", type=int)
    parser.add_argument("filters", help="Filters to add", type=int)
    parser.add_argument("network", help="Input network", type=str)
    parser.add_argument("--noise", nargs='?', help="Standard deviation of noise to add to new filters/blocks. Default: 5e-3",
            default=5e-3, type=float)
    parser.add_argument("--dir_alpha", nargs='?', help=\
        """Dirichlet distribution parameter for input weight distribution for replicated channels. """\
        """Larger values divide input values more equally. """\
        """Smaller ones give one large input weight while others are very small. """\
        """You probably want this to be at least 1 to avoid near zero weights. """\
        """Set to 0 to divide input weights equally. Default: 10""",
        default=10, type=float)
    parser.add_argument("--verify", help="Verify that output matches. Noise must be disabled.",
            default=False, action='store_true')
    parser.add_argument("--add_inputs", help="Adds input planes to network",
            default=0, type=int)

    args = parser.parse_args()
    new_blocks = args.blocks
    new_channels = args.filters
    net_filename = args.network
    noise_std = args.noise
    dir_alpha = args.dir_alpha
    verify = args.verify

    if dir_alpha <= 0:
        dir_alpha = None

    if verify and noise_std != 0:
        raise ValueError("Noise must be zero if verify is enabled.")

    base, ext = os.path.splitext(net_filename)
    output_filename = base + "_net2net" + ext
    blocks, channels, weights = read_net(net_filename)

    if new_blocks < 0:
        raise ValueError("Blocks must be non-negative")

    if new_channels < 0:
        raise ValueError("Filters must be non-negative")

    print("Output will have {} blocks and {} channels.".format(
        blocks+new_blocks, channels+new_channels))

    input_planes = 18

    #Input convolution, bias, batch norm means, batch norm variances
    w_input = weights[:4]

    #Residual block convolution + batch norm
    w_convs = []
    for b in range(2*blocks):
        w_convs.append(weights[4 + b*4: 4 + (b+1)*4])

    i = ((b+1)*4) + 4
    w_pol = weights[i:i+6]
    w_val = weights[i+6:]

    if new_blocks > 0:
        #New blocks must have zero output due to the residual connection
        new_block_conv = np.random.normal(0, noise_std, 9*(channels)**2)
        new_block_bias = np.zeros(channels)
        new_block_bn_mean = new_block_bias.copy()
        new_block_bn_variances = np.ones(channels)
        new_block = [new_block_conv, new_block_bias, new_block_bn_mean, new_block_bn_variances]

        for i in range(2*new_blocks):
            w_convs.append(deepcopy(new_block))

        blocks += new_blocks

    out_file = open(output_filename, 'w')

    #Version
    out_file.write('1\n')

    #Making widening choice deterministic allows residual connection to be left
    #as identity map. If the choice is not deterministic then the output of the
    #widened network doesn't match the original one.
    rand = list(range(channels))
    rand.extend(np.random.randint(0, channels, new_channels))

    if args.add_inputs > 0:
        w_in_new = np.array(w_input[0]).reshape(channels, input_planes, 3, 3)
        noise = np.random.normal(0, noise_std, [channels, args.add_inputs, 3, 3])
        w_in_new = np.append(w_in_new, noise, axis=1)

        input_planes = input_planes + args.add_inputs
        print("Output will have {} input planes".format(input_planes))

        w_input[0] = w_in_new.flatten()

    #Input
    w_wider, conv_next = conv_bn_wider(w_input, [w_convs[0][0]], input_planes,
            channels, new_channels, noise_std, rand=rand, dir_alpha=dir_alpha, verify=verify)
    w_convs[0][0] = conv_next[0]

    write_layer(w_wider, out_file)

    for e, w in enumerate(w_convs[:-1]):
        r = rand
        if e % 2 == 0:
            print("Processing block", 1 + e//2)
            #First convolution in residual block can be widened randomly
            r = None

        w_wider, conv_next = conv_bn_wider(w, [w_convs[e+1][0]], channels + new_channels,
                channels, new_channels, noise_std, rand=r, dir_alpha=dir_alpha, verify=verify)
        w_convs[e+1][0] = conv_next[0]
        write_layer(w_wider, out_file)

    #The last block is special case because of policy and value heads
    w_wider, w_next = conv_bn_wider(w_convs[-1], [w_pol[0], w_val[0]], channels + new_channels,
            channels, new_channels, noise_std, last_block=True, rand=rand, dir_alpha=dir_alpha, verify=verify)
    w_pol[0] = w_next[0]
    w_val[0] = w_next[1]

    write_layer(w_wider, out_file)

    write_layer(w_pol, out_file)
    write_layer(w_val, out_file)

    out_file.close()
