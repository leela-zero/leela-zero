#!/usr/bin/env python3
import numpy as np
import sys
import torch

net = torch.load(sys.argv[1])
state = net['state_dict']

def tensor_to_str(t):
    return ' '.join(map(str, np.array(t).flatten()))

def convert_block(t, name):
    weight = np.array(t[name + '.0.weight'])
    bias = np.array(t[name + '.0.bias'])
    bn_gamma = np.array(t[name + '.1.weight'])
    bn_beta = np.array(t[name + '.1.bias'])
    bn_mean = np.array(t[name + '.1.running_mean'])
    bn_var = np.array(t[name + '.1.running_var'])

    # y1 = weight * x + bias
    # y2 = gamma * (y1 - mean) / sqrt(var + e) + beta

    # convolution: [out, in, x, y]

    weight *= bn_gamma[:, np.newaxis, np.newaxis, np.newaxis]

    bias = bn_gamma * bias + bn_beta * np.sqrt(bn_var + 1e-5)

    bn_mean *= bn_gamma

    return [weight, bias, bn_mean, bn_var]

def write_block(f, b):
    for w in b:
        f.write(' '.join(map(str, w.flatten())) + '\n')

if 0:
    for key in state.keys():
        print(key, state[key].shape)

with open('elf_converted_weights.txt', 'w') as f:
    # version 2 means value head is for black, not for side to move
    f.write('2\n')
    if 'init_conv.0.weight' in state:
        b = convert_block(state, 'init_conv')
    else:
        b = convert_block(state, 'init_conv.module')

    # Permutate input planes
    p = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 16, 17]

    b[0] = b[0][:,p,:,:]

    write_block(f, b)
    for block in range(20):
        b = convert_block(state, 'resnet.module.resnet.{}.conv_lower'.format(block))
        write_block(f, b)
        b = convert_block(state, 'resnet.module.resnet.{}.conv_upper'.format(block))
        write_block(f, b)
    b = convert_block(state, 'pi_final_conv')
    write_block(f, b)
    f.write(tensor_to_str(state['pi_linear.weight']) + '\n')
    f.write(tensor_to_str(state['pi_linear.bias']) + '\n')
    b = convert_block(state, 'value_final_conv')
    write_block(f, b)
    f.write(tensor_to_str(state['value_linear1.weight']) + '\n')
    f.write(tensor_to_str(state['value_linear1.bias']) + '\n')
    f.write(tensor_to_str(state['value_linear2.weight']) + '\n')
    f.write(tensor_to_str(state['value_linear2.bias']) + '\n')
