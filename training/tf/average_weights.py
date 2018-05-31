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
import numpy as np

def swa(inputs, output, weights=None):
    """ Average weights of the weight files.

    inputs : List of filenames to use as inputs
    output : String of output filename
    weights : List of numbers to use for weighting the inputs
    """

    out_weights = []

    if weights == None:
        weights = [1.0]*len(inputs)

    if len(weights) != len(inputs):
        raise ValueError("Number of weights doesn't match number of input files")

    # Normalize weights
    weights = [float(w)/sum(weights) for w in weights]

    for count, filename in enumerate(inputs):
        with open(filename, 'r') as f:
            weights_in = []
            for line in f:
                weights_in.append(weights[count] * np.array(list(map(float, line.split(' ')))))
            if count == 0:
                out_weights = weights_in
            else:
                if len(out_weights) != len(weights_in):
                    raise ValueError("Nets have different sizes")
                for e, w in enumerate(weights_in):
                    if len(w) != len(out_weights[e]):
                        raise ValueError("Nets have different sizes")
                    out_weights[e] += w

    with open(output, 'w') as f:
        for e, w in enumerate(out_weights):
            if e == 0:
                #Version
                f.write('1\n')
            else:
                f.write(' '.join(map(str, w)) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Average weight files.')
    parser.add_argument('-i', '--inputs', nargs='+',
                        help='List of input weight files')
    parser.add_argument('-w', '--weights', type=float, nargs='+',
                        help='List of weights to use for the each weight file during averaging.')
    parser.add_argument('-o', '--output', help='Output filename')

    args = parser.parse_args()

    swa(args.inputs, args.output, args.weights)
