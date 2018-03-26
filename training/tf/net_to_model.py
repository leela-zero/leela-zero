#!/usr/bin/env python3
import tensorflow as tf
import os
import sys
from tfprocess import TFProcess, read_weights


if __name__ == "__main__":
    version, blocks, channels, weights = read_weights(sys.argv[1])

    if version == None:
        raise ValueError("Unable to read version number")

    print("Version", version)
    print("Channels", channels)
    print("Blocks", blocks)

    x = [
        tf.placeholder(tf.float32, [None, 18, 19 * 19]),
        tf.placeholder(tf.float32, [None, 362]),
        tf.placeholder(tf.float32, [None, 1])
        ]

    tfprocess = TFProcess()
    tfprocess.init_net(x)
    if tfprocess.RESIDUAL_BLOCKS != blocks:
        raise ValueError("Number of blocks in tensorflow model doesn't match "\
                "number of blocks in input network")
    if tfprocess.RESIDUAL_FILTERS != channels:
        raise ValueError("Number of filters in tensorflow model doesn't match "\
                "number of filters in input network")
    tfprocess.replace_weights(weights)
    path = os.path.join(os.getcwd(), "leelaz-model")
    save_path = tfprocess.saver.save(tfprocess.session, path, global_step=0)
