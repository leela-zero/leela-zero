#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
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


from tfprocess import TFProcess
from chunkparser import ChunkParser
import binascii
import glob
import gzip
import itertools
import math
import multiprocessing as mp
import numpy as np
import os
import queue
import random
import shufflebuffer as sb
import struct
import sys
import tensorflow as tf
import time
import threading
import unittest

# 16 planes, 1 side to move, 1 x 362 probs, 1 winner = 19 lines
DATA_ITEM_LINES = 16 + 1 + 1 + 1

# Sane values are from 4096 to 64 or so. The maximum depends on the amount
# of RAM in your GPU and the network size. You need to adjust the learning rate
# if you change this.
BATCH_SIZE = 512

# Use a random sample of 1/16th of the input data read. This helps
# improve the spread of games in the shuffle buffer.
DOWN_SAMPLE = 16

def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")

class FileDataSrc:
    """
        data source yielding chunkdata from chunk files.
    """
    def __init__(self, chunks):
        self.chunks = []
        self.done = chunks
    def next(self):
        if not self.chunks:
            self.chunks, self.done = self.done, self.chunks
            random.shuffle(self.chunks)
        if not self.chunks:
            return None
        while len(self.chunks):
            filename = self.chunks.pop()
            try:
                with gzip.open(filename, 'rb') as chunk_file:
                    self.done.append(filename)
                    return chunk_file.read()
            except:
                print("failed to parse {}".format(filename))

def benchmark(parser):
    """
        Benchmark for parser
    """
    gen = parser.parse()
    batch=100
    while True:
        start = time.time()
        for _ in range(batch):
            next(gen)
        end = time.time()
        print("{} pos/sec {} secs".format( BATCH_SIZE * batch / (end - start), (end - start)))

def benchmark1(t):
    """
        Benchmark for full input pipeline, including tensorflow conversion
    """
    batch=100
    while True:
        start = time.time()
        for _ in range(batch):
            t.session.run([t.next_batch],
                feed_dict={t.training: True, t.handle: t.train_handle})

        end = time.time()
        print("{} pos/sec {} secs".format( BATCH_SIZE * batch / (end - start), (end - start)))


def split_chunks(chunks, test_ratio):
    splitpoint = 1 + int(len(chunks) * (1.0 - test_ratio))
    return (chunks[:splitpoint], chunks[splitpoint:])

def _parse_function(planes, probs, winner):
    planes = tf.decode_raw(planes, tf.uint8)
    probs = tf.decode_raw(probs, tf.float32)
    winner = tf.decode_raw(winner, tf.float32)

    planes = tf.to_float(planes)

    planes = tf.reshape(planes, (BATCH_SIZE, 18, 19*19))
    probs = tf.reshape(probs, (BATCH_SIZE, 19*19 + 1))
    winner = tf.reshape(winner, (BATCH_SIZE, 1))

    return (planes, probs, winner)

def main(args):
    train_data_prefix = args.pop(0)

    chunks = get_chunks(train_data_prefix)
    print("Found {0} chunks".format(len(chunks)))

    if not chunks:
        return

    # The following assumes positions from one game are not
    # spread through chunks.
    random.shuffle(chunks)
    training, test = split_chunks(chunks, 0.1)
    print("Training with {0} chunks, validating on {1} chunks".format(
        len(training), len(test)))

    train_parser = ChunkParser(FileDataSrc(training),
        shuffle_size=1<<19, sample=DOWN_SAMPLE, batch_size=BATCH_SIZE)
    #benchmark(train_parser)
    dataset = tf.data.Dataset.from_generator(
        train_parser.parse, output_types=(tf.string, tf.string, tf.string))
    dataset = dataset.map(_parse_function)
    dataset = dataset.prefetch(4)
    train_iterator = dataset.make_one_shot_iterator()

    test_parser = ChunkParser(FileDataSrc(test),
        shuffle_size=1<<19, sample=DOWN_SAMPLE, batch_size=BATCH_SIZE)
    dataset = tf.data.Dataset.from_generator(
        test_parser.parse, output_types=(tf.string, tf.string, tf.string))
    dataset = dataset.map(_parse_function)
    dataset = dataset.prefetch(4)
    test_iterator = dataset.make_one_shot_iterator()

    tfprocess = TFProcess()
    tfprocess.init(dataset, train_iterator, test_iterator)

    #benchmark1(tfprocess)

    if args:
        restore_file = args.pop(0)
        tfprocess.restore(restore_file)
    while True:
        tfprocess.process(BATCH_SIZE)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main(sys.argv[1:])
    mp.freeze_support()

# Tests.
# To run: python3 -m unittest parse.TestParse
class TestParse(unittest.TestCase):
    def test_datasrc(self):
        # create chunk files
        num_chunks = 3
        chunks = []
        for x in range(num_chunks):
            filename = '/tmp/parse-unittest-chunk'+str(x)+'.gz'
            chunk_file = gzip.open(filename, 'w', 1)
            chunk_file.write(bytes(x))
            chunk_file.close()
            chunks.append(filename)
        # create a data src, passing a copy of the
        # list of chunks.
        ds = FileDataSrc(list(chunks))
        # get sample of 200 chunks from the data src
        counts={}
        for _ in range(200):
            data = ds.next()
            if data in counts:
                counts[data] += 1
            else:
                counts[data] = 1
        # Every chunk appears at least thrice. Note! This is probabilistic
        # but the probably of false failure is < 1e-9
        for x in range(num_chunks):
            self.assertGreater(counts[bytes(x)], 3)
        # check that there are no stray chunks
        self.assertEqual(len(counts.keys()), num_chunks)
        # clean up: remove temp files.
        for c in chunks:
            os.remove(c)
