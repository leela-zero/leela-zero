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
import argparse
import glob
import gzip
import multiprocessing as mp
import os
import random
import shufflebuffer as sb
import sys
import tensorflow as tf
import time
import unittest

# Sane values are from 4096 to 64 or so.
# You need to adjust the learning rate if you change this. Should be
# a multiple of RAM_BATCH_SIZE. NB: It's rare that large batch sizes are
# actually required.
BATCH_SIZE = 512
# Number of examples in a GPU batch. Higher values are more efficient.
# The maximum depends on the amount of RAM in your GPU and the network size.
# Must be smaller than BATCH_SIZE.
RAM_BATCH_SIZE = 128

# Use a random sample input data read. This helps improve the spread of
# games in the shuffle buffer.
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
        print("{} pos/sec {} secs".format(
            RAM_BATCH_SIZE * batch / (end - start), (end - start)))

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
        print("{} pos/sec {} secs".format(
            RAM_BATCH_SIZE * batch / (end - start), (end - start)))


def split_chunks(chunks, test_ratio):
    splitpoint = 1 + int(len(chunks) * (1.0 - test_ratio))
    return (chunks[:splitpoint], chunks[splitpoint:])

def main():
    parser = argparse.ArgumentParser(
        description='Train network from game data.')
    parser.add_argument("trainpref",
        help='Training file prefix', nargs='?', type=str)
    parser.add_argument("restorepref",
        help='Training snapshot prefix', nargs='?', type=str)
    parser.add_argument("--train", '-t',
        help="Training file prefix", type=str)
    parser.add_argument("--test", help="Test file prefix", type=str)
    parser.add_argument("--restore", type=str,
        help="Prefix of tensorflow snapshot to restore from")
    parser.add_argument("--logbase", default='leelalogs', type=str,
        help="Log file prefix (for tensorboard)")
    parser.add_argument("--sample", default=DOWN_SAMPLE, type=int,
        help="Rate of data down-sampling to use")
    args = parser.parse_args()

    train_data_prefix = args.train or args.trainpref
    restore_prefix = args.restore or args.restorepref

    training = get_chunks(train_data_prefix)
    if not args.test:
        # Generate test by taking 10% of the training chunks.
        random.shuffle(training)
        training, test = split_chunks(training, 0.1)
    else:
        test = get_chunks(args.test)

    if not training:
        print("No data to train on!")
        return

    print("Training with {0} chunks, validating on {1} chunks".format(
        len(training), len(test)))

    train_parser = ChunkParser(FileDataSrc(training),
                               shuffle_size=1<<20, # 2.2GB of RAM.
                               sample=args.sample,
                               batch_size=RAM_BATCH_SIZE).parse()

    test_parser = ChunkParser(FileDataSrc(test),
                              shuffle_size=1<<19,
                              sample=args.sample,
                              batch_size=RAM_BATCH_SIZE).parse()

    tfprocess = TFProcess()
    tfprocess.init(RAM_BATCH_SIZE,
                   logbase=args.logbase,
                   macrobatch=BATCH_SIZE // RAM_BATCH_SIZE)

    #benchmark1(tfprocess)

    if restore_prefix:
        tfprocess.restore(restore_prefix)
    tfprocess.process(train_parser, test_parser)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
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
