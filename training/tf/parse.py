#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017 Gian-Carlo Pascutto
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

import sys
import glob
import gzip
import random
import math
import multiprocessing as mp
import numpy as np
import time
import tensorflow as tf
from tfprocess import TFProcess

# 16 planes, 1 stm, 1 x 362 probs, 1 winner = 19 lines
DATA_ITEM_LINES = 16 + 1 + 1 + 1

BATCH_SIZE = 256

def remap_vertex(vertex, symmetry):
    """
        Remap a go board coordinate according to a symmetry.
    """
    assert vertex >= 0 and vertex < 361
    x = vertex % 19
    y = vertex // 19
    if symmetry >= 4:
        x, y = y, x
        symmetry -= 4
    if symmetry == 1 or symmetry == 3:
        x = 19 - x - 1
    if symmetry == 2 or symmetry == 3:
        y = 19 - y - 1
    return y * 19 + x

class ChunkParser:
    def __init__(self, chunks):
        # Build probility reflection tables. The last element is 'pass' and is identity mapped.
        self.prob_reflection_table = [[remap_vertex(vertex, sym) for vertex in range(361)]+[361] for sym in range(8)]
        # Build full 16-plane reflection tables.
        self.full_reflection_table = [
            [remap_vertex(vertex, sym) + p * 361 for p in range(16) for vertex in range(361) ]
                for sym in range(8) ]
        # Convert both to np.array. This avoids a conversion step when they're actually used.
        self.prob_reflection_table = [ np.array(x, dtype=np.int64) for x in self.prob_reflection_table ]
        self.full_reflection_table = [ np.array(x, dtype=np.int64) for x in self.full_reflection_table ]
        # Build the all-zeros and all-ones flat planes, used for color-to-move.
        self.flat_planes = [ b'\0' * 361, b'\1' * 361 ]

        # Start worker processes, leave 1 for TensorFlow
        workers = max(1, mp.cpu_count() - 1)
        print("Using {} worker processes.".format(workers))
        self.readers = []
        for _ in range(workers):
            read, write = mp.Pipe(False)
            mp.Process(target=self.task,
                       args=(chunks, write)).start()
            self.readers.append(read)

    def convert_train_data(self, text_item, symmetry):
        """
            Convert textual training data to a tf.train.Example

            Converts a set of 19 lines of text into a pythonic dataformat.
            [[plane_1],[plane_2],...],...
            [probabilities],...
            winner,...
        """
        assert symmetry >= 0 and symmetry < 8

        # We start by building a list of 16 planes, each being a 19*19 == 361 element array
        # of type np.uint8
        planes = []
        for plane in range(0, 16):
            # first 360 first bits are 90 hex chars, encoded MSB
            hex_string = text_item[plane][0:90]
            array = np.unpackbits(np.frombuffer(bytearray.fromhex(hex_string), dtype=np.uint8))
            # Remaining bit that didn't fit. Encoded LSB so
            # it needs to be specially handled.
            last_digit = text_item[plane][90]
            assert last_digit == "0" or last_digit == "1"
            # Apply symmetry and append
            planes.append(array)
            planes.append(np.array([last_digit], dtype=np.uint8))

        # We flatten to a single array of len 16*19*19, type=np.uint8
        planes = np.concatenate(planes)

        # We use the full length reflection tables to apply symmetry
        # to all 16 planes simultaneously
        planes = planes[self.full_reflection_table[symmetry]]
        # Convert the array to a byte string
        planes = [ planes.tobytes() ]

        # Now we add the two final planes, being the 'color to move' planes.
        # These already a fully symmetric, so we add them directly as byte
        # strings of length 361.
        stm = text_item[16][0]
        assert stm == "0" or stm == "1"
        stm = int(stm)
        planes.append(self.flat_planes[1 - stm])
        planes.append(self.flat_planes[stm])

        # Flatten all planes to a single byte string
        planes = b''.join(planes)
        assert len(planes) == (18 * 19 * 19)

        # Load the probabilities.
        probabilities = np.array(text_item[17].split()).astype(float)
        if np.any(np.isnan(probabilities)):
            # Work around a bug in leela-zero v0.3, skipping any
            # positions that have a NaN in the probabilities list.
            return False, None
        # Apply symmetries to the probabilities.
        probabilities = probabilities[self.prob_reflection_table[symmetry]]
        assert len(probabilities) == 362

        # Load the game winner color.
        winner = float(text_item[18])
        assert winner == 1.0 or winner == -1.0

        # Construct the Example protobuf
        example = tf.train.Example(features=tf.train.Features(feature={
            'planes' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[planes])),
            'probs' : tf.train.Feature(float_list=tf.train.FloatList(value=probabilities)),
            'winner' : tf.train.Feature(float_list=tf.train.FloatList(value=[winner]))}))
        return True, example.SerializeToString()

    def task(self, chunks, writer):
        while True:
            random.shuffle(chunks)
            for chunk in chunks:
                with gzip.open(chunk, 'r') as chunk_file:
                    file_content = chunk_file.readlines()
                    item_count = len(file_content) // DATA_ITEM_LINES
                    for item_idx in range(item_count):
                        pick_offset = item_idx * DATA_ITEM_LINES
                        item = file_content[pick_offset:pick_offset + DATA_ITEM_LINES]
                        str_items = [str(line, 'ascii') for line in item]
                        # Pick a random symmetry to apply
                        symmetry = random.randrange(8)
                        success, data = self.convert_train_data(str_items, symmetry)
                        if success:
                            # Send it down the pipe.
                            writer.send_bytes(data)

    def parse_chunk(self):
        while True:
            for r in self.readers:
                yield r.recv_bytes();

def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")


#
# Tests to check that records can round-trip successfully
def generate_fake_pos():
    """
        Generate a random game position.
        Result is ([[361] * 18], [362], [1])
    """
    # 1. 18 binary planes of length 361
    planes = [np.random.randint(2, size=361).tolist() for plane in range(16)]
    stm = float(np.random.randint(2))
    planes.append([stm] * 361)
    planes.append([1. - stm] * 361)
    # 2. 362 probs
    probs = np.random.randint(3, size=362).tolist()
    # 3. And a winner: 1 or -1
    winner = [ 2 * float(np.random.randint(2)) - 1 ]
    return (planes, probs, winner)

def run_test(parser):
    """
        Test game position decoding.
    """

    # First, build a random game position.
    planes, probs, winner = generate_fake_pos()

    # Convert that to a text record in the same format
    # generated by dump_supervised
    items = []
    for p in range(16):
        # generate first 360 bits
        h = np.packbits([int(x) for x in planes[p][0:360]]).tobytes().hex()
        # then add the stray single bit
        h += str(planes[p][360]) + "\n"
        items.append(h)
    # then who to move
    items.append(str(int(planes[17][0])) + "\n")
    # then probs
    items.append(' '.join([str(x) for x in probs]) + "\n")
    # and finally a winner
    items.append(str(int(winner[0])) + "\n")

    # Have an input string. Running it through parsing to see
    # if it gives the same result we started with.
    # We need a tf.Session() as we're going to use the tensorflow
    # decoding framework for part of the parsing.
    with tf.Session() as sess:
        # We apply and check every symmetry.
        for symmetry in range(8):
            result = parser.convert_train_data(items, symmetry)
            assert result[0] == True
            # We got back a serialized tf.train.Example, which we need to decode.
            graph = _parse_function(result[1])
            data = sess.run(graph)
            data = (data[0].tolist(), data[1].tolist(), data[2].tolist())

            # Apply the symmetry to the original
            sym_planes = [ [ plane[remap_vertex(vertex, symmetry)] for vertex in range(361) ] for plane in planes ]
            sym_probs = [ probs[remap_vertex(vertex, symmetry)] for vertex in range(361)] + [probs[361]]

            # Check that what we got out matches what we put in.
            assert data == (sym_planes, sym_probs, winner)
    print("Test parse passes")


# Convert a tf.train.Example protobuf into a tuple of tensors
# NB: This conversion is done in the tensorflow graph, NOT in python.
def _parse_function(example_proto):
    features = {"planes": tf.FixedLenFeature((1), tf.string),
                "probs": tf.FixedLenFeature((19*19+1), tf.float32),
                "winner": tf.FixedLenFeature((1), tf.float32)}
    parsed_features = tf.parse_single_example(example_proto, features)
    # We receives the planes as a byte array, but we really want
    # floats of shape (18, 19*19), so decode, cast, and reshape.
    planes = tf.decode_raw(parsed_features["planes"], tf.uint8)
    planes = tf.to_float(planes)
    planes = tf.reshape(planes, (18, 19*19))
    # the other features are already in the correct shape as return as-is.
    return planes, parsed_features["probs"], parsed_features["winner"]

def benchmark(parser):
    gen = parser.parse_chunk()
    while True:
        start = time.time()
        for _ in range(10000):
            next(gen)
        end = time.time()
        print("{} pos/sec {} secs".format( 10000. / (end - start), (end - start)))

def main(args):

    train_data_prefix = args.pop(0)

    chunks = get_chunks(train_data_prefix)
    print("Found {0} chunks".format(len(chunks)))

    if not chunks:
        return

    parser = ChunkParser(chunks)

    run_test(parser)
    #benchmark(parser)

    dataset = tf.data.Dataset.from_generator(
        parser.parse_chunk, output_types=(tf.string))
    dataset = dataset.shuffle(65536)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(16)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    tfprocess = TFProcess(next_batch)
    if args:
        restore_file = args.pop(0)
        tfprocess.restore(restore_file)
    while True:
        tfprocess.process(BATCH_SIZE)

if __name__ == "__main__":
    main(sys.argv[1:])
    mp.freeze_support()
