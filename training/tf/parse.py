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
import binascii
import glob
import gzip
import itertools
import math
import multiprocessing as mp
import numpy as np
import queue
import random
import struct
import sys
import tensorflow as tf
import time
import threading

# 16 planes, 1 side to move, 1 x 362 probs, 1 winner = 19 lines
DATA_ITEM_LINES = 16 + 1 + 1 + 1

# Sane values are from 4096 to 64 or so. The maximum depends on the amount
# of RAM in your GPU and the network size. You need to adjust the learning rate
# if you change this.
BATCH_SIZE = 512

# Use a random sample of 1/16th of the input data read. This helps
# improve the spread of games in the shuffle buffer.
DOWN_SAMPLE = 16

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
    def __init__(self, chunkdatagen, shuffle_size=1, sample=1, buffer_size=1, workers=None):
        """
            Read data and yield batches of raw tensors.

            'chunkdatagen' is a generator yeilding chunkdata
            'shuffle_size' is the size of the shuffle buffer.
            'sample' is the rate to down-sample.
            'workers' is the number of child workers to use.

            The data is represented in a number of formats through this dataflow
            pipeline. In order, they are:

            chunk: The name of a file containing chunkdata

            chunkdata: type Bytes. Either mutiple records of v1 format, or multiple records
            of v2 format.

            v1: The original text format describing a move. 19 lines long. VERY slow
            to decode. Typically around 2500 bytes long. Used only for backward
            compatability.

            v2: Packed binary representation of v1. Fixed length, no record seperator.
            The most compact format. Data in the shuffle buffer is held in this
            format as it allows the largest possible shuffle buffer. Very fast to
            decode. Preferred format to use on disk. 2176 bytes long.

            raw: A byte string holding raw tensors contenated together. This is used
            to pass data from the workers to the parent. Exists because TensorFlow doesn't
            have a fast way to unpack bit vectors. 7950 bytes long.
        """
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

        # set the down-sampling rate
        self.sample = sample

        # V2 Format
        # int32 version (4 bytes)
        # (19*19+1) float32 probabilities (1448 bytes)
        # 19*19*16 packed bit planes (722 bytes)
        # uint8 side_to_move (1 byte)
        # uint8 is_winner (1 byte)
        self.v2_struct = struct.Struct('4s1448s722sBB')

        # Struct used to return data from child workers.
        # float32 winner
        # float32*392 probs
        # uint*6498 planes
        # (order is to ensure that no padding is required to make float32 be 32-bit aligned)
        self.raw_struct = struct.Struct('4s1448s6498s')

        # Start worker processes, leave 2 for TensorFlow
        if workers is None:
            workers = max(1, mp.cpu_count() - 2)
        print("Using {} worker processes.".format(workers))

        # Spread shuffle buffer over workers.
        self.shuffle_size = int(shuffle_size / workers)

        # Start the child workers running
        self.readers = []
        for _ in range(workers):
            read, write = mp.Pipe(duplex=False)
            mp.Process(target=self.task,
                    args=(chunkdatagen, write)).start()
            self.readers.append(read)
            write.close()

    def convert_v1_to_v2(self, text_item):
        """
            Convert v1 text format to v2 packed binary format

            Converts a set of 19 lines of text into a byte string
            [[plane_1],[plane_2],...],...
            [probabilities],...
            winner,...
        """
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
        # and then to a byte string
        planes = np.packbits(planes).tobytes()

        # Get the 'side to move'
        stm = text_item[16][0]
        assert stm == "0" or stm == "1"
        stm = int(stm)

        # Load the probabilities.
        probabilities = np.array(text_item[17].split()).astype(np.float32)
        if np.any(np.isnan(probabilities)):
            # Work around a bug in leela-zero v0.3, skipping any
            # positions that have a NaN in the probabilities list.
            return False, None
        assert len(probabilities) == 362

        probs = probabilities.tobytes()
        assert(len(probs) == 362 * 4)

        # Load the game winner color.
        winner = float(text_item[18])
        assert winner == 1.0 or winner == -1.0
        winner = int((winner + 1) / 2)

        version = struct.pack('i', 1)

        return True, self.v2_struct.pack(version, probs, planes, stm, winner)
        #return True, b''.join([b'\1\0\0\0', probs, planes, b'\0\1'[stm:stm+1], b'\0\1'[winner:winner+1]])

    def convert_v2_to_raw(self, symmetry, content):
        """
            Convert v2 binary training data to packed tensors 

            v2 struct format is
                int32 ver
                float probs[19*18+1]
                byte planes[19*19*16/8]
                byte to_move
                byte winner

            packed tensor format is
                float32 winner
                float32*362 probs
                uint8*6498 planes
        """
        assert symmetry >= 0 and symmetry < 8

        (ver, probs, planes, to_move, winner) = self.v2_struct.unpack(content)

        planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8))
        # We use the full length reflection tables to apply symmetry
        # to all 16 planes simultaneously
        planes = planes[self.full_reflection_table[symmetry]]
        assert len(planes) == 19*19*16
        # Convert the array to a byte string
        planes = [ planes.tobytes() ]

        # Now we add the two final planes, being the 'color to move' planes.
        # These already a fully symmetric, so we add them directly as byte
        # strings of length 361.
        stm = to_move
        assert stm == 0 or stm == 1
        planes.append(self.flat_planes[1 - stm])
        planes.append(self.flat_planes[stm])

        # Flatten all planes to a single byte string
        planes = b''.join(planes)
        assert len(planes) == (18 * 19 * 19)

        probs = np.frombuffer(probs, dtype=np.float32)
        # Apply symmetries to the probabilities.
        probs = probs[self.prob_reflection_table[symmetry]]
        assert len(probs) == 362

        winner = float(winner * 2 - 1)
        assert winner == 1.0 or winner == -1.0

        winner = struct.pack('f', winner)

        return self.raw_struct.pack(winner, probs.tobytes(), planes)

    def convert_chunkdata_to_v2(self, chunkdata):
        """
            Take chunk of unknown format, and return it as a list of
            v2 format records.
        """
        if chunkdata[0:4] == b'\1\0\0\0':
            #print("V2 chunkdata")
            items = [ chunkdata[i:i+self.v2_struct.size]
                        for i in range(0, len(chunkdata), self.v2_struct.size) ]
            if self.sample > 1:
                # Downsample to 1/Nth of the items.
                items = random.sample(items, len(items) // self.sample)
            return items
        else:
            #print("V1 chunkdata")
            file_chunkdata = chunkdata.splitlines()

            result = []
            for i in range(0, len(file_chunkdata), DATA_ITEM_LINES):
                if self.sample > 1:
                    # Downsample, using only 1/Nth of the items.
                    if random.randint(0, self.sample-1) != 0:
                        continue  # Skip this record.
                item = file_chunkdata[i:i+DATA_ITEM_LINES]
                str_items = [str(line, 'ascii') for line in item]
                success, data = self.convert_v1_to_v2(str_items)
                if success:
                    result.append(data)
            return result


    def task(self, chunkdatagen, writer):
        """
            Run in fork'ed process, read data off disk, parsing, shuffling and
            sending raw data through pipe back to main process.
        """
        moves = []
        for chunkdata in chunkdatagen:
            items = self.convert_chunkdata_to_v2(chunkdata)
            moves.extend(items)
            if len(moves) <= self.shuffle_size:
                continue
            # randomize the order of the loaded moves.
            random.shuffle(moves)
            while len(moves) > self.shuffle_size:
                move = moves.pop()
                # Pick a random symmetry to apply
                symmetry = random.randrange(8)
                data = self.convert_v2_to_raw(symmetry, move)
                writer.send_bytes(data)

    def convert_raw_to_tuple(self, data):
        """
            Convert packed tensors to tuple of raw tensors
        """
        (winner, probs, planes) = self.raw_struct.unpack(data)
        return (planes, probs, winner)

    def chunk_gen(self):
        """
            Read packed tensors from child workers and yield
            tuples of raw tensors
        """
        while True:
            for r in mp.connection.wait(self.readers):
                try:
                    s = r.recv_bytes()
                except EOFError:
                    print("Reader EOF")
                    self.readers.remove(r)
                yield self.convert_raw_to_tuple(s)

    def batch_gen(self, gen):
        """
            Pack multiple records into a single batch
        """
        # Get N records. We flatten the returned generator to
        # a list because we need to reuse it.
        while True:
            s = list(itertools.islice(gen, BATCH_SIZE))
            if not len(s):
                return
            yield ( b''.join([x[0] for x in s]),
                    b''.join([x[1] for x in s]),
                    b''.join([x[2] for x in s]) )

    def parse(self):
        """
            Read data from child workers and yield batches
            of raw tensors
        """
        for b in self.batch_gen(self.chunk_gen()):
            yield b

def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")

def build_chunkgen(chunks):
    """
        generator yeilding chunkdata from chunk files.
    """
    yield b''  # To ensure that the shuffle happens in child workers.
    while True:
        random.shuffle(chunks)
        for filename in chunks:
            with gzip.open(filename, 'rb') as chunk_file:
                yield chunk_file.read()

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

def run_test():
    """
        Test game position decoding pipeline.

        We generate a V1 record, and feed it all the way
        through the parsing pipeline to final tensors,
        checking that what we get out is what we put in.
    """
    # First, build a random game position.
    planes, probs, winner = generate_fake_pos()

    # Convert that to a v1 text record.
    items = []
    for p in range(16):
        # generate first 360 bits
        h = np.packbits([int(x) for x in planes[p][0:360]]).tobytes().hex()
        # then add the stray single bit
        h += str(planes[p][360]) + "\n"
        items.append(h)
    # then side to move
    items.append(str(int(planes[17][0])) + "\n")
    # then probabilities
    items.append(' '.join([str(x) for x in probs]) + "\n")
    # and finally if the side to move is a winner
    items.append(str(int(winner[0])) + "\n")

    # Convert to a chunkdata byte string.
    chunkdata = ''.join(items).encode('ascii')

    # feed BATCH_SIZE copies into parser
    parser = ChunkParser(
            (chunkdata for _ in range(BATCH_SIZE)),
            shuffle_size=1, workers=1)

    # Get one batch from the parser.
    batchgen = parser.parse()
    batch = next(batchgen)

    # We need a tf.Session() as we're going to use the tensorflow
    # decoding framework to decode the raw byte strings into tensors.
    with tf.Session() as sess:
        graph = _parse_function(*batch)
        data = sess.run(graph)

    # Convert tensors to python lists.
    batch = (data[0].tolist(), data[1].tolist(), data[2].tolist())

    # Check that every record in the batch is a some valid symmetry
    # of the original data.
    for i in range(BATCH_SIZE):
        data = (batch[0][i], batch[1][i], batch[2][i])

        # We have an unknown symmetry, so search for a matching one.
        result = False
        for symmetry in range(8):
            # Apply the symmetry to the original
            sym_planes = [ [ plane[remap_vertex(vertex, symmetry)] for vertex in range(361) ] for plane in planes ]
            sym_probs = [ probs[remap_vertex(vertex, symmetry)] for vertex in range(361)] + [probs[361]]

            # Check that what we got out matches what we put in.
            if data == (sym_planes, sym_probs, winner):
                result = True
                break
        # Check that there is at least one matching symmetry.
        assert(result == True)
    print("Test parse passes")

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
                feed_dict={t.training: True, t.handle: t.train_handle, t.learning_rate: 0.05})

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

    #run_test()

    train_parser = ChunkParser(build_chunkgen(training),
            shuffle_size=1<<19, sample=DOWN_SAMPLE)
    dataset = tf.data.Dataset.from_generator(
        train_parser.parse, output_types=(tf.string, tf.string, tf.string))
    dataset = dataset.map(_parse_function)
    dataset = dataset.prefetch(4)
    train_iterator = dataset.make_one_shot_iterator()

    test_parser = ChunkParser(build_chunkgen(test))
    dataset = tf.data.Dataset.from_generator(
        test_parser.parse, output_types=(tf.string, tf.string, tf.string))
    dataset = dataset.map(_parse_function)
    dataset = dataset.prefetch(4)
    test_iterator = dataset.make_one_shot_iterator()

    #benchmark(train_parser)

    tfprocess = TFProcess()
    tfprocess.init(dataset, train_iterator, test_iterator)

    #benchmark1(tfprocess)

    if args:
        restore_file = args.pop(0)
        tfprocess.restore(restore_file)
    while True:
        tfprocess.process(BATCH_SIZE)

if __name__ == "__main__":
    main(sys.argv[1:])
    mp.freeze_support()
