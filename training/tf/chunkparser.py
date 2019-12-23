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

import binascii
import glob
import gzip
import itertools
import math
import multiprocessing as mp
import numpy as np
import queue
import random
import shufflebuffer as sb
import struct
import sys
import threading
import time
import unittest

# 16 planes, 1 side to move (komi), 1 x 362 probs, 1 winner = 19 lines
DATA_ITEM_LINES = 16 + 1 + 1 + 1

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

# Interface for a chunk data source.
class ChunkDataSrc:
    def __init__(self, items):
        self.items = items
    def next(self):
        if not self.items:
            return None
        return self.items.pop()

class ChunkParser:
    def __init__(self, chunkdatasrc, shuffle_size=1, sample=1,
                 buffer_size=1, batch_size=256, workers=None):
        """
            Read data and yield batches of raw tensors.

            'chunkdatasrc' is an object yeilding chunkdata
            'shuffle_size' is the size of the shuffle buffer.
            'sample' is the rate to down-sample.
            'workers' is the number of child workers to use.

            The data is represented in a number of formats through this dataflow
            pipeline. In order, they are:

            chunk: The name of a file containing chunkdata

            chunkdata: type Bytes. Either mutiple records of v1 format,
            or multiple records of v2 format.

            v1: The original text format describing a move. 19 lines long.
            VERY slow to decode. Typically around 2500 bytes long.
            Used only for backward compatability.

            v2: Packed binary representation of v1. Fixed length,
            no record seperator. The most compact format.
            Data in the shuffle buffer is held in this
            format as it allows the largest possible shuffle buffer.
            Very fast to decode. Preferred format to use on disk.
            2176 bytes long.

            raw: A byte string holding raw tensors contenated together.
            This is used to pass data from the workers to the parent.
            Exists because TensorFlow doesn't have a fast way to
            unpack bit vectors.
            7950 bytes long.
        """
        # Build probility reflection tables.
        # The last element is 'pass' and is identity mapped.
        self.prob_reflection_table = [
            [remap_vertex(vertex, sym)
              for vertex in range(361)]+[361] for sym in range(8)]
        # Build full 16-plane reflection tables.
        self.full_reflection_table = [
            np.array([remap_vertex(vertex, sym) + p * 361
                for p in range(16) for vertex in range(361)])
                    for sym in range(8)]
        # Convert both to np.array.
        # This avoids a conversion step when they're actually used.
        self.prob_reflection_table = [
            np.array(x, dtype=np.int64) for x in self.prob_reflection_table ]
        self.full_reflection_table = [
            np.array(x, dtype=np.int64) for x in self.full_reflection_table ]

        # set the down-sampling rate
        self.sample = sample
        # set the mini-batch size
        self.batch_size = batch_size
        # set number of elements in the shuffle buffer.
        self.shuffle_size = shuffle_size
        # Start worker processes, leave 2 for TensorFlow
        if workers is None:
            workers = max(1, mp.cpu_count() - 2)
        print("Using {} worker processes.".format(workers))

        # Start the child workers running
        self.readers = []
        for _ in range(workers):
            read, write = mp.Pipe(duplex=False)
            mp.Process(target=self.task,
                       args=(chunkdatasrc, write),
                       daemon=True).start()
            self.readers.append(read)
            write.close()
        self.init_structs()

    def init_structs(self):
        # struct.Struct doesn't pickle, so it needs to be separately
        # constructed in workers.

        # V2 Format
        # int32 version (4 bytes)
        # (19*19+1) float32 probabilities (1448 bytes)
        # 19*19*16 packed bit planes (722 bytes)
        # float32 side_to_move (4 bytes)
        # uint8 is_winner (1 byte)
        self.v2_struct = struct.Struct('4s1448s722sfB')

        # Struct used to return data from child workers.
        # float32 winner
        # float32*392 probs
        # uint*6498 planes
        # (order is to ensure that no padding is required to
        #  make float32 be 32-bit aligned)
        self.raw_struct = struct.Struct('4s1448s6498s')

    def convert_v1_to_v2(self, text_item):
        """
            Convert v1 text format to v2 packed binary format

            Converts a set of 19 lines of text into a byte string
            [[plane_1],[plane_2],...],...
            [probabilities],...
            winner,...
        """
        # We start by building a list of 16 planes,
        # each being a 19*19 == 361 element array
        # of type np.uint8
        planes = []
        for plane in range(0, 16):
            # first 360 first bits are 90 hex chars, encoded MSB
            hex_string = text_item[plane][0:90]
            try:
                array = np.unpackbits(np.frombuffer(
                    bytearray.fromhex(hex_string), dtype=np.uint8))
            except:
                return False, None
            # Remaining bit that didn't fit. Encoded LSB so
            # it needs to be specially handled.
            last_digit = text_item[plane][90]
            if not (last_digit == "0" or last_digit == "1"):
                return False, None
            # Apply symmetry and append
            planes.append(array)
            planes.append(np.array([last_digit], dtype=np.uint8))

        # We flatten to a single array of len 16*19*19, type=np.uint8
        planes = np.concatenate(planes)
        # and then to a byte string
        planes = np.packbits(planes).tobytes()

        # Get the 'side to move' (komi)
        stm = float(text_item[16])

        # Load the probabilities.
        probabilities = np.array(text_item[17].split()).astype(np.float32)
        if np.any(np.isnan(probabilities)):
            # Work around a bug in leela-zero v0.3, skipping any
            # positions that have a NaN in the probabilities list.
            return False, None
        if not(len(probabilities) == 362):
            return False, None

        probs = probabilities.tobytes()
        if not(len(probs) == 362 * 4):
            return False, None

        # Load the game winner color.
        winner = float(text_item[18])
        if not(winner == 1.0 or winner == -1.0):
            return False, None
        winner = int((winner + 1) / 2)

        version = struct.pack('i', 1)

        return True, self.v2_struct.pack(version, probs, planes, stm, winner)

    def v2_apply_symmetry(self, symmetry, content):
        """
            Apply a random symmetry to a v2 record.
        """
        assert symmetry >= 0 and symmetry < 8

        # unpack the record.
        (ver, probs, planes, to_move, winner) = self.v2_struct.unpack(content)

        planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8))
        # We use the full length reflection tables to apply symmetry
        # to all 16 planes simultaneously
        planes = planes[self.full_reflection_table[symmetry]]
        assert len(planes) == 19*19*16
        planes = np.packbits(planes)
        planes = planes.tobytes()

        probs = np.frombuffer(probs, dtype=np.float32)
        # Apply symmetries to the probabilities.
        probs = probs[self.prob_reflection_table[symmetry]]
        assert len(probs) == 362
        probs = probs.tobytes()

        # repack record.
        return self.v2_struct.pack(ver, probs, planes, to_move, winner)


    def convert_v2_to_tuple(self, content):
        """
            Convert v2 binary training data to packed tensors

            v2 struct format is
                int32 ver
                float probs[19*18+1]
                byte planes[19*19*16/8]
                float to_move
                byte winner

            packed tensor formats are
                float32 winner
                float32*362 probs
                uint8*6498 planes
        """
        (ver, probs, planes, to_move, winner) = self.v2_struct.unpack(content)
        # Unpack planes.
        planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype('f')
        assert len(planes) == 19*19*16
        # Now we add the two final planes, being the 'color to move' (komi) planes.
        stm = to_move
        # Flattern all planes to a single byte string
        planes = planes.tobytes() + (np.array([1.0-stm]*361 + [stm]*361)).astype('f').tobytes()
        assert len(planes) == (18 * 19 * 19 * 4), len(planes)

        winner = float(winner * 2 - 1)
        assert winner == 1.0 or winner == -1.0, winner
        winner = struct.pack('f', winner)

        return (planes, probs, winner)

    def convert_chunkdata_to_v2(self, chunkdata):
        """
            Take chunk of unknown format, and return it as a list of
            v2 format records.
        """
        if chunkdata[0:4] == b'\1\0\0\0':
            #print("V2 chunkdata")
            for i in range(0, len(chunkdata), self.v2_struct.size):
                if self.sample > 1:
                    # Downsample, using only 1/Nth of the items.
                    if random.randint(0, self.sample-1) != 0:
                        continue  # Skip this record.
                yield chunkdata[i:i+self.v2_struct.size]
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
                    yield data

    def task(self, chunkdatasrc, writer):
        """
            Run in fork'ed process, read data from chunkdatasrc,
            parsing, shuffling and sending v2 data through pipe back
            to main process.
        """
        self.init_structs()
        while True:
            chunkdata = chunkdatasrc.next()
            if chunkdata is None:
                break
            for item in self.convert_chunkdata_to_v2(chunkdata):
                # Apply a random symmetry
                symmetry = random.randrange(8)
                item = self.v2_apply_symmetry(symmetry, item)
                writer.send_bytes(item)

    def v2_gen(self):
        """
            Read v2 records from child workers, shuffle, and yield
            records.
        """
        sbuff = sb.ShuffleBuffer(self.v2_struct.size, self.shuffle_size)
        while len(self.readers):
            #for r in mp.connection.wait(self.readers):
            for r in self.readers:
                try:
                    s = r.recv_bytes()
                    s = sbuff.insert_or_replace(s)
                    if s is None:
                        continue  # shuffle buffer not yet full
                    yield s
                except EOFError:
                    print("Reader EOF")
                    self.readers.remove(r)
        # drain the shuffle buffer.
        while True:
            s = sbuff.extract()
            if s is None:
                return
            yield s

    def tuple_gen(self, gen):
        """
            Take a generator producing v2 records and convert them to tuples.
            applying a random symmetry on the way.
        """
        for r in gen:
            yield self.convert_v2_to_tuple(r)

    def batch_gen(self, gen):
        """
            Pack multiple records into a single batch
        """
        # Get N records. We flatten the returned generator to
        # a list because we need to reuse it.
        while True:
            s = list(itertools.islice(gen, self.batch_size))
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
        gen = self.v2_gen()        # read from workers
        gen = self.tuple_gen(gen)  # convert v2->tuple
        gen = self.batch_gen(gen)  # assemble into batches
        for b in gen:
            yield b


# Tests to check that records can round-trip successfully
class ChunkParserTest(unittest.TestCase):
    def generate_fake_pos(self):
        """
            Generate a random game position.
            Result is ([[361] * 18], [362], [1])
        """
        # 1. 18 binary planes of length 361
        planes = [np.random.randint(2, size=361).tolist()
                  for plane in range(16)]
        stm = float(np.random.randint(2))
        planes.append([stm] * 361)
        planes.append([1. - stm] * 361)
        # 2. 362 probs
        probs = np.random.randint(3, size=362).tolist()
        # 3. And a winner: 1 or -1
        winner = [ 2 * float(np.random.randint(2)) - 1 ]
        return (planes, probs, winner)

    def test_parsing(self):
        """
            Test game position decoding pipeline.

            We generate a V1 record, and feed it all the way
            through the parsing pipeline to final tensors,
            checking that what we get out is what we put in.
        """
        batch_size=256
        # First, build a random game position.
        planes, probs, winner = self.generate_fake_pos()

        # Convert that to a v1 text record.
        items = []
        for p in range(16):
            # generate first 360 bits
            h = np.packbits([int(x) for x in planes[p][0:360]]).tobytes().hex()
            # then add the stray single bit
            h += str(planes[p][360]) + "\n"
            items.append(h)
        # then side to move/komi
        items.append(str(planes[17][0]) + "\n")
        # then probabilities
        items.append(' '.join([str(x) for x in probs]) + "\n")
        # and finally if the side to move is a winner
        items.append(str(int(winner[0])) + "\n")

        # Convert to a chunkdata byte string.
        chunkdata = ''.join(items).encode('ascii')

        # feed batch_size copies into parser
        chunkdatasrc = ChunkDataSrc([chunkdata for _ in range(batch_size*2)])
        parser = ChunkParser(chunkdatasrc,
                             shuffle_size=1, workers=1, batch_size=batch_size)

        # Get one batch from the parser.
        batchgen = parser.parse()
        data = next(batchgen)

        # Convert batch to python lists.
        batch = ( np.reshape(np.frombuffer(data[0], dtype=np.float32),
                             (batch_size, 18, 19*19)).tolist(),
                  np.reshape(np.frombuffer(data[1], dtype=np.float32),
                             (batch_size, 19*19+1)).tolist(),
                  np.reshape(np.frombuffer(data[2], dtype=np.float32),
                             (batch_size, 1)).tolist() )

        # Check that every record in the batch is a some valid symmetry
        # of the original data.
        for i in range(batch_size):
            data = (batch[0][i], batch[1][i], batch[2][i])

            # We have an unknown symmetry, so search for a matching one.
            result = False
            for symmetry in range(8):
                # Apply the symmetry to the original
                sym_planes = [
                    [plane[remap_vertex(vertex, symmetry)]
                        for vertex in range(361)]
                            for plane in planes]
                sym_probs = [
                    probs[remap_vertex(vertex, symmetry)]
                        for vertex in range(361)] + [probs[361]]

                if symmetry == 0:
                    assert sym_planes == planes
                    assert sym_probs == probs

                # Check that what we got out matches what we put in.
                if data == (sym_planes, sym_probs, winner):
                    result = True
                    break
            # Check that there is at least one matching symmetry.
            assert result == True
        print("Test parse passes")
        # drain parser
        for _ in batchgen:
            pass

if __name__ == '__main__':
    unittest.main()
