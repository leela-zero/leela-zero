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
import tensorflow as tf
from tfprocess import TFProcess

# 16 planes, 1 stm, 1 x 362 probs, 1 winner = 19 lines
DATA_ITEM_LINES = 16 + 1 + 1 + 1

BATCH_SIZE = 256
reflection_table = []

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
        # Build reflection tables.
        self.reflection_table = [[remap_vertex(vertex, sym) for vertex in range(361)] for sym in range(8)]
        self.queue = mp.Queue(4096)
        # Start worker processes, leave 1 for TensorFlow
        workers = max(1, mp.cpu_count() - 1)
        print("Using {} worker processes.".format(workers))
        for _ in range(workers):
            mp.Process(target=self.task,
                       args=(chunks, self.queue)).start()

    def apply_symmetry(self, plane, symmetry):
        """
            Applies one of 8 symmetries to the go board. Assumes 'plane'
            is an np.array type.

            The supplied go board can have 361 or 362 elements. The 362th
            element is pass will which get the identity mapping.
        """
        assert symmetry >= 0 and symmetry < 8
        work_plane = plane[self.reflection_table[symmetry]]
        # Map back "pass"
        if len(plane) == 362:
            work_plane = np.append(work_plane, plane[361])
        return work_plane.tolist()

    def convert_train_data(self, text_item):
        """
            Convert textual training data to python lists.

            Converts a set of 19 lines of text into a pythonic dataformat.
            [[plane_1],[plane_2],...],...
            [probabilities],...
            winner,...
        """
        # Pick a random symmetry to apply
        symmetry = random.randrange(8)
        planes = []
        for plane in range(0, 16):
            # first 360 first bits are 90 hex chars, encoded MSB
            hex_string = text_item[plane][0:90]
            array = np.unpackbits(np.frombuffer(bytearray.fromhex(hex_string), dtype=np.uint8))
            array = array.astype(float)
            # Remaining bit that didn't fit. Encoded LSB so
            # it needs to be specially handled.
            last_digit = text_item[plane][90]
            assert last_digit == "0" or last_digit == "1"
            array = np.append(array, float(last_digit))
            assert len(array) == 361
            # Apply symmetry and append
            planes.append(self.apply_symmetry(array, symmetry))
        stm = float(text_item[16][0])
        assert stm == 0. or stm == 1.
        planes.append([1.0 - stm] * 361)
        planes.append([stm] * 361)
        assert len(planes) == 18

        probabilities = np.array(text_item[17].split()).astype(float)
        if np.any(np.isnan(probabilities)):
            # Work around a bug in leela-zero v0.3
            return False, None
        assert len(probabilities) == 362
        winner = float(text_item[18])
        assert winner == 1.0 or winner == -1.0
        # Apply symmetry
        probabilities = self.apply_symmetry(probabilities, symmetry)

        return True, (planes, probabilities, [winner])

    def task(self, chunks, queue):
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
                        success, data = self.convert_train_data(str_items)
                        if success:
                            queue.put(data)

    def parse_chunk(self):
        while True:
            yield self.queue.get()

def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")

def main(args):
    train_data_prefix = args.pop(0)

    chunks = get_chunks(train_data_prefix)
    print("Found {0} chunks".format(len(chunks)))

    if not chunks:
        return

    parser = ChunkParser(chunks)

    dataset = tf.data.Dataset.from_generator(
        parser.parse_chunk, output_types=(tf.float32, tf.float32, tf.float32))
    dataset = dataset.shuffle(65536)
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
