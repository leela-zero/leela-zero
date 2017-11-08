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
from tfprocess import TFProcess

TRAINFILE_PREFIX = "train.out"

# 16 planes, 1 stm, 1 x 362 probs, 1 winner = 19 lines
DATA_ITEM_LINES = 16 + 1 + 1 + 1

BATCH_SIZE = 256
BATCH_CACHE = []

def get_batch(chunks, batch_size):
    """
        Get one minibatch worth of data.

        Uncompresses a chunk and randomly selects up to batch_size items
        from it.
    """
    if not BATCH_CACHE:
        # fill cache
        chunk = random.choice(chunks)
        with gzip.open(chunk, 'r') as chunk_file:
            file_content = chunk_file.readlines()
            item_count = len(file_content) // DATA_ITEM_LINES
            # print("Found {0} items".format(item_count))
            picks = list(range(item_count))
            random.shuffle(picks)
            batch = []
            for pick in picks:
                pick_offset = pick * DATA_ITEM_LINES
                item = file_content[pick_offset:pick_offset + DATA_ITEM_LINES]
                str_items = [str(line, 'ascii') for line in item]
                batch.append(str_items)
                if len(batch) == batch_size:
                    BATCH_CACHE.append(batch)
                    batch = []
    return BATCH_CACHE.pop()

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
    elif symmetry == 2 or symmetry == 3:
        y = 19 - y - 1
    return y * 19 + x

def apply_symmetry(plane, symmetry):
    """
        Applies one of 8 symmetries to the go board.

        The supplied go board can have 361 or 362 elements. The 362th
        element is pass will which get the identity mapping.
    """
    assert symmetry >= 0 and symmetry < 8
    work_plane = [0.0] * 361
    for vertex in range(0, 361):
        work_plane[vertex] = plane[remap_vertex(vertex, symmetry)]
    # Map back "pass"
    if len(plane) == 362:
        work_plane.append(plane[361])
    return work_plane

def convert_train_data(text_item):
    """"
        Convert textual training data to python lists.

        Converts a set of 19 lines of text into a pythonic dataformat.
        [[plane_1],[plane_2],...],...
        [probabilities],...
        winner,...
    """
    planes = []
    for plane in range(0, 16):
        # 360 first bits are 90 hex chars
        hex_string = text_item[plane][0:90]
        integer = int(hex_string, 16)
        as_str = format(integer, '0>360b')
        # remaining bit that didn't fit
        last_digit = text_item[plane][90]
        assert last_digit == "0" or last_digit == "1"
        as_str += last_digit
        assert len(as_str) == 361
        plane = [0.0 if digit == "0" else 1.0 for digit in as_str]
        planes.append(plane)
    stm = text_item[16][0]
    assert stm == "0" or stm == "1"
    if stm == "0":
        planes.append([1.0] * 361)
        planes.append([0.0] * 361)
    else:
        planes.append([0.0] * 361)
        planes.append([1.0] * 361)
    assert len(planes) == 18
    probabilities = [float(val) for val in text_item[17].split()]
    assert len(probabilities) == 362
    winner = float(text_item[18])
    assert winner == 1.0 or winner == -1.0
    # Get one of 8 symmetries
    symmetry = random.randrange(8)
    sym_planes = [apply_symmetry(plane, symmetry) for plane in planes]
    sym_probabilities = apply_symmetry(probabilities, symmetry)
    return (sym_planes, sym_probabilities, [winner])

def do_train_loop(chunks, tfprocess):
    while True:
        batch = get_batch(chunks, BATCH_SIZE)
        batch_data = [[], [], []]
        for item in batch:
            planes, probs, winner = convert_train_data(item)
            batch_data[0].append(planes)
            batch_data[1].append(probs)
            batch_data[2].append(winner)
        tfprocess.process(batch_data)

def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")

def main(args):
    train_data_prefix = args.pop(0)

    chunks = get_chunks(train_data_prefix)
    print("Found {0} chunks".format(len(chunks)))

    if chunks:
        tfprocess = TFProcess()
        if args:
            restore_file = args.pop(0)
            tfprocess.restore(restore_file)
        do_train_loop(chunks, tfprocess)

if __name__ == "__main__":
    main(sys.argv[1:])
