#!/usr/bin/env python3

import sys
import glob
import gzip
import random
from tfprocess import TFProcess

TRAINFILE_PREFIX = "train.out"

# 16 planes, 1 stm, 1 x 362 probs, 1 winner = 19 lines
DATA_ITEM_LINES = 16 + 1 + 1 + 1

BATCH_SIZE = 128

def get_batch(chunks, batch_size):
    """
        Get one minibatch worth of data.

        Uncompresses a chunk and randomly selects up to batch_size items
        from it.
    """
    chunk = random.choice(chunks)
    batch = []
    with gzip.open(chunk, 'r') as chunk_file:
        file_content = chunk_file.readlines()
        item_count = len(file_content) // DATA_ITEM_LINES
        print("Found {0} items".format(item_count))
        picks = random.sample(range(item_count), batch_size)
        for pick in picks:
            pick_offset = pick * DATA_ITEM_LINES
            item = file_content[pick_offset:pick_offset + DATA_ITEM_LINES]
            str_items = [str(line, 'ascii') for line in item]
            batch.append(str_items)
    return batch

def convert_train_data(text_item):
    """"
        Convert textual training data to python lists.

        Converts a set of 19 lines of text into a pythonic dataformat.
        [[[plane_1],[plane_2],...], [probabilities], winner]
    """
    data = []
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
        planes.append([1.0 for _ in range(0, 361)])
        planes.append([0.0 for _ in range(0, 361)])
    else:
        planes.append([0.0 for _ in range(0, 361)])
        planes.append([1.0 for _ in range(0, 361)])
    assert len(planes) == 18
    data.append(planes)
    probabilities = [float(val) for val in text_item[17].split()]
    data.append(probabilities)
    winner = float(text_item[18])
    assert winner == 1.0 or winner == -1.0
    data.append(winner)
    return data

def do_train_loop(chunks, tfprocess):
    while True:
        batch = get_batch(chunks, BATCH_SIZE)
        data = [convert_train_data(b) for b in batch]
        tfprocess.process(data)

def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")

def main(args):
    train_data_prefix = args.pop(0)

    chunks = get_chunks(train_data_prefix)
    print("Found {0} chunks".format(len(chunks)))

    if chunks:
        tfprocess = TFProcess()
        do_train_loop(chunks, tfprocess)

if __name__ == "__main__":
    main(sys.argv[1:])