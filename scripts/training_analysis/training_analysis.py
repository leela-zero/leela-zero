#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017 Andy Olsen
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
import gzip
import math
import fileinput
import collections
import os

BOARDSIZE = 19
NIBBLE = 4
HISTORY_PLANES = 8

def hook_compressed_encoded(encoding):
    def hook_compressed(filename, mode):
        ext = os.path.splitext(filename)[1]
        if ext == '.gz':
            #return gzip.open(filename, mode, encoding=encoding) # this doesn't work, why?
            return gzip.open(filename, "rt", encoding=encoding)
        else:
            return open(filename, mode, encoding=encoding)
    return hook_compressed

def findEmptyBoard(tfh, hist):
    first_move_cnt = 0
    zeros = "0"*math.ceil((BOARDSIZE**2+1)/NIBBLE) + "\n" # 362 bits in hex nibbles
    while (1):
        empty_board = True
        for _ in range(HISTORY_PLANES*2):
            line = tfh.readline()
            if not line: break
            empty_board = empty_board and line == zeros
        if not line: break
        to_move = int(tfh.readline())           # 0 = black, 1 = white
        policy_weights = tfh.readline()         # 361 moves + 1 pass
        side_to_move_won = int(tfh.readline())  # 1 = side to move won, -1 = lost
        # Note: it can be empty_board and white to move if black passed
        if empty_board and to_move == 0:
            first_move_cnt += 1
            policy_weights = policy_weights.split()
            cnt = policy_weights.count("0")
            hist[cnt] += 1
    print(dict(hist))
    return first_move_cnt

if __name__ == "__main__":
    usage_str = """
This script analyzes the training data for abnormal results,
such as issues #359/#375. It looks for the first move of a game
and does a histogram of how many policy training targets have 
a value of "0". Normally there should be zero but with the bug
there can be many. Note the bug is intermittent so there will
be more games with bad data in them that this script can detect.

Example output:
    ../../../train_2184b750/train_2184b750_0.gz  {0: 32}
        <snip>
    ../../../train_2184b750/train_2184b750_9.gz  {0: 10970, 38: 1, 10: 1, 348: 1, 345: 1, 139: 1, 259: 1, 335: 1, 97: 1, 23: 1, 276: 1}
    total first move cnt =  10980

Out of 10980 games, 10970 games had no 0 in the training target.
1 game had 38 0s
1 game had 345 0s
etc.
"""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=usage_str)
    parser.add_argument("files", metavar="files", type=str, nargs="+", help="training files (with or without *.gz)")
    args = parser.parse_args()
    total_first_move_cnt = 0
    hist = collections.defaultdict(int)
    for filename in args.files:
        tfh = fileinput.FileInput(filename, openhook=hook_compressed_encoded("utf-8"))
        print(filename, " ", end="")
        total_first_move_cnt += findEmptyBoard(tfh, hist)
    print("total first move cnt = ", total_first_move_cnt)


