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

import sys
import argparse
import gzip
import math
import fileinput


def findEmptyBoard(tfh, hist):
    first_move_cnt = 0
    zeros = "0"*math.ceil(19*19/4) + "\n"
    while (1):
        #print(tfh.filelineno())
        empty_board = True
        board = []
        for _ in range(16):
            line = tfh.readline()
            if not line: break
            line = line.decode("utf-8")               # Board input planes
            empty_board = empty_board and line == zeros
            board.append(line)
            #print(empty_board, line)
        if not line: break
        to_move = int(tfh.readline().decode("utf-8"))           # 0 = black, 1 = white
        policy_weights = tfh.readline().decode("utf-8")         # 361 moves + 1 pass
        side_to_move_won = int(tfh.readline().decode("utf-8"))  # 1 = side to move won, -1 = lost
        # Note: it can be empty_board and white to move if black passed
        if empty_board and to_move == 0:
            first_move_cnt += 1
            policy_weights = policy_weights.split()
            cnt = policy_weights.count("0")
            if not cnt in hist: hist[cnt] = 0
            hist[cnt] += 1
    print(hist)
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
    hist = {}
    for filename in args.files:
        tfh = fileinput.input(filename, openhook=fileinput.hook_compressed)
        print(filename, " ", end="")
        first_move_cnt = findEmptyBoard(tfh, hist)
        total_first_move_cnt += first_move_cnt
    print("total first move cnt = ", total_first_move_cnt)


