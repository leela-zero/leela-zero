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
import itertools

NIBBLE = 4
BOARDSIZE = 19
TURNSIZE = 19
HISTORY_PLANES = 8
TOMOVE = 16
POLICY = 17
WINNER = 18
ZERO_LINE = "0"*math.ceil((BOARDSIZE**2+1)/NIBBLE) + "\n" # 361+1 bits in hex nibbles

def findEmptyBoard(tfh):
    count = collections.Counter()
    while (1):
        empty_board = True
        turn = [line.decode("utf-8") for line in itertools.islice(tfh, TURNSIZE)]
        if len(turn) < TURNSIZE:
            break
        board = turn[:HISTORY_PLANES*2]
        to_move = int(turn[TOMOVE])              # 0 = black, 1 = white
        policy_weights = turn[POLICY].split()    # 361 moves + 1 pass
        side_to_move_won = int(turn[WINNER])     # 1 = side to move won, -1 = lost
        empty_board = all(line == ZERO_LINE for line in board)
        # Note: it can be empty_board and white to move if black passed
        if empty_board and to_move == 0:
            #pass_weight = float(policy_weights[-1])
            #print("pass weight %0.2f" % (pass_weight*1000))
            count[policy_weights.count("0")] += 1
    return count

def main():
    usage_str = """
Update: This method will not work on newer nets, becauase the
newer nets are starting to narrow the search. So even a normally
working system will have many zeros in the policy target for the
first move. This was originally developed against net 2184b750,
where this method did work.

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
    totalCount = collections.Counter()
    for filename in args.files:
        tfh = fileinput.FileInput(filename, openhook=fileinput.hook_compressed)
        fileCount = findEmptyBoard(tfh)
        print(filename, fileCount)
        totalCount.update(fileCount)
    print("first move counts =", totalCount)
    print("total first move counts =", sum(totalCount.values()))

if __name__ == "__main__":
    main()


