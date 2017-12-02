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

class Best:
    def __init__(self):
        self.value = None
        self.movenum = None
        self.bestmovevisits = None
    def update(self, value, movenum, bestmovevisits):
        if self.value == None or value < self.value:
            self.value = value
            self.movenum = movenum
            self.bestmovevisits = bestmovevisits
    def tostr(self):
        s = "%f %d %d" % (self.value, self.movenum, self.bestmovevisits)
        return s

class GameStats:
    def __init__(self, filename):
        self.filename = filename
        self.w_best_netwinrate = Best()
        self.w_best_uctwinrate = Best()
        self.w_best_bestmovevisits = Best()
        self.total_moves = None
        self.resign_movenum = None
        self.resign_type = None     # "Correct", "Incorrect"
        self.winner = None

def parseGames(filenames, resignrate, verbose):
    gsd = {}
    for filename in filenames:
        training_filename = filename.replace(".debug", "")
        gs = GameStats(filename)
        gsd[filename] = gs
        with open(filename) as fh, open(training_filename) as tfh:
            movenum = 0
            version = fh.readline()
            assert version == "1\n"
            while 1:
                movenum += 1
                for _ in range(16):
                    line = tfh.readline()               # Board input planes
                if not line: break
                to_move = int(tfh.readline())           # 0 = black, 1 = white
                policy_weights = tfh.readline()         # 361 moves + 1 pass
                side_to_move_won = int(tfh.readline())  # 1 = side to move won, -1 = lost
                if not gs.winner:
                    if side_to_move_won == 1: gs.winner = to_move
                    else : gs.winner = 1 - to_move
                (netwinrate, root_uctwinrate, child_uctwinrate, bestmovevisits) = fh.readline().split()
                netwinrate = float(netwinrate)
                root_uctwinrate = float(root_uctwinrate)
                child_uctwinrate = float(child_uctwinrate)
                bestmovevisits = int(bestmovevisits)
                if side_to_move_won == 1:
                    if verbose: print("+", to_move, movenum, netwinrate, child_uctwinrate, bestmovevisits)
                    gs.w_best_netwinrate.update(netwinrate, movenum, bestmovevisits)
                    gs.w_best_uctwinrate.update(child_uctwinrate, movenum, bestmovevisits)
                    gs.w_best_bestmovevisits.update(bestmovevisits, movenum, bestmovevisits)
                    if not gs.resign_type and child_uctwinrate < resignrate:
                        if verbose: print("Incorrect resign")
                        gs.resign_type = "Incorrect"
                        gs.resign_movenum = movenum
                else:
                    if verbose: print("-", to_move, movenum, netwinrate, child_uctwinrate, bestmovevisits)
                    if not gs.resign_type and child_uctwinrate < resignrate:
                        if verbose: print("Correct resign")
                        gs.resign_type = "Correct"
                        gs.resign_movenum = movenum
            gs.total_moves = movenum
        if verbose: print(filename, gs.w_best_netwinrate.tostr(), gs.w_best_uctwinrate.tostr(), gs.total_moves)
    return gsd

def resignStats(gsd, resignrate):
    print("Resign rate: %0.2f" % (resignrate))
    num_games = len(gsd)
    no_resign_count = 0
    correct_resign_count = 0
    incorrect_resign_count = 0
    game_len_sum = 0
    resigned_game_len_sum = 0
    black_wins = 0
    for gs in gsd.values():
        if not gs.resign_type:
            no_resign_count += 1
            resigned_game_len_sum += gs.total_moves
        elif gs.resign_type == "Correct":
            correct_resign_count += 1
            resigned_game_len_sum += gs.resign_movenum
        else:
            assert gs.resign_type == "Incorrect"
            incorrect_resign_count += 1
            resigned_game_len_sum += gs.resign_movenum
        game_len_sum += gs.total_moves
        if gs.winner == 0:
            black_wins += 1
    avg_len = 1.0*game_len_sum/num_games
    resigned_avg_len = 1.0*resigned_game_len_sum/num_games
    avg_reduction = 1.0*(avg_len-resigned_avg_len)/avg_len
    print("Black won %d/%d (%0.2f%%)" % (black_wins, num_games, 100.0 * black_wins / num_games))
    print("Incorrect uct resigns = %d/%d (%0.2f%%)" % (incorrect_resign_count, num_games, 100.0 * incorrect_resign_count / num_games))
    print("Average game length = %d. Average game length with resigns = %d (%0.2f%% reduction)" % (avg_len, resigned_avg_len, avg_reduction*100))
    print()

if __name__ == "__main__":
    usage_str = """
This script analyzes the debug output from leelaz
to determine the impact of various resign thresholds.

Process flow:
  Run autogtp with debug on:
    autogtp -k savedir -d savedir

  Unzip training and debug files:
    gunzip savedir/*.gz

  Analyze results with this script:
    ./resign_analysis.py savedir/*.txt.debug.0

Note the script takes the debug files hash.txt.debug.0
as the input arguments, but it also expects the training
files hash.txt.0 to be in the same directory."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=usage_str)
    default_resignrates="0.5,0.2,0.15,0.1,0.05"
    parser.add_argument("--r", metavar="Resign rates", dest="resignrates", type=str, default=default_resignrates, help="comma separated resign thresholds (default %s)" % (default_resignrates))
    parser.add_argument("--v", metavar="Verbose", dest="verbose", type=bool, default=False)
    parser.add_argument("data", metavar="files", type=str, nargs="+", help="Debug data files (*.txt.debug.0)")
    args = parser.parse_args()
    resignrates = [float(i) for i in args.resignrates.split(",")]
    for resignrate in (resignrates):
        gs = parseGames(args.data, resignrate, args.verbose)
        resignStats(gs, resignrate)
