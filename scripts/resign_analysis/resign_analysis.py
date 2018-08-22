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
import math
import os
import sys

class GameStats:
    def __init__(self, filename):
        self.filename = filename
        self.total_moves = None
        self.resign_movenum = None
        self.resign_type = None     # "Correct", "Wrong"
        self.winner = None

class TotalStats:
    def __init__(self):
        self.num_games = 0
        self.no_resign_count = 0
        self.correct_resign_count = 0
        self.wrong_resign_count = 0
        self.game_len_sum = 0
        self.resigned_game_len_sum = 0
    def calcOverall(self, b, w):
        self.num_games = b.num_games + w.num_games
        self.no_resign_count = b.no_resign_count + w.no_resign_count
        self.correct_resign_count = (
                b.correct_resign_count + w.correct_resign_count)
        self.wrong_resign_count = (
                b.wrong_resign_count + w.wrong_resign_count)
        self.game_len_sum = b.game_len_sum + w.game_len_sum
        self.resigned_game_len_sum = (
                b.resigned_game_len_sum + w.resigned_game_len_sum)

def to_move_str(to_move):
    if (to_move): return "W"
    else: return "B"

def parseGameBody(filename, fh, tfh, verbose, resignthr):
    gs = GameStats(filename)
    movenum = 0
    while 1:
        movenum += 1
        for _ in range(16):
            line = tfh.readline()               # Board input planes
        if not line: break
        to_move = int(tfh.readline())           # 0 = black, 1 = white
        policy_weights = tfh.readline()         # 361 moves + 1 pass
        side_to_move_won = int(tfh.readline())  # 1 for win, -1 for loss
        if not gs.winner:
            if side_to_move_won == 1: gs.winner = to_move
            else : gs.winner = 1 - to_move
        (netwinrate, root_uctwinrate, child_uctwinrate, bestmovevisits) = (
                fh.readline().split())
        netwinrate = float(netwinrate)
        root_uctwinrate = float(root_uctwinrate)
        child_uctwinrate = float(child_uctwinrate)
        bestmovevisits = int(bestmovevisits)
        if side_to_move_won == 1:
            if verbose >= 3:
                print("+", to_move, movenum, netwinrate, child_uctwinrate,
                      bestmovevisits)
            if not gs.resign_type and child_uctwinrate < resignthr:
                if verbose >= 1:
                    print(("Wrong resign -- %s rt=%0.3f wr=%0.3f "
                           "winner=%s movenum=%d") %
                          (filename, resignthr, child_uctwinrate,
                           to_move_str(to_move), movenum))
                    if verbose >= 3:
                        print("policy_weights", policy_weights)
                gs.resign_type = "Wrong"
                gs.resign_movenum = movenum
        else:
            if verbose >= 2:
                print("-", to_move, movenum, netwinrate, child_uctwinrate,
                      bestmovevisits)
            if not gs.resign_type and child_uctwinrate < resignthr:
                if verbose >= 2:
                    print("Correct resign -- %s" % (filename))
                gs.resign_type = "Correct"
                gs.resign_movenum = movenum
    gs.total_moves = movenum
    return gs

def parseGames(filenames, resignthr, verbose, prefixes):
    gsd = {}
    for filename in filenames:
        training_filename = filename.replace(".debug", "")
        with open(filename) as fh, open(training_filename) as tfh:
            version = fh.readline().rstrip()
            assert version == "2"
            (cfg_resignpct, network) = fh.readline().split()
            if prefixes:
                net_name = os.path.basename(network)
                matches = filter(lambda n: net_name.startswith(n), prefixes)
                # Require at least one matching net prefix.
                if not list(matches):
                    continue
            cfg_resignpct = int(cfg_resignpct)
            if cfg_resignpct == 0:
                gsd[filename] = parseGameBody(filename, fh, tfh, verbose, resignthr)
            elif verbose >= 2:
                print("{} was played with -r {}, skipping".format(
                        filename, cfg_resignpct))
    return gsd

def resignStats(gsd, resignthr):
    # [ B wins, W wins, Overall ]
    stats = [ TotalStats(), TotalStats(), TotalStats() ]
    for gs in gsd.values():
        stats[gs.winner].num_games += 1
        if not gs.resign_type:
            stats[gs.winner].no_resign_count += 1
            stats[gs.winner].resigned_game_len_sum += gs.total_moves
        elif gs.resign_type == "Correct":
            stats[gs.winner].correct_resign_count += 1
            stats[gs.winner].resigned_game_len_sum += gs.resign_movenum
        else:
            assert gs.resign_type == "Wrong"
            stats[gs.winner].wrong_resign_count += 1
            stats[gs.winner].resigned_game_len_sum += gs.resign_movenum
        stats[gs.winner].game_len_sum += gs.total_moves
    stats[2].calcOverall(stats[0], stats[1])
    print("Resign thr: %0.2f - Black won %d/%d (%0.2f%%)" % (
        resignthr,
        stats[0].num_games,
        stats[0].num_games+stats[1].num_games,
        100 * stats[0].num_games / (stats[0].num_games+stats[1].num_games)))
    for winner in (0,1,2):
        win_str = 'Overall   '
        if winner==0:
            win_str = 'Black wins'
        elif winner==1:
            win_str = 'White wins'
        if stats[winner].num_games == 0:
            print("    No games to report")
            continue
        avg_len = stats[winner].game_len_sum / stats[winner].num_games
        resigned_avg_len = (stats[winner].resigned_game_len_sum /
                            stats[winner].num_games)
        avg_reduction = (avg_len - resigned_avg_len) / avg_len
        print(("%s - Wrong: %d/%d (%0.2f%%) Correct: %d/%d (%0.2f%%) "
               "No Resign: %d/%d (%0.2f%%)") % (
            win_str,
            stats[winner].wrong_resign_count,
            stats[winner].num_games,
            100 * stats[winner].wrong_resign_count / stats[winner].num_games,
            stats[winner].correct_resign_count,
            stats[winner].num_games,
            100 * stats[winner].correct_resign_count / stats[winner].num_games,
            stats[winner].no_resign_count,
            stats[winner].num_games,
            100 * stats[winner].no_resign_count / stats[winner].num_games))
        print("%s - Average game length: %d/%d (%0.2f%% reduction)" % (
            win_str, resigned_avg_len, avg_len, avg_reduction*100))
    print()
    return stats

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
    ./resign_analysis.py savedir/*.debug.txt.0

Note the script takes the debug files hash.debug.txt.0
as the input arguments, but it also expects the training
files hash.txt.0 to be in the same directory."""
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=usage_str)
    default_resignthrs="0.5,0.2,0.15,0.1,0.05,0.02,0.01"
    parser.add_argument(
            "-r", metavar="Resign_thresholds", dest="resignthrs", type=str,
            default=default_resignthrs,
            help="comma separated resign thresholds (default {})".format(
                    default_resignthrs))
    parser.add_argument(
            "-R", metavar="Resign_rate", dest="resignrate", type=float,
            help="If specified, a search is performed that finds the maximum \
            resign threshold that can be set without exceeding the given \
            resign rate")
    parser.add_argument(
            "-v", metavar="Verbose", dest="verbose", type=int, default=0,
            help="Verbosity level (default 0)")
    parser.add_argument(
            "data", metavar="files", type=str, nargs="+",
            help="Debug data files (*.debug.txt.0)")
    parser.add_argument(
            "-n", metavar="Prefix", dest="networks", nargs="+",
            help="Prefixes of specific networks to analyze")
    args = parser.parse_args()
    resignthrs = [float(i) for i in args.resignthrs.split(",")]
    if args.networks:
        print("Analyzing networks starting with: {}".format(
                ",".join(args.networks)))

    for resignthr in (resignthrs):
        gsd = parseGames(args.data, resignthr, args.verbose, args.networks)
        if gsd:
            resignStats(gsd, resignthr)
        else:
            print("No games to analyze (for more info try running with -v 2)")

    if args.resignrate:
        L = 0.0
        R = 0.5
        while L < R :
            resignthr = math.floor((L + R) * 50) / 100
            gsd = parseGames(args.data, resignthr, args.verbose, args.networks)
            if not gsd:
                print("No games to analyze (for more info try running with -v 2)")
                break
            stats = resignStats(gsd, resignthr)
            wrong_rate = stats[2].wrong_resign_count / stats[2].num_games
            if wrong_rate > args.resignrate:
                if R == resignthr:
                    R = (math.floor(resignthr * 100) - 1) / 100
                else:
                    R = resignthr
            else:
                L = (math.floor(resignthr * 100) + 1) / 100
        if (L == R):      
            print(("The highest the resign threshold should be set to: %0.2f")
                  % (R - 0.01))
