# Frequently Asked Questions about Leela Zero #

## Why doesn't the network get stronger every time ##

AZ also had this behavior, besides we're testing our approach right now. Please be patient.

## Why the network size is only 6 blocks comparing to 20 blocks of AZ ##

This is effectively a testing run to see if the system works, and which things are important for doing a full run. I expected 10 to 100 people to run the client, not 600.

Even so, the 20 block version is 13 times more computationally expensive, and expected to make SLOWER progress early on. I think it's unwise to do such a run unless it's proven that the setup works, because you are going to be in for a very long haul.

## Why only dozens of games are played when comparing two networks ##

We use SPRT to decide if a newly trained network is better. A better network is only chosen if SPRT finds it's 95% confident that the new network has a 55% (boils down to 35 elo) win rate over the previous best network.

## Why the game generated during self-play contains quite a few bad moves ##

The MCTS playouts of self-play games is only 1600, and with noises added (For randomness of each move thus training has something to learn from). If you load Leela Zero with Sabaki, you'll probably find it is actually not that weak.

## Very short self-play games ends with White win?! ##

This is expected. Due to randomness of self-play games, once Black choose to pass at the beginning, there is a big chance for White to pass too (7.5 komi advantage for White). See issue #198 for defailed explanation.
