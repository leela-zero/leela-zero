[![Linux Build Status](https://travis-ci.org/gcp/leela-zero.svg?branch=next)](https://travis-ci.org/gcp/leela-zero)
[![Windows Build Status](https://ci.appveyor.com/api/projects/status/pf1hcgly8f1a8iu0/branch/next?svg=true)](https://ci.appveyor.com/project/gcp/leela-zero/branch/next)



# What

A Go program with no human provided knowledge. Using MCTS (but without
Monte Carlo playouts) and a deep residual convolutional neural network stack.

This is a fairly faithful reimplementation of the system described
in the Alpha Go Zero paper "[Mastering the Game of Go without Human Knowledge](https://deepmind.com/documents/119/agz_unformatted_nature.pdf)".
For all intents and purposes, it is an open source AlphaGo Zero.

# Wait, what?

If you are wondering what the catch is: you still need the network weights.
No network weights are in this repository. If you manage to obtain the
AlphaGo Zero weights, this program will be about as strong, provided you
also obtain a few Tensor Processing Units. Lacking those TPUs, I'd recommend
a top of the line GPU - it's not exactly the same, but the result would still
be an engine that is far stronger than the top humans.

# Gimme the weights

Recomputing the AlphaGo Zero weights will [take about 1700 years on commodity hardware](http://computer-go.org/pipermail/computer-go/2017-October/010307.html).

One reason for publishing this program is that we are running a public,
distributed effort to repeat the work. Working together, and especially
when starting on a smaller scale, it will take less than 1700 years to get
a good network (which you can feed into this program, suddenly making it strong).

# I want to help

You need a PC with a GPU, i.e. a discrete graphics card made by NVIDIA or AMD,
preferably not too old, and with the most recent drivers installed.

It is possible to run the program without a GPU, but performance will be much
lower. If your CPU is not *very* recent (Haswell or newer, Ryzen or newer),
performance will be outright bad, and it's probably of no use trying to join
the distributed effort. But you can still play, especially if you are patient.

[Running Leela Zero client on a Tesla K80 GPU for free (Google Colaboratory)](COLAB.md)

## Windows

Head to the Github releases page at https://github.com/gcp/leela-zero/releases,
download the latest release, unzip, and launch autogtp.exe. It will connect to
the server automatically and do its work in the background, uploading results
after each game. You can just close the autogtp window to stop it.

## macOS and Linux

Follow the instructions below to compile the leelaz binary, then go into
the autogtp subdirectory and follow [the instructions there](autogtp/README.md)
to build the autogtp binary. Copy the leelaz binary into the autogtp dir, and
launch autogtp.

# I just want to play right now

Download the best known network weights file from: http://zero.sjeng.org/best-network

And head to the [Usage](#usage) section of this README.

If you prefer a more human style, a network trained from human games is available here: https://sjeng.org/zero/best_v1.txt.zip.

# Compiling

## Requirements

* GCC, Clang or MSVC, any C++14 compiler
* Boost 1.58.x or later, headers and program_options library (libboost-dev & libboost-program-options-dev on Debian/Ubuntu)
* BLAS Library: OpenBLAS (libopenblas-dev) or (optionally) Intel MKL
* zlib library (zlib1g & zlib1g-dev on Debian/Ubuntu)
* Standard OpenCL C headers (opencl-headers on Debian/Ubuntu, or at
https://github.com/KhronosGroup/OpenCL-Headers/tree/master/opencl22/)
* OpenCL ICD loader (ocl-icd-libopencl1 on Debian/Ubuntu, or reference implementation at https://github.com/KhronosGroup/OpenCL-ICD-Loader)
* An OpenCL capable device, preferably a very, very fast GPU, with recent
drivers is strongly recommended (OpenCL 1.1 support is enough).
If you do not have a GPU, modify config.h in the source and remove
the line that says "#define USE_OPENCL".
* The program has been tested on Windows, Linux and macOS.

## Example of compiling and running - Ubuntu

    # Test for OpenCL support & compatibility
    sudo apt install clinfo && clinfo

    # Clone github repo
    git clone https://github.com/gcp/leela-zero
    cd leela-zero/src
    sudo apt install libboost-dev libboost-program-options-dev libopenblas-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev zlib1g-dev
    make
    cd ..
    wget http://zero.sjeng.org/best-network
    src/leelaz --weights best-network

## Example of compiling and running - macOS

    # Clone github repo
    git clone https://github.com/gcp/leela-zero
    cd leela-zero/src
    brew install boost
    make
    cd ..
    curl -O http://zero.sjeng.org/best-network
    src/leelaz --weights best-network

## Example of compiling and running - Windows

    # Clone github repo
    git clone https://github.com/gcp/leela-zero
    cd leela-zero
    cd msvc
    Double-click the leela-zero2015.sln or leela-zero2017.sln corresponding
    to the Visual Studio version you have.
    # Build from Visual Studio 2015 or 2017
    # Download <http://zero.sjeng.org/best-network> to msvc\x64\Release
    msvc\x64\Release\leelaz.exe --weights best-network

## Example of compiling and running - CMake (macOS/Ubuntu)

    # Clone github repo
    git clone https://github.com/gcp/leela-zero
    cd leela-zero
    git submodule update --init --recursive

    # Use stand alone directory to keep source dir clean
    mkdir build && cd build
    cmake ..
    make leelaz
    make tests
    ./tests
    curl -O http://zero.sjeng.org/best-network
    ./leelaz --weights best-network


# Usage

The engine supports the [GTP protocol, version 2](https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html).

Leela Zero is not meant to be used directly. You need a graphical interface
for it, which will interface with Leela Zero through the GTP protocol.

[Lizzie](https://github.com/featurecat/lizzie/releases) is a client specifically
for Leela Zero which shows live search probilities, a win rate graph, and has
an automatic game analysis mode. Has binaries for Windows, Mac, and Linux.

[Sabaki](http://sabaki.yichuanshen.de/) is a very nice looking GUI with GTP 2
capability.

[LeelaSabaki](https://github.com/SabakiHQ/LeelaSabaki) is modified to
show variations and winning statistics in the game tree, as well as a heatmap
on the game board.

A lot of go software can interface to an engine via GTP,
so look around.

Add the --gtp commandline option on the engine command line to enable Leela
Zero's GTP support. You will need a weights file, specify that with the -w option.

All required commands are supported, as well as the tournament subset, and
"loadsgf". The full set can be seen with "list_commands". The time control
can be specified over GTP via the time\_settings command. The kgs-time\_settings
extension is also supported. These have to be supplied by the GTP 2 interface,
not via the command line!

# Weights format

The weights file is a text file with each line containing a row of coefficients.
The layout of the network is as in the AlphaGo Zero paper, but any number of
residual blocks is allowed, and any number of outputs (filters) per layer,
as long as the latter is the same for all layers. The program will autodetect
the amounts on startup. The first line contains a version number.

* Convolutional layers have 2 weight rows:
    1) convolution weights
    2) channel biases
* Batchnorm layers have 2 weight rows:
    1) batchnorm means
    2) batchnorm variances
* Innerproduct (fully connected) layers have 2 weight rows:
    1) layer weights
    2) output biases

The convolution weights are in [output, input, filter\_size, filter\_size]
order, the fully connected layer weights are in [output, input] order.
The residual tower is first, followed by the policy head, and then the value
head. All convolution filters are 3x3 except for the ones at the start of the policy and value head, which are 1x1 (as in the paper).

There are 18 inputs to the first layer, instead of 17 as in the paper. The
original AlphaGo Zero design has a slight imbalance in that it is easier
for the black player to see the board edge (due to how padding works in
neural networks). This has been fixed in Leela Zero. The inputs are:

```
1) Side to move stones at time T=0
2) Side to move stones at time T=-1  (0 if T=0)
...
8) Side to move stones at time T=-7  (0 if T<=6)
9) Other side stones at time T=0
10) Other side stones at time T=-1   (0 if T=0)
...
16) Other side stones at time T=-7   (0 if T<=6)
17) All 1 if black is to move, 0 otherwise
18) All 1 if white is to move, 0 otherwise
```

Each of these forms a 19 x 19 bit plane.

In the training/caffe directory there is a zero.prototxt file which contains a
description of the full 40 residual block design, in (NVIDIA)-Caffe protobuff
format. It can be used to set up nv-caffe for training a suitable network.
The zero\_mini.prototxt file describes a smaller 12 residual block case. The
training/tf directory contains the network construction in TensorFlow format,
in the tfprocess.py file.

Expert note: the channel biases seem redundant in the network topology
because they are followed by a batchnorm layer, which is supposed to normalize
the mean. In reality, they encode "beta" parameters from a center/scale
operation in the batchnorm layer, corrected for the effect of the batchnorm mean/variance adjustment. At inference time, Leela Zero will fuse the channel
bias into the batchnorm mean, thereby offsetting it and performing the center operation. This roundabout construction exists solely for backwards
compatibility. If this paragraph does not make any sense to you, ignore its
existence and just add the channel bias layer as you normally would, output
will be correct.

# Training

## Getting the data

At the end of the game, you can send Leela Zero a "dump\_training" command,
followed by the winner of the game (either "white" or "black") and a filename,
e.g:

    dump_training white train.txt

This will save (append) the training data to disk, in the format described below,
and compressed with gzip.

Training data is reset on a new game.

## Supervised learning

Leela can convert a database of concatenated SGF games into a datafile suitable
for learning:

    dump_supervised sgffile.sgf train.txt

This will cause a sequence of gzip compressed files to be generated,
starting with the name train.txt and containing training data generated from
the specified SGF, suitable for use in a Deep Learning framework.

## Training data format

The training data consists of files with the following data, all in text
format:

* 16 lines of hexadecimal strings, each 361 bits longs, corresponding to the
first 16 input planes from the previous section
* 1 line with 1 number indicating who is to move, 0=black, 1=white, from which
the last 2 input planes can be reconstructed
* 1 line with 362 (19x19 + 1) floating point numbers, indicating the search probabilities
(visit counts) at the end of the search for the move in question. The last
number is the probability of passing.
* 1 line with either 1 or -1, corresponding to the outcome of the game for the
player to move

## Running the training

For training a new network, you can use an existing framework (Caffe,
TensorFlow, PyTorch, Theano), with a set of training data as described above.
You still need to contruct a model description (2 examples are provided for
Caffe), parse the input file format, and outputs weights in the proper format.

There is a complete implementation for TensorFlow in the training/tf directory.

### Supervised learning with TensorFlow

This requires a working installation of TensorFlow 1.4 or later:

    src/leelaz -w weights.txt
    dump_supervised bigsgf.sgf train.out
    exit
    training/tf/parse.py train.out

This will run and regularly dump Leela Zero weight files to disk, as
well as snapshots of the learning state numbered by the batch number.
If interrupted, training can be resumed with:

    training/tf/parse.py train.out leelaz-model-batchnumber

# Todo

- [ ] Optimize Winograd transformations.
- [ ] Implement GPU batching.
- [ ] Parameter setting over GTP.
- More backends:
- [ ] Eigen based BLAS backend.
- [ ] MKL-DNN based backend.
- [ ] CUDA specific version using cuDNN.
- [ ] AMD specific version using MIOpen.

# Related links

* Status page of the distributed effort:
http://zero.sjeng.org
* Watch Leela Zero's training games live in a GUI:
https://github.com/fsparv/LeelaWatcher
* GUI and study tool for Leela Zero:
https://github.com/CamWagner/lizzie
* Stockfish chess engine ported to Leela Zero framework:
https://github.com/glinscott/leela-chess
* Original Alpha Go (Lee Sedol) paper:
https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf
* Newer Alpha Zero (Go, Chess, Shogi) paper:
https://arxiv.org/pdf/1712.01815.pdf
* AlphaGo Zero Explained In One Diagram:
https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0

# License

The code is released under the GPLv3 or later, except for ThreadPool.h, cl2.hpp,
half.hpp and the clblast_level3 subdirs, which have specific licenses (compatible with GPLv3) mentioned in those files.

# Dynamic Komi

Implemented and maintained by @alreadydone in this repository.
See https://github.com/gcp/leela-zero/pull/1772 for the latest version.

### Instructions 使用说明
+ Dynamic komi works by adjusting the color plane inputs of the neural network (which the network interprets/perceives as komi to some extent, since there's a trend of black winrate decreasing with komi increasing for many networks) to make it output winrates within a certain range, to avoid too low winrates causing desparate moves being played and/or too high winrates causing slack moves being played. It currently works as a hack as no network has been trained on games with komi values other than +/-7.5, so the winrates under other komi cannot be trusted, but they are meaningful relative to each other, and dynamic komi does improve handicap performance and allows the engine to play non-slack when it's leading.
动态贴目是通过调整颜色平面的输入起作用：某种程度上，网络可以将颜色平面的输入解释/感知为贴目信息，因为许多网络在多数局面下黑胜率随贴目上升有下降趋势。将胜率调整到合理范围，有利于防止胜率过低导致的绝望放弃“发疯”着法（逃死征、往一二路落子等）及胜率过高导致的过于放松的着法（大龙不杀、官子损目等）。但由于网络并未在正负7.5以外的贴目下对战训练过，在其他贴目下的胜率是不准确的，但这些胜率的相对高低是有意义的，确能提高引擎的让子/不退让能力。
+ Not theoretically usable with any network that says `Detecting residual layers...v2` (so far these are just ELFv0 62b5417b and ELFv1 d13c4099). 不适用于ELFv0或ELFv1。
+ Without parameters, the engine should behave identically to the official /next branch (not necessarily the latest). 不加参数时，引擎的表现应与官方next分支完全一致。
+ Parameters recommended for handicap games 让子棋使用参数: --handicap -r 0 --target-komi 0
Raising --max-wr and --min-wr may lead to better performance depending on the opponent and maybe the network; the defaults are --max-wr 0.12 --min-wr 0.06 --wr-margin 0.03 for the handicap mode. In general, the lower these parameters are, the more aggressively (but possibly unreasonably or desparately) the engine plays. 
根据对手或权重的不同，加大--max-wr和--min-wr可能带来更好的表现（让子模式下默认是--max-wr 0.12 --min-wr 0.06 --wr-margin 0.03）。一般而言，两参数值越低，引擎表现越进取，但过低则会出现过多无理手或绝望放弃的着法。
+ Parameter recommended for not being slack when leading 不退让使用参数: --nonslack
Defaults 默认值: --max-wr 0.9 --min-wr 0.1 --wr-margin 0.1
+ Development has focused on stronger handicap performance. With --handicap --pos -r 0 -t 1 -p 3000 -w GX37 against ELFv0 -p 1 (@Splee99's test, but the same number of playouts takes somewhat more time with dynamic komi) with 7 handicap stones, it scored 3-3, and with 8 stones it scored 4-20 (GX37 runs into ladder issues early on in many of the lost games). It should perform pretty well against human as well with up to 5-6 stones according to people who tested.

### Changelog: Improvements and over komi/endgame-v0.3x releases 在上一主要版本（让子版、不退让版）基础上的改进/修正的缺陷
+ Safe with AutoGTP, since adjusted komi not be enabled without --handicap, --nonslack or --target-komi parameters. Also the single engine can now play both in handicap mode and in non-slack mode, or under different komi. 可以安全地与跑谱程序AutoGTP一同使用，因为贴目在未启用--handicap, --nonslack 或 --target-komi 选项/参数时不会改变。让子、不退让及不同贴目的功能现包含在一个引擎中。
+ --pos and --neg are implemented. Since it's observed that some networks seem to generalize better (monotonicity of winrate w.r.t. komi) in the positive komi range and some in the negative range, it's natural to use only winrates from one of the ranges, which is enabled with the --pos or --neg option.
实现了--pos和--neg选项。因为观察到有些权重在正贴目范围泛化更好(具备胜率随贴目的单调性)，有些在负贴目范围更好，所以设置了以下两个选项用来选择性地采用正贴目或负贴目下的胜率。
+ More accurate/stable komi/winrate adjustment; addresses the observed problem of plummeting winrate after search causing engine go crazy. Now we don't just use the single position (and its rotations/reflections) at the root node for komi adjustment, we collect positions during tree search. Accuracy/stability can potentially be improved by increasing --adj-positions (default 200). On fast machines --adj-positions 2000 may be reasonable. 更精确/稳定的贴目/胜率调整，在树搜索过程中收集用于调整贴目的局面，可以解决搜索后胜率跳水导致的胜率过低发疯问题。--adj-positions越高，调整越精确；默认--adj-positions 200，强机2000或许合适。
+ Better strategy of komi adjustment: (1) retires --mid-wr and replaces it with --wr-margin. Before, when winrate goes out of the range [min,max], it's adjusted to mid. Now, if winrate goes above max, it's adjusted to max - margin, and if it goes below min, it's adjusted to min + margin. (2) In nonslack mode, adjust back to the target komi if the winrate under the target komi is within [min + margin, max - margin]; addresses https://github.com/alreadydone/lz/issues/42 by @anonymousAwesome and should mostly eliminate the possibility of losing a winning game. 更好的贴目调整策略：(1) 去除--mid-wr参数并以--wr-margin代替。原让子/不退让版中，胜率超过--max-wr或低于--min-wr都会导致其被调整到--mid-wr；在此新版本中，胜率超过max则调整到max - margin，而低于min则调整到min + margin。(2) 在不退让模式中，如目标贴目下的胜率在[min+margin,max-margin]范围内，则直接调整回目标贴目。
+ Thanks to everyone who tested the previous versions! 感谢大家的测试！


### Meaning of options/parameters 选项/参数释义 
(An abbreviated version of the following can be displayed with the option -h or --help. 使用选项-h或--help可显示以下说明的缩略版。)
+ --handicap  Handicap mode 让子模式
+ --nonslack  Non-slack mode 不退让模式
+ --max-wr  Maximal white winrate 最大白胜率
+ --min-wr  Minimal white winrate 最小白胜率
+ --wr-margin  Winrate margin 胜率调整宽余量
Example: `--max-wr 0.9 --min-wr 0.1 --wr-margin 0.05` means that the winrate will be adjusted to 0.85 = 0.9 - 0.05 if the winrate under the current komi exceeds 90%, and to 0.15 = 0.1 + 0.05 if it drops below 10%. There are some exceptions to be detailed later. 
示例：`--max-wr 0.9 --min-wr 0.1 --wr-margin 0.05` 意味着(当前贴目下的)胜率超过90%时将被调整到85% (0.9 - 0.05 = 0.85)，低于10%时将被调整到15%。有一些例外情况见后。
+ --target-komi  Target komi 目标贴目 (Default 默认值: --target-komi 7.5)
Normally, only 7.5, -7.5 and possibly 0 are recommended, but see FAQ. 在一般情况下，只推荐设置7.5, -7.5两个值(0或许也能行)，不过在FAQ中有另一种不同的用法。
In handicap mode, the komi will be adjusted back to target-komi if the (white) winrate under current komi is above max and the winrate under the target komi is above (min + margin). In non-slack mode, this happens if the winrate under the target komi is between (min + margin) and (max - margin), regardless of the winrate under the current komi. 在让子模式中，如果当前贴目下的白胜率超过最大值且目标贴目下的白胜率超过最小值+宽余量，贴目将被调整回目标贴目。在不退让模式中，如果目标贴目下的白胜率在(最小值+宽余量)和(最大值-宽余量)之间，无论当前贴目下胜率如何，贴目都将被调整回目标贴目。
--target-komi is also used for scoring after double-pass, but the komi for scoring purpose can be changed any time during the game with the GTP command `komi` without affecting the dynamic komi adjustment. 目标贴目也用于终局数子判定胜负，但数子使用的贴目随时可用GTP命令komi更改，对动态贴目调整没有影响。
Notice: When handicap or non-slack mode is enabled, scoring doesn't take into account the number of handicap stones; this agrees with Chinese scoring in Sabaki and is more reasonable, since the neural network is ignorant of the handicap. 注意：启用让子/不退让模式时，终局确定胜负时黑方不会将让子贴还给白方；这与Sabaki中的中国规则一致，并且更合理，因为神经网络并不知道让子数。
+ --adj-positions  Number of positions to collect during tree search for komi adjustment 树搜索中要收集的用于调整贴目的局面的数量 (Default 默认值: --adj-positions 200)
+ --adj-pct  Percentage of positions to use for komi adjustment 收集的局面中实际用于调整贴目的局面的百分比 (Default 默认值: --adj-pct 4) We choose the 4% positions that have winrates closest to the average collected winrate and only try different komi values for these positions. 选取收集的所有局面中4%胜率最接近平均胜率的，仅对这些局面尝试不同贴目，将胜率调整到范围内。
+ --num-adj  Maximal number of adjustments during each genmove/lz-analyze (Default 默认值：--num-adj 1) Probably no need to change. 每次genmove/lz-analyze命令中调整贴目的最大次数，应无改动的必要。
By default, each komi adjustment takes about 2x9x200x4%=144 forward passes of the network (somewhat more time than 144 playouts cost). Also notice that any komi adjustment will destroy the search tree, though in handicap games the opponent will frequently play unexpected moves that destroy the tree. 默认设置下，每次调整贴目需要调用网络2x9x200x4%=144次（用时比144po稍多）。任何贴目的调整都会清空搜索树，不过让子棋中对手不理想的/意料之外的着法也会砍掉很大一部分搜索树。

+ --pos  Use winrates with "positive" (>=-7.5) komi values (for side-to-move) only, i.e. use winrates at black's turns only when komi is >=-7.5, and use winrates at white's turns only when komi is <=7.5. 仅采用不小于-7.5的贴目(对轮走方而言)下的胜率，也即：只有贴目不小于-7.5时才采用轮黑时的胜率，只有贴目不大于7.5时才采用轮白时的胜率。
+ --neg  Use winrates with "negative" (<=7.5) komi values (for side-to-move) only. 仅采用不大于7.5的贴目(对轮走方而言)下的胜率。
--pos or --neg are automatically set at startup by default using a test (GTP command dyn_komi_test) for the empty board (previously done with the "wrout" utility). Some networks may be rejected by this test, but you can force using them for dynamic komi with the option --tg-auto-pn and possibly manually set --pos or --neg. 默认情况下--pos和--neg选项在引擎启动时会通过一个测试(GTP命令dyn_komi_test)自动设置。测试可能认为有些权重不适用于动态贴目，但如果想强制在让子/不退让模式下使用权重，可以加上--tg-auto-pn选项（将不会自动设置--pos或--neg）。
+ --tg-auto-pn  Toggle automatic setting of --pos or --neg. 开启/关闭--pos/--neg的自动设置。

+ --fixed-symmetry  Use fixed symmetry instead of random symmetry (rotation/reflection); to be followed by the symmetry you'd like to use exclusively (an integer from 0 to 7). Since the winrates may differ a lot with different symmetries under high komi, better performance might be achieved by fixing a symmetry (but there could be more blind spots as well). 使用固定的旋转/镜像而不使用随机旋转/镜像。因为贴目高时旋转/镜像后的胜率彼此可能很不一样，采用固定的旋转/镜像或不做旋转/镜像可能增强棋力（但也可能增加盲点）。
Example: `--fixed-symmetry 0` means that we use the identity symmetry (do no rotation/reflection of the board before feeding into the neural net).
示例：`--fixed-symmetry 0`表示将局面不经旋转/镜像直接输入到神经网络中。

+ --tg-sure-backup  Toggle backup for nodes with winrates invalid under --pos or --neg. Suppose that --neg is enabled, the komi is 8.0 and it's black to move, then the winrate at this node would be invalid: if backup is enabled (default), the winrate of the first child (where it's white to move) will be used instead; if backup is disabled, the winrate will simply be discarded. Setting this option will change the shape of the tree so it is not recommended. 实验性质的选项，不推荐设置。

+ --tg-orig-policy  Toggle original/adjusted policy under --pos or --neg. Without this option, if --pos is set in a handicap game and the komi is high, the policy for black will be adjusted to suggest more aggressive moves as if black isn't leading by a lot, but the policy for white will not be adjusted so it suggests super-aggressive/desparate moves; if --neg is set, the policy for white will be adjusted to suggest not so aggressive/desparate moves, but the policy for black will not be adjusted so it suggests very conservative/safe moves. With this option, both black and white policy will be adjusted, so the moves suggested are closer to those in an even game. May worth experimenting with. 使用--pos或--neg选项时，启用该选项将总是采用调整后的网络概率，使得网络建议的着法更接近形势差不多时的着法（但让子时白更积极）。让子模式、未启用该选项时，如启用--pos选项，轮黑时的建议着法是调整过的，不会过于保守，而轮白时的建议着法则未经调整，会非常进取；如启用--neg选项，轮白时的建议着法不会过于进取，而轮黑时的建议着法会非常保守。

+ --tg-dyn-fpu  Toggle dynamic first-play urgency (FPU). Under handicap/non-slack mode the default is using the dynamic evaluation (the average neural network winrate of the tree below the parent) of the parent node (minus reduction) as FPU; without --handicap or --nonslack the default is using the neural network winrate of the single parent node as FPU. Since the winrates may vary a lot under high komi, setting this option is not recommended. 实验性质的选项，不推荐设置。


### FAQ/Technical details 常见问题和技术细节
+ What is opp_komi? opp_komi是什么？
Since the networks are never trained on komi values other than +/-7.5, the winrates under other komi are inaccurate, and it turns out that many networks generalize differently at the positive range than at the negative range, and a different komi value is needed at black's turns than at white's turns to bring the winrates close together and within the range. The displayed komi is the input at the current side-to-move's turns, while opp_komi is the komi input to the network at the opponent's turns. 由于网络对贴目的感受的不精确性，轮黑走和轮白走时的贴目要设为不同的值，才能分别将胜率调整到预定范围。komi是轮己方走时输入网络的贴目值，而opp_komi是轮对方走时的值。

+ How does dyn_komi_test and automatic setting of --pos or --neg work? 引擎如何自动设置--pos和--neg？
用空棋盘检测胜率随贴目的单调性，得出两个分数(score)，如果正贴目范围的分数超过5e-2就认为权重在正贴目范围不能使用，自动设置--neg，反之自动设置--pos。
Take the initial empty board position, from -300 to 300 komi with increment 0.5, record every place where the black winrate is increasing instead of decreasing, and add up the increments separately for the positive and negative ranges; finally add 1 - (winrate at -300 komi) to the negative range sum, and add (winrate at 300 komi) to the positive range sum. If the positive range sum exceeds 0.05, then the positive range is deemed unusable and --neg is automatically set; similarly for the negative range.

+ How would I use the engine to play games with 6.5 komi as white (e.g. on Fox) or other values? 如何用于6.5贴目的对局？
You may be tempted to use --target-komi 6.5, but due to inaccuracy of winrates under komi values other than +/-7.5, it's safer to set --target-komi 3, for example. Then you may type in GTP command `komi 6.5` to set the komi for scoring. 设置 --target-komi 6.5 看似能行，但因为网络对贴目感受的不精确性，启动时设置譬如说 --target-komi 3 会更为安全。之后可以通过GTP命令`komi 6.5`设置终局数子时采用的贴目（中国/日韩规则的不同此处无法保证正确处理）。

### Possible further improvements
+ Parallelizing and batching komi adjustment (mean_white_eval) and dyn_komi_test will save some time.
+ Instead of saving GameState for komi adjustment, maybe only save KoState + input_data (output of gather_features); should save some memory.
+ Not collecting half of the positions when --pos or --neg is enabled.
