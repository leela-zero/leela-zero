[![Linux Build Status](https://travis-ci.org/leela-zero/leela-zero.svg?branch=next)](https://travis-ci.org/leela-zero/leela-zero)
[![Windows Build Status](https://ci.appveyor.com/api/projects/status/dcvp31x1e0yavrtf/branch/next?svg=true)](https://ci.appveyor.com/project/gcp/leela-zero-8arv1/branch/next)

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

## Using your own hardware

You need a PC with a GPU, i.e. a discrete graphics card made by NVIDIA or AMD,
preferably not too old, and with the most recent drivers installed.

It is possible to run the program without a GPU, but performance will be much
lower. If your CPU is not *very* recent (Haswell or newer, Ryzen or newer),
performance will be outright bad, and it's probably of no use trying to join
the distributed effort. But you can still play, especially if you are patient.

### Windows

Head to the Github releases page at https://github.com/leela-zero/leela-zero/releases,
download the latest release, unzip, and launch autogtp.exe. It will connect to
the server automatically and do its work in the background, uploading results
after each game. You can just close the autogtp window to stop it.

### macOS and Linux

Follow the instructions below to compile the leelaz and autogtp binaries in
the build subdirectory. Then run autogtp as explained in the
[contributing](#contributing) instructions below.
Contributing will start when you run autogtp.

## Using a Cloud provider

Many cloud companies offer free trials (or paid solutions, not discussed here)
that are usable for helping the leela-zero project.

There are community maintained instructions available here:
* [Running Leela Zero client on a Tesla V100 GPU for free (Google Cloud Free Trial)](https://docs.google.com/document/d/1P_c-RbeLKjv1umc4rMEgvIVrUUZSeY0WAtYHjaxjD64/edit?usp=sharing)

* [Running Leela Zero client on a Tesla V100 GPU for free (Microsoft Azure Cloud Free Trial)](https://docs.google.com/document/d/1DMpi16Aq9yXXvGj0OOw7jbd7k2A9LHDUDxxWPNHIRPQ/edit?usp=sharing)

# I just want to play with Leela Zero right now

Download the best known network weights file from [here](https://zero.sjeng.org/best-network), or, if you prefer a more human style,
a (weaker) network trained from human games [here](https://sjeng.org/zero/best_v1.txt.zip).

If you are on Windows, download an official release from [here](https://github.com/leela-zero/leela-zero/releases) and head to the [Usage](#usage-for-playing-or-analyzing-games)
section of this README.

If you are on Unix or macOS, you have to compile the program yourself. Follow
the compilation instructions below and then read the [Usage](#usage-for-playing-or-analyzing-games) section.

# Compiling AutoGTP and/or Leela Zero

## Requirements

* GCC, Clang or MSVC, any C++14 compiler
* Boost 1.58.x or later, headers and program_options, filesystem and system libraries (libboost-dev, libboost-program-options-dev and libboost-filesystem-dev on Debian/Ubuntu)
* zlib library (zlib1g & zlib1g-dev on Debian/Ubuntu)
* Standard OpenCL C headers (opencl-headers on Debian/Ubuntu, or at
https://github.com/KhronosGroup/OpenCL-Headers/tree/master/CL)
* OpenCL ICD loader (ocl-icd-libopencl1 on Debian/Ubuntu, or reference implementation at https://github.com/KhronosGroup/OpenCL-ICD-Loader)
* An OpenCL capable device, preferably a very, very fast GPU, with recent
drivers is strongly recommended (OpenCL 1.1 support is enough). Don't
forget to install the OpenCL driver if this part is packaged seperately
by the Linux distribution (e.g. nvidia-opencl-icd).
If you do not have a GPU, add the define "USE_CPU_ONLY", for example
by adding -DUSE_CPU_ONLY=1 to the cmake command line.
* Optional: BLAS Library: OpenBLAS (libopenblas-dev) or Intel MKL
* The program has been tested on Windows, Linux and macOS.

## Example of compiling - Ubuntu & similar

    # Test for OpenCL support & compatibility
    sudo apt install clinfo && clinfo

    # Clone github repo
    git clone https://github.com/leela-zero/leela-zero
    cd leela-zero
    git submodule update --init --recursive

    # Install build depedencies
    sudo apt install libboost-dev libboost-program-options-dev libboost-filesystem-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev zlib1g-dev

    # Use a stand alone build directory to keep source dir clean
    mkdir build && cd build

    # Compile leelaz and autogtp in build subdirectory with cmake
    cmake ..
    cmake --build .

    # Optional: test if your build works correctly
    ./tests

## Example of compiling - macOS

    # Clone github repo
    git clone https://github.com/leela-zero/leela-zero
    cd leela-zero
    git submodule update --init --recursive

    # Install build depedencies
    brew install boost cmake zlib

    # Use a stand alone build directory to keep source dir clean
    mkdir build && cd build

    # Compile leelaz and autogtp in build subdirectory with cmake
    cmake ..
    cmake --build .

    # Optional: test if your build works correctly
    ./tests

## Example of compiling - Windows

    # Clone github repo
    git clone https://github.com/leela-zero/leela-zero
    cd leela-zero
    git submodule update --init --recursive

    cd msvc
    Double-click the leela-zero2015.sln or leela-zero2017.sln corresponding
    to the Visual Studio version you have.
    # Build from Visual Studio 2015 or 2017

# Contributing

For Windows, you can use a release package, see ["I want to help"](#windows).

Unix and macOS, after finishing the compile and while in the build directory:

    # Copy leelaz binary to autogtp subdirectory
    cp leelaz autogtp

    # Run AutoGTP to start contributing
    ./autogtp/autogtp


# Usage for playing or analyzing games

Leela Zero is not meant to be used directly. You need a graphical interface
for it, which will interface with Leela Zero through the GTP protocol.

The engine supports the [GTP protocol, version 2](https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html).

[Lizzie](https://github.com/featurecat/lizzie/releases) is a client specifically
for Leela Zero which shows live search probilities, a win rate graph, and has
an automatic game analysis mode. Has binaries for Windows, Mac, and Linux.

[Sabaki](http://sabaki.yichuanshen.de/) is a very nice looking GUI with GTP 2
capability.

[LeelaSabaki](https://github.com/SabakiHQ/LeelaSabaki) is modified to
show variations and winning statistics in the game tree, as well as a heatmap
on the game board.

[GoReviewPartner](https://github.com/pnprog/goreviewpartner) is a tool for
automated review and analysis of games using bots (saved as .rsgf files),
Leela Zero is supported.

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

- [ ] Further optimize Winograd transformations.
- [ ] Implement GPU batching in the search.
- [ ] Root filtering for handicap play.
- More backends:
- [ ] MKL-DNN based backend.
- [ ] CUDA specific version using cuDNN or cuBLAS.
- [ ] AMD specific version using MIOpen/ROCm.

# Related links

* Status page of the distributed effort:
https://zero.sjeng.org
* GUI and study tool for Leela Zero:
https://github.com/featurecat/lizzie
* Watch Leela Zero's training games live in a GUI:
https://github.com/fsparv/LeelaWatcher
* Original Alpha Go (Lee Sedol) paper:
https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf
* Alpha Go Zero paper:
https://deepmind.com/documents/119/agz_unformatted_nature.pdf
* Alpha Zero (Go, Chess, Shogi) paper:
https://arxiv.org/pdf/1712.01815.pdf
* AlphaGo Zero Explained In One Diagram:
https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0
* Stockfish chess engine ported to Leela Zero framework:
https://github.com/LeelaChessZero/lczero
* Leela Chess Zero (chess optimized client)
https://github.com/LeelaChessZero/lc0

# License

The code is released under the GPLv3 or later, except for ThreadPool.h, cl2.hpp, half.hpp and the eigen and clblast_level3 subdirs, which have specific licenses (compatible with GPLv3) mentioned in those files.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or
combining it with NVIDIA Corporation's libraries from the
NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
Network library and/or the NVIDIA TensorRT inference library
(or a modified version of those libraries), containing parts covered
by the terms of the respective license agreement, the licensors of
this Program grant you additional permission to convey the resulting
work.
