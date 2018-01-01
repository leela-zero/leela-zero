# What

A Go program with no human provided knowledge. Using MCTS (but without
Monte Carlo playouts) and a deep residual convolutional neural network stack.

This is a fairly faithful reimplementation of the system described
in the Alpha Go Zero paper "Mastering the Game of Go without Human Knowledge".
For all intents and purposes, it is an open source AlphaGo Zero.

# Wait, what

If you are wondering what the catch is: you still need the network weights.
No network weights are in this repository. If you manage to obtain the
AlphaGo Zero weights, this program will be about as strong, provided you
also obtain a few Tensor Processing Units. Lacking those TPUs, I'd recommend
a top of the line GPU - it's not exactly the same, but the result would still
be an engine that is far stronger than the top humans.

# Gimme the weights

Recomputing the AlphaGo Zero weights will take about 1700 years on commodity
hardware, see for example: http://computer-go.org/pipermail/computer-go/2017-October/010307.html

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

## Windows

Head to the Github releases page at https://github.com/gcp/leela-zero/releases,
download the latest release, unzip, and launch autogtp.exe. It will connect to
the server automatically and do its work in the background, uploading results
after each game. You can just close the autogtp window to stop it.

## macOS and Linux

Follow the instructions below to compile the leelaz binary, then go into
the autogtp subdirectory and follow the instructions there to build the
autogtp binary. Copy the leelaz binary into the autogtp dir, and launch
autogtp.

# I just want to play right now

A small network with some very limited training from human games is available here: https://sjeng.org/zero/best_v1.txt.zip.

It's not very strong right now (and it's trained from human games, boo!).
It will clobber gnugo, but lose to any serious engine. Hey, you said you just
wanted to play right now!

# Compiling

## Requirements

* GCC, Clang or MSVC, any C++14 compiler
* boost 1.58.x or later (libboost-all-dev on Debian/Ubuntu)
* BLAS Library: OpenBLAS (libopenblas-dev) or (optionally) Intel MKL
* zlib library (zlib1g & zlib1g-dev on Debian/Ubuntu)
* Standard OpenCL C headers (opencl-headers on Debian/Ubuntu, or at
https://github.com/KhronosGroup/OpenCL-Headers/tree/master/opencl22/)
* OpenCL ICD loader (ocl-icd-libopencl1 on Debian/Ubuntu, or reference implementation at https://github.com/KhronosGroup/OpenCL-ICD-Loader)
* An OpenCL capable device, preferably a very, very fast GPU, with recent
drivers is strongly recommended (OpenCL 1.2 support should be enough,
even OpenCL 1.1 might work). If you do not have a GPU, modify config.h in the
source and remove the line that says "#define USE_OPENCL".
* The program has been tested on Windows, Linux and macOS.

## Example of compiling and running - Ubuntu

    # Test for OpenCL support & compatibility
    sudo apt install clinfo && clinfo

    # Clone github repo
    git clone https://github.com/gcp/leela-zero
    cd leela-zero/src
    sudo apt install libboost-all-dev libopenblas-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev zlib1g-dev
    make
    cd ..
    wget https://sjeng.org/zero/best_v1.txt.zip
    unzip best_v1.txt.zip
    src/leelaz --weights weights.txt

## Example of compiling and running - macOS

    # Clone github repo
    git clone https://github.com/gcp/leela-zero
    cd leela-zero/src
    brew install boost
    make
    cd ..
    curl -O https://sjeng.org/zero/best_v1.txt.zip
    unzip best_v1.txt.zip
    src/leelaz --weights weights.txt

## Example of compiling and running - Windows

    # Clone github repo
    git clone https://github.com/gcp/leela-zero
    cd leela-zero
    cd msvc
    Double-click the leela-zero2015.sln or leela-zero2017.sln corresponding
    to the Visual Studio version you have.
    # Build from Visual Studio 2015 or 2017
    # Download and extract <https://sjeng.org/zero/best_v1.txt.zip> to msvc/x64/Release
    # msvc/x64/Release/leela-zero --weights weights.txt

# Usage

The engine supports the GTP protocol, version 2, specified at: https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html

Leela Zero is not meant to be used directly. You need a graphical interface
for it, which will interface with Leela Zero through the GTP protocol.

Sabaki (http://sabaki.yichuanshen.de/) is a very nice looking GUI with GTP 2
capability. It should work with this engine. A lot of go software can
interface to an engine via GTP, so look around.

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
as long as the latter is the same for all residual layers. The program will
autodetect the amounts on startup. The first line contains a version number.

* Convolutional layers have 2 weight rows:
    1) convolution weights
    2) channel biases
* Batchnorm layers have 2 weight rows:
    1) batchnorm means
    2) batchnorm variances
* Innerproduct (fully connected) layers have 2 weight rows:
    1) layer weights
    2) output biases

 The convolution weights are in [output, input, filter\_size, filter\_size] order,
 the fully connected layer weights are in [output, input] order.
 The residual tower is first, followed by the policy head, and then the value head.
 All convolution filters are 3x3 except for the ones at the start of the policy and
 value head, which are 1x1 (as in the paper).

There are 18 inputs to the first layer, instead of 17 as in the paper. The
original AlphaGo Zero design has a slight imbalance in that it is easier
for the white player to see the board edge (due to how padding works in
neural networks). This has been fixed in Leela Zero. The inputs are:

```
1) Side to move stones at time T=0
2) Side to move stones at time T=-1  (0 if T=0)
...
8) Side to move stones at time T=-8  (0 if T<=7)
9) Other side stones at time T=0
10) Other side stones at time T=-1   (0 if T=0)
...
16) Other side stones at time T=-8   (0 if T<=7)
17) All 1 if black is to move, 0 otherwise
18) All 1 if white is to move, 0 otherwise
```

Each of these forms a 19 x 19 bit plane.

In the training/caffe directory there is a zero.prototxt file which contains a
description of the full 40 residual block design, in (NVIDIA)-Caffe protobuff
format. It can be used to set up nv-caffe for training a suitable network.
The zero\_mini.prototxt file describes a smaller 12 residual block case. The
training/tf directory contains a 6 residual block version in TensorFlow format,
in the tfprocess.py file.

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
* 1 line with 362 floating point numbers, indicating the search probabilities
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

- [ ] List of package names for more distros
- [x] A real build system like CMake would nice
- [x] Provide or link to self-play tooling
- [x] CPU support for Xeon Phi and for people without a GPU
- [ ] Faster GPU usage via batching
- [ ] Faster GPU usage via Winograd transforms
- [ ] CUDA specific version using cuDNN
- [ ] AMD specific version using MIOpen
- [ ] Faster GPU usage via supporting multiple GPU
(not very urgent, we need to generate the data & network first and this can be
done with multiple processes each bound to a GPU)

# Related links

* Status page of the distributed effort:
http://zero.sjeng.org
* Watch Leela Zero's training games live in a GUI:
https://github.com/fsparv/LeelaWatcher

# License

The code is released under the GPLv3 or later, except for ThreadPool.h, half.hpp
and cl2.hpp, which have specific licenses (compatible with GPLv3) mentioned in
those files.
