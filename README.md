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
also obtain a few Tensor Processing Units. Lacking those TPU, I'd recommend
a top of the line GPU - it's not the same, but the result would stil be an
engine that is far stronger than the top humans.

# Gimme the weights

Recomputing the AlphaGo Zero weights will take about 1700 years on commodity
hardware, see for example: http://computer-go.org/pipermail/computer-go/2017-October/010307.html

One reason for publishing this program is that we are setting up a public,
distributed effort to repeat the work. Working together, and especially
when starting on a smaller scale, it will take less than 1700 years to get
a good network (which you can feed into this program, suddenly making it strong).
Further details about this will be announced soon.

# I just want to play

A small network with some very limited training from human games is available here: https://sjeng.org/zero/supervised.txt.zst

It's not very strong right now. It will clobber gnugo, but lose to any serious
engine.

I plan to update this network with more training when available - just feeding it
into this program will make it stronger. Unzip it with unzstd (zstandard/zstd package)
and specify it on the command line with the -w option.

# Compiling

Requirements:

* GCC, Clang or MSVC, any C++14 compiler
* OpenBLAS or (optional) Intel MKL
* OpenCL C++ headers, https://github.com/KhronosGroup/OpenCL-CLHPP
(You can just copy input_cl.hpp into CL/cl2.hpp)
* Standard OpenCL headers (opencl-headers on Debian/Ubuntu)
* OpenCL ICD loader (ocl-icd-libopencl1, or reference implementation at https://github.com/KhronosGroup/OpenCL-ICD-Loader)
* An OpenCL capable device, preferably a very, very fast GPU, with drivers
(OpenCL 1.2 support should be enough, even OpenCL 1.1 might work)
* The program has been tested on Windows, Linux and MacOS.

Run make and hope it works. You might need to edit the paths in the Makefile.

# Usage

The engine supports the GTP protocol, version 2, specified at: https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html

Add the --gtp commandline option to enable it. You will need a weights file,
specify that with the -w option.

All required commands are supported, as well as the tournament subset, and
"loadsgf". The full set can be seen with "list_commands". The time control
can be specified over GTP via the time\_settings command. The kgs-time\_settings
extension is also supported. These have to be supplied by the GTP 2 interface,
not via the command line!

Sabaki (http://sabaki.yichuanshen.de/) is a very nice looking GUI with GTP 2
capablity. It should work with this engine.

# Weights format

The weights file is a text file with each line containing a row of coefficients.
The layout of the network is as in the AlphaGo Zero paper, but the number of
residual blocks is allowed to vary. The program will autodetect the amount on
startup.

* Convolutional layers have 2 weight rows:
    1) convolution weights
    2) channel biases
* Batchnorm layers have 3 weight rows:
    1) batchnorm means
    2) batchnorm variances
    3) a single 0 (this is an nv-caffe oddity)
* Innerproduct (fully connected) layers have 2 weight rows:
    1) weights
    2) output biases

You might note there are 18 inputs instead of 17 as in the paper. The original
AlphaGo Zero design has a slight imbalance in that it is easier for the white
player to see the board edge (due to how padding works in neural networks).
This has been fixed in Leela Zero. The inputs are:

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

The zero.prototxt file contains a description of the full 40 residual layer design,
in (NVIDIA)-Caffe protobuff format. It can be used to set up nv-caffe for training
a suitable network. The zero\_mini.protoxt file describes the 12 residual layer
case that was used for the example supervised network listed in "I just want to
play".

# Todo

- [ ] Less atrocious build instructions, list of package names for distros
- [ ] Add the ability to provide the softmax temperature on the search results
(this is required for randomizing the engine more in the opening)
- [ ] CPU support for Xeon Phi and for people without GPU
- [ ] Faster GPU usage via batching
- [ ] Faster GPU usage via Winograd transforms
- [ ] CUDA specific version using cuDNN
- [ ] AMD specific version using MIOpen
- [ ] Faster GPU usage via supporting multiple GPU
(not very urgent, we need to generate the data & network first and this can be
done with multiple processes each bound to a GPU)
