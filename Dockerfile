FROM ubuntu:16.04

# Install
RUN apt-get -qq update
RUN apt-get install -y cmake g++
RUN apt-get install -y libboost-all-dev libopenblas-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev zlib1g-dev
RUN apt-get install -y qt5-default qt5-qmake

RUN mkdir -p /src/
WORKDIR /src/

# GPU build
RUN mkdir -p /src/gpu/
WORKDIR /src/gpu/
RUN CXX=g++ CC=gcc cmake -DUSE_OPENBLAS=1 -DUSE_OPENCL=1 ..
RUN make -j2

# CPU build
RUN mkdir -p /src/cpu
WORKDIR /src/cpu/
RUN CXX=g++ CC=gcc cmake -DUSE_OPENBLAS=1 ..
RUN make -j2
RUN ./tests

WORKDIR /src/autogtp/
RUN qmake
RUN make -j2