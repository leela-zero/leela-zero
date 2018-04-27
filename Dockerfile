FROM ubuntu:16.04

# Install
RUN apt-get -qq update
RUN apt-get install -y cmake g++
RUN apt-get install -y libboost-all-dev libopenblas-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev zlib1g-dev
RUN apt-get install -y qt5-default qt5-qmake

RUN mkdir -p /src/gpu/
RUN mkdir -p /src/gpu-half/
RUN mkdir -p /src/cpu/
COPY . /src/

# GPU build
WORKDIR /src/gpu/
RUN CXX=g++ CC=gcc cmake ..
RUN cmake --build . --target leelaz --config Release -- -j2

# GPU 16-bit (memory storage) build
WORKDIR /src/gpu-half/
RUN CXX=g++ CC=gcc cmake -DUSE_HALF=1 ..
RUN cmake --build . --target leelaz --config Release -- -j2

# CPU build
WORKDIR /src/cpu/
RUN CXX=g++ CC=gcc cmake -DUSE_CPU_ONLY=1 ..
RUN cmake --build . --config Release -- -j2
RUN ./tests
