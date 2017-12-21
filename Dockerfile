FROM ubuntu:16.04

# Install
RUN apt-get -qq update
RUN apt-get install -y curl
RUN curl https://apt.llvm.org/llvm-snapshot.gpg.key|apt-key add -
RUN apt-get install -y clang-4.0 lldb-4.0 cmake
RUN apt-get install -y libboost-all-dev libopenblas-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev zlib1g-dev

RUN mkdir -p /src/
WORKDIR /src/

COPY . /src/
RUN CXX=/usr/bin/clang++-4.0 CC=/usr/bin/clang-4.0 cmake CMakeLists.txt
RUN make -j2 tests
RUN ./tests