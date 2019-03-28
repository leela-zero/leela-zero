let
  pkgs = import <nixpkgs> {};
  stdenv = pkgs.stdenv;

in stdenv.mkDerivation (with pkgs; {
  name = "env";
  nativeBuildInputs = [
    cmake
  ];

  buildInputs = [
    zlib
    boost
    qt5.full
    opencl-headers
    ocl-icd
    python3
    python37Packages.tensorflow
    python37Packages.pymongo
    python37Packages.numpy
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DZLIB_INCLUDE_DIR=${zlib}/include"
    "-DZLIB_LIBRARY=${zlib}/lib/libz.so"
    "-DOpenCL_LIBRARY=${ocl-icd}/lib/libOpenCL.so"
    "-DOpenCL_INCLUDE_DIR=${opencl-headers}/include"
  ];

  BOOST_ROOT = "${boost}";
  LIBRARY_PATH = "${ocl-icd}/lib:${boost}/lib:${zlib}/lib:$LIBRARY_PATH";
})
