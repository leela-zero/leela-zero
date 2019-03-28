let
  pkgs = import <nixpkgs> {};
  stdenv = pkgs.stdenv;

in stdenv.mkDerivation {
  name = "env";
  nativeBuildInputs = [
    pkgs.cmake
  ];

  buildInputs = [
    pkgs.zlib
    pkgs.boost
    pkgs.qt5.full
    pkgs.opencl-headers
    pkgs.ocl-icd
    pkgs.python3
    pkgs.python37Packages.tensorflow
    pkgs.python37Packages.pymongo
    pkgs.python37Packages.numpy
  ];

  cmakeFlags = with stdenv; [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DZLIB_INCLUDE_DIR=${pkgs.zlib.dev}/include"
    "-DZLIB_LIBRARY=${pkgs.zlib}/lib/libz.so"
    "-DOpenCL_LIBRARY=${pkgs.ocl-icd}/lib/libOpenCL.so"
    "-DOpenCL_INCLUDE_DIR=${pkgs.opencl-headers}/include"
  ];

  PKG_CONFIG_PATH = "${pkgs.zlib}/lib/pkgconfig";
  BOOST_ROOT = "${pkgs.boost.dev}";
  BOOST_INCLUDEDIR = "${pkgs.boost.dev}/include";
  BOOST_LIBRARYDIR = "${pkgs.boost}/lib";
  LIBRARY_PATH = "${pkgs.ocl-icd}/lib:${pkgs.boost}/lib:${pkgs.zlib}/lib";
}
