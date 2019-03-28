let
  pkgs = import <nixpkgs> {};
  stdenv = pkgs.stdenv;

in stdenv.mkDerivation {
  name = "env";
  buildInputs = [
    pkgs.zlib.dev
    pkgs.boost.dev
    pkgs.qt5.full
    pkgs.opencl-headers
    pkgs.ocl-icd
  ];

  shellHook = ''
    export PKG_CONFIG_PATH=${pkgs.zlib}/lib/pkgconfig
    export BOOST_ROOT=${pkgs.boost.dev}
    export BOOST_INCLUDEDIR=${pkgs.boost.dev}/include
    export BOOST_LIBRARYDIR=${pkgs.boost}/lib
    export LIBRARY_PATH=${pkgs.ocl-icd}/lib:${pkgs.boost}/lib:${pkgs.zlib}/lib
    mkdir -p build/
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
      -DZLIB_INCLUDE_DIR=${pkgs.zlib.dev}/include \
      -DZLIB_LIBRARY=${pkgs.zlib}/lib/libz.so \
      -DOpenCL_LIBRARY=${pkgs.ocl-icd}/lib/libOpenCL.so \
      -DOpenCL_INCLUDE_DIR=${pkgs.opencl-headers}/include \
      .. 1>&2
  '';
}
