set PKG_FOLDER="%cd%\msvc\packages"
git submodule update --init --recursive
mkdir build
cd build
set BLAS_HOME="..\msvc\packages\OpenBLAS.0.2.14.1\lib\native"
for /F %%f in ("%features%") do set DEFINES=%DEFINES% -D%%f=1
cmake -G "Visual Studio 15 2017 Win64" %DEFINES% -DCMAKE_PREFIX_PATH="%QTDIR%/lib/cmake/" -DBOOST_ROOT="C:/Libraries/boost_1_65_1" -DBOOST_LIBRARYDIR="C:/Libraries/boost_1_65_1/lib64-msvc-14.1" -DBoost_USE_STATIC_LIBS=ON -DZLIB_ROOT="%PKG_FOLDER%/zlib-msvc14-x64.1.2.11.7795/build/native" -DZLIB_LIBRARY="%PKG_FOLDER%/zlib-msvc14-x64.1.2.11.7795/build/native/zlib-msvc14-x64.targets" -DOpenCL_LIBRARY="%PKG_FOLDER%/opencl-nug.0.777.12/build/native/opencl-nug.targets" -DOpenCL_INCLUDE_DIR="%PKG_FOLDER%/opencl-nug.0.777.12/build/native/include" -DBLAS_LIBRARIES="%PKG_FOLDER%/OpenBLAS.0.2.14.1/build/native/openblas.targets" -Dgtest_force_shared_crt=ON ..
cmake --build . --config Release -- /maxcpucount:1
