/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef CONFIG_H_INCLUDED
#define CONFIG_H_INCLUDED

/*
 * We need to check for input while we are thinking.
 * That code isn't portable, so select something appropriate for the system.
 */
#ifdef _WIN32
#undef HAVE_SELECT
#define NOMINMAX
#else
#define HAVE_SELECT
#endif

/*
 * BOARD_SIZE: Define size of the board to compile Leela with, must be an odd
 * number due to winograd tiles
 */
static constexpr auto BOARD_SIZE = 19;
static_assert(BOARD_SIZE % 2 == 1,
              "Code assumes odd board size, remove at your own risk!");

static constexpr auto NUM_INTERSECTIONS = BOARD_SIZE * BOARD_SIZE;
static constexpr auto POTENTIAL_MOVES = NUM_INTERSECTIONS + 1; // including pass

/*
 * TRAINED_UNIT_KOMI: Define the komi used during training that represents a
 * value of 1.0 for the side to move NN inputs
 */
static constexpr auto TRAINED_UNIT_KOMI = 7.5f;

/*
 * Features
 *
 * USE_BLAS: Optionally use a basic linear algebra library.
 * This is may perform faster than the included Eigen library,
 * and some BLAS libraries can target multiple CPU models.
 * Not all operations are performed on the GPU -
 * some operations won't get any speedup from it.
 * Also used for OpenCL self-checks.
 */
//#define USE_BLAS

/*
 * We use OpenBLAS by default, except on macOS, which has a fast BLAS
 * built-in. (Accelerate)
 */
#if !defined(__APPLE__) && !defined(__MACOSX)
#if defined(USE_BLAS)
#define USE_OPENBLAS
#endif
#endif

/*
 * USE_MKL: Optionally allows using Intel Math Kernel library as
 * BLAS implementation. Note that MKL's license is not compatible with the GPL,
 * so do not redistribute the resulting binaries. It is fine to use it on your
 * own system.
 */
//#define USE_MKL
/*
 * USE_OPENCL: Use OpenCL acceleration for GPUs. This makes the program a lot
 * faster if you have a recent GPU. Don't use it on CPUs even if they have
 * OpenCL drivers - the BLAS version is much faster for those.
 */
#ifndef USE_CPU_ONLY
#define USE_OPENCL

/*
 * USE_HALF: Include the half-precision OpenCL implementation when building.
 * The current implementation autodetects whether half-precision is better
 * or single-precision is better (half precision is chosen if it's 5% faster)
 * Half-precision OpenCL gains performance on some GPUs while losing some
 * accuracy on the calculation, but generally it is worth using half precision
 * if it is at least 5% faster.
 */
#define USE_HALF

#endif

/*
 * USE_TUNER: Expose some extra command line parameters that allow tuning the
 * search algorithm.
 */
//#define USE_TUNER

static constexpr auto PROGRAM_NAME = "Leela Zero";
static constexpr auto PROGRAM_VERSION = "0.17";

/*
 * OpenBLAS limitation: the default configuration on some Linuxes
 * is limited to 64 cores.
 */
#if defined(USE_BLAS) && defined(USE_OPENBLAS)
static constexpr auto MAX_CPUS = 64;
#else
static constexpr auto MAX_CPUS = 256;
#endif

#ifdef USE_HALF
#include "half/half.hpp"
#endif

#ifdef USE_OPENCL
// If OpenCL are fully usable, then check the OpenCL against CPU
// implementation with some probability.
#define USE_OPENCL_SELFCHECK
static constexpr auto SELFCHECK_PROBABILITY = 2000;
#endif

#if (_MSC_VER >= 1400) /* VC8+ Disable all deprecation warnings */
    #pragma warning(disable : 4996)
#endif /* VC8+ */

#endif
