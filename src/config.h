/*
    This file is part of Leela Zero.

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
*/

#ifndef CONFIG_INCLUDED
#define CONFIG_INCLUDED

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
 * Features
 *
 * USE_BLAS: Use a basic linear algebra library.
 * We currently require this, as not all operations are performed on
 * the GPU - some operations won't get any speedup from it.
 * Also used for OpenCL self-checks.
 */
#define USE_BLAS
/*
 * We use OpenBLAS by default, except on macOS, which has a fast BLAS
 * built-in. (Accelerate)
 */
#if !defined(__APPLE__) && !defined(__MACOSX)
#define USE_OPENBLAS
#endif
/*
 * USE_MKL: Optionally allows using Intel Math Kernel library as
 * BLAS implementation. Note that MKL's license is not compatible with the GPL,
 * so do not redistribute the resulting binaries. It is fine to use it on your
 * own system.
 */
//# define USE_MKL
/*
 * USE_OPENCL: Use OpenCL acceleration for GPUs. This makes the program a lot
 * faster if you have a recent GPU. Don't use it on CPUs even if they have
 * OpenCL drivers - the BLAS version is much faster for those.
 */
#define USE_OPENCL
/*
 * USE_HALF: Use 16-bit floating point storage for network parameters.
 * Only works for OpenCL implementations. Gives a slight speedup on some
 * cards at the cost of some accuracy.
 */
// #define USE_HALF
/*
 * USE_TUNER: Expose some extra command line parameters that allow tuning the
 * search algorithm.
 */
// #define USE_TUNER
/*
 * SGEMM_TUNERS: CLBlast xgemm_direct OpenCL kernel tuner values.
 * To tune SGEMM for your GPU download CLBlast, compile it with tuners enabled
 * and run "./clblast_tuner_xgemm_direct -m 128 -n 128 -k 128 -alpha 1 -beta 0".
 * After tuning is done copy the found parameters here.
 */
#define SGEMM_TUNERS "KWID=16 MDIMAD=8 MDIMCD=8 NDIMBD=16 NDIMCD=16 PADA=1 PADB=1 PRECISION=32 VWMD=1 VWND=1 WGD=32"

#define PROGRAM_NAME "Leela Zero"
#define PROGRAM_VERSION "0.9"

/*
 * OpenBLAS limitation: the default configuration on some Linuxes
 * is limited to 64 cores.
 */
#if defined(USE_BLAS) && defined(USE_OPENBLAS)
#define MAX_CPUS 64
#else
#define MAX_CPUS 128
#endif

#ifdef USE_HALF
#ifndef USE_OPENCL
#error "Half-precision not supported without OpenCL"
#endif
#include "half/half.hpp"
using net_t = half_float::half;
#else
using net_t = float;
#endif

#if defined(USE_BLAS) && defined(USE_OPENCL) && !defined(USE_HALF)
// If both BLAS and OpenCL are fully usable, then check the OpenCL
// results against BLAS with some probability.
#define USE_OPENCL_SELFCHECK
#define SELFCHECK_PROBABILITY 2000
#endif

#if (_MSC_VER >= 1400) /* VC8+ Disable all deprecation warnings */
    #pragma warning(disable : 4996)
#endif /* VC8+ */

#endif
