/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

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


#include "config.h"
#include "Network.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <boost/utility.hpp>
#include <boost/format.hpp>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
#ifdef USE_OPENCL
#include "OpenCL.h"
#include "UCTNode.h"
#endif

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GameState.h"
#include "GTP.h"
#include "Im2Col.h"
#include "NNCache.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Timing.h"
#include "Utils.h"

using namespace Utils;

// Input + residual block tower
static std::vector<std::vector<float>> conv_weights;
static std::vector<std::vector<float>> conv_biases;
static std::vector<std::vector<float>> batchnorm_means;
static std::vector<std::vector<float>> batchnorm_stddivs;

// Policy head
static std::vector<float> conv_pol_w;
static std::vector<float> conv_pol_b;
static std::array<float, 2> bn_pol_w1;
static std::array<float, 2> bn_pol_w2;

static std::array<float, 261364> ip_pol_w;
static std::array<float, 362> ip_pol_b;

// Value head
static std::vector<float> conv_val_w;
static std::vector<float> conv_val_b;
static std::array<float, 1> bn_val_w1;
static std::array<float, 1> bn_val_w2;

static std::array<float, 92416> ip1_val_w;
static std::array<float, 256> ip1_val_b;

static std::array<float, 256> ip2_val_w;
static std::array<float, 1> ip2_val_b;

const int rotate_nn_idx_table[8][361] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
            38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
            89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
            105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
            118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
            131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
            144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
            157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
            170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
            183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
            196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
            209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221,
            222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,
            235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
            248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
            261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273,
            274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286,
            287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299,
            300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312,
            313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325,
            326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338,
            339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
            352, 353, 354, 355, 356, 357, 358, 359, 360 }, 
    { 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356,
            357, 358, 359, 360, 323, 324, 325, 326, 327, 328, 329, 330, 331,
            332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 304, 305, 306,
            307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319,
            320, 321, 322, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294,
            295, 296, 297, 298, 299, 300, 301, 302, 303, 266, 267, 268, 269,
            270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282,
            283, 284, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257,
            258, 259, 260, 261, 262, 263, 264, 265, 228, 229, 230, 231, 232,
            233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
            246, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
            221, 222, 223, 224, 225, 226, 227, 190, 191, 192, 193, 194, 195,
            196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
            171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
            184, 185, 186, 187, 188, 189, 152, 153, 154, 155, 156, 157, 158,
            159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 133,
            134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
            147, 148, 149, 150, 151, 114, 115, 116, 117, 118, 119, 120, 121,
            122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, 113, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
            90, 91, 92, 93, 94, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
            69, 70, 71, 72, 73, 74, 75, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 19, 20, 21, 22, 23, 24, 25, 26,
            27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 }, 
    { 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 37, 36,
            35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,
            56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40,
            39, 38, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61,
            60, 59, 58, 57, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82,
            81, 80, 79, 78, 77, 76, 113, 112, 111, 110, 109, 108, 107, 106, 105,
            104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 132, 131, 130, 129,
            128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116,
            115, 114, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141,
            140, 139, 138, 137, 136, 135, 134, 133, 170, 169, 168, 167, 166,
            165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153,
            152, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178,
            177, 176, 175, 174, 173, 172, 171, 208, 207, 206, 205, 204, 203,
            202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190,
            227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216, 215,
            214, 213, 212, 211, 210, 209, 246, 245, 244, 243, 242, 241, 240,
            239, 238, 237, 236, 235, 234, 233, 232, 231, 230, 229, 228, 265,
            264, 263, 262, 261, 260, 259, 258, 257, 256, 255, 254, 253, 252,
            251, 250, 249, 248, 247, 284, 283, 282, 281, 280, 279, 278, 277,
            276, 275, 274, 273, 272, 271, 270, 269, 268, 267, 266, 303, 302,
            301, 300, 299, 298, 297, 296, 295, 294, 293, 292, 291, 290, 289,
            288, 287, 286, 285, 322, 321, 320, 319, 318, 317, 316, 315, 314,
            313, 312, 311, 310, 309, 308, 307, 306, 305, 304, 341, 340, 339,
            338, 337, 336, 335, 334, 333, 332, 331, 330, 329, 328, 327, 326,
            325, 324, 323, 360, 359, 358, 357, 356, 355, 354, 353, 352, 351,
            350, 349, 348, 347, 346, 345, 344, 343, 342 }, 
    { 360, 359, 358, 357, 356, 355, 354, 353, 352, 351, 350, 349, 348, 347, 346,
            345, 344, 343, 342, 341, 340, 339, 338, 337, 336, 335, 334, 333,
            332, 331, 330, 329, 328, 327, 326, 325, 324, 323, 322, 321, 320,
            319, 318, 317, 316, 315, 314, 313, 312, 311, 310, 309, 308, 307,
            306, 305, 304, 303, 302, 301, 300, 299, 298, 297, 296, 295, 294,
            293, 292, 291, 290, 289, 288, 287, 286, 285, 284, 283, 282, 281, 
            280, 279, 278, 277, 276, 275, 274, 273, 272, 271, 270, 269, 268,
            267, 266, 265, 264, 263, 262, 261, 260, 259, 258, 257, 256, 255,
            254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242,
            241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230, 229,
            228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216,
            215, 214, 213, 212, 211, 210, 209, 208, 207, 206, 205, 204, 203,
            202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190,
            189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177,
            176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 
            163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151,
            150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138,
            137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125,
            124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112,
            111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98,
            97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81,
            80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64,
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47,
            46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30,
            29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
            12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }, 
    { 0, 19, 38, 57, 76, 95, 114, 133, 152, 171, 190, 209, 228, 247, 266, 285,
            304, 323, 342, 1, 20, 39, 58, 77, 96, 115, 134, 153, 172, 191, 210,
            229, 248, 267, 286, 305, 324, 343, 2, 21, 40, 59, 78, 97, 116, 135,
            154, 173, 192, 211, 230, 249, 268, 287, 306, 325, 344, 3, 22, 41,
            60, 79, 98, 117, 136, 155, 174, 193, 212, 231, 250, 269, 288, 307,
            326, 345, 4, 23, 42, 61, 80, 99, 118, 137, 156, 175, 194, 213, 232,
            251, 270, 289, 308, 327, 346, 5, 24, 43, 62, 81, 100, 119, 138, 157,
            176, 195, 214, 233, 252, 271, 290, 309, 328, 347, 6, 25, 44, 63, 82,
            101, 120, 139, 158, 177, 196, 215, 234, 253, 272, 291, 310, 329,
            348, 7, 26, 45, 64, 83, 102, 121, 140, 159, 178, 197, 216, 235, 254,
            273, 292, 311, 330, 349, 8, 27, 46, 65, 84, 103, 122, 141, 160, 179,
            198, 217, 236, 255, 274, 293, 312, 331, 350, 9, 28, 47, 66, 85, 104,
            123, 142, 161, 180, 199, 218, 237, 256, 275, 294, 313, 332, 351, 10,
            29, 48, 67, 86, 105, 124, 143, 162, 181, 200, 219, 238, 257, 276,
            295, 314, 333, 352, 11, 30, 49, 68, 87, 106, 125, 144, 163, 182,
            201, 220, 239, 258, 277, 296, 315, 334, 353, 12, 31, 50, 69, 88,
            107, 126, 145, 164, 183, 202, 221, 240, 259, 278, 297, 316, 335,
            354, 13, 32, 51, 70, 89, 108, 127, 146, 165, 184, 203, 222, 241,
            260, 279, 298, 317, 336, 355, 14, 33, 52, 71, 90, 109, 128, 147,
            166, 185, 204, 223, 242, 261, 280, 299, 318, 337, 356, 15, 34, 53,
            72, 91, 110, 129, 148, 167, 186, 205, 224, 243, 262, 281, 300, 319,
            338, 357, 16, 35, 54, 73, 92, 111, 130, 149, 168, 187, 206, 225,
            244, 263, 282, 301, 320, 339, 358, 17, 36, 55, 74, 93, 112, 131,
            150, 169, 188, 207, 226, 245, 264, 283, 302, 321, 340, 359, 18, 37,
            56, 75, 94, 113, 132, 151, 170, 189, 208, 227, 246, 265, 284, 303,
            322, 341, 360 }, 
    { 342, 323, 304, 285, 266, 247, 228, 209, 190, 171, 152, 133, 114, 95, 76,
            57, 38, 19, 0, 343, 324, 305, 286, 267, 248, 229, 210, 191, 172,
            153, 134, 115, 96, 77, 58, 39, 20, 1, 344, 325, 306, 287, 268, 249,
            230, 211, 192, 173, 154, 135, 116, 97, 78, 59, 40, 21, 2, 345, 326,
            307, 288, 269, 250, 231, 212, 193, 174, 155, 136, 117, 98, 79, 60, 
            41, 22, 3, 346, 327, 308, 289, 270, 251, 232, 213, 194, 175, 156,
            137, 118, 99, 80, 61, 42, 23, 4, 347, 328, 309, 290, 271, 252, 233,
            214, 195, 176, 157, 138, 119, 100, 81, 62, 43, 24, 5, 348, 329, 310,
            291, 272, 253, 234, 215, 196, 177, 158, 139, 120, 101, 82, 63, 44,
            25, 6, 349, 330, 311, 292, 273, 254, 235, 216, 197, 178, 159, 140,
            121, 102, 83, 64, 45, 26, 7, 350, 331, 312, 293, 274, 255, 236, 217,
            198, 179, 160, 141, 122, 103, 84, 65, 46, 27, 8, 351, 332, 313, 294,
            275, 256, 237, 218, 199, 180, 161, 142, 123, 104, 85, 66, 47, 28, 9,
            352, 333, 314, 295, 276, 257, 238, 219, 200, 181, 162, 143, 124,
            105, 86, 67, 48, 29, 10, 353, 334, 315, 296, 277, 258, 239, 220,
            201, 182, 163, 144, 125, 106, 87, 68, 49, 30, 11, 354, 335, 316,
            297, 278, 259, 240, 221, 202, 183, 164, 145, 126, 107, 88, 69, 50,
            31, 12, 355, 336, 317, 298, 279, 260, 241, 222, 203, 184, 165, 146,
            127, 108, 89, 70, 51, 32, 13, 356, 337, 318, 299, 280, 261, 242,
            223, 204, 185, 166, 147, 128, 109, 90, 71, 52, 33, 14, 357, 338,
            319, 300, 281, 262, 243, 224, 205, 186, 167, 148, 129, 110, 91, 72,
            53, 34, 15, 358, 339, 320, 301, 282, 263, 244, 225, 206, 187, 168,
            149, 130, 111, 92, 73, 54, 35, 16, 359, 340, 321, 302, 283, 264,
            245, 226, 207, 188, 169, 150, 131, 112, 93, 74, 55, 36, 17, 360,
            341, 322, 303, 284, 265, 246, 227, 208, 189, 170, 151, 132, 113, 94,
            75, 56, 37, 18 }, 
    { 18, 37, 56, 75, 94, 113, 132, 151, 170, 189, 208, 227, 246, 265, 284, 303,
            322, 341, 360, 17, 36, 55, 74, 93, 112, 131, 150, 169, 188, 207,
            226, 245, 264, 283, 302, 321, 340, 359, 16, 35, 54, 73, 92, 111,
            130, 149, 168, 187, 206, 225, 244, 263, 282, 301, 320, 339, 358, 15,
            34, 53, 72, 91, 110, 129, 148, 167, 186, 205, 224, 243, 262, 281,
            300, 319, 338, 357, 14, 33, 52, 71, 90, 109, 128, 147, 166, 185,
            204, 223, 242, 261, 280, 299, 318, 337, 356, 13, 32, 51, 70, 89,
            108, 127, 146, 165, 184, 203, 222, 241, 260, 279, 298, 317, 336,
            355, 12, 31, 50, 69, 88, 107, 126, 145, 164, 183, 202, 221, 240,
            259, 278, 297, 316, 335, 354, 11, 30, 49, 68, 87, 106, 125, 144,
            163, 182, 201, 220, 239, 258, 277, 296, 315, 334, 353, 10, 29, 48,
            67, 86, 105, 124, 143, 162, 181, 200, 219, 238, 257, 276, 295, 314,
            333, 352, 9, 28, 47, 66, 85, 104, 123, 142, 161, 180, 199, 218, 237,
            256, 275, 294, 313, 332, 351, 8, 27, 46, 65, 84, 103, 122, 141, 160,
            179, 198, 217, 236, 255, 274, 293, 312, 331, 350, 7, 26, 45, 64, 83,
            102, 121, 140, 159, 178, 197, 216, 235, 254, 273, 292, 311, 330,
            349, 6, 25, 44, 63, 82, 101, 120, 139, 158, 177, 196, 215, 234, 253,
            272, 291, 310, 329, 348, 5, 24, 43, 62, 81, 100, 119, 138, 157, 176,
            195, 214, 233, 252, 271, 290, 309, 328, 347, 4, 23, 42, 61, 80, 99,
            118, 137, 156, 175, 194, 213, 232, 251, 270, 289, 308, 327, 346, 3,
            22, 41, 60, 79, 98, 117, 136, 155, 174, 193, 212, 231, 250, 269,
            288, 307, 326, 345, 2, 21, 40, 59, 78, 97, 116, 135, 154, 173, 192,
            211, 230, 249, 268, 287, 306, 325, 344, 1, 20, 39, 58, 77, 96, 115,
            134, 153, 172, 191, 210, 229, 248, 267, 286, 305, 324, 343, 0, 19,
            38, 57, 76, 95, 114, 133, 152, 171, 190, 209, 228, 247, 266, 285,
            304, 323, 342 }, 
    { 360, 341, 322, 303, 284, 265, 246, 227, 208, 189, 170, 151, 132, 113, 94,
            75, 56, 37, 18, 359, 340, 321, 302, 283, 264, 245, 226, 207, 188,
            169, 150, 131, 112, 93, 74, 55, 36, 17, 358, 339, 320, 301, 282,
            263, 244, 225, 206, 187, 168, 149, 130, 111, 92, 73, 54, 35, 16,
            357, 338, 319, 300, 281, 262, 243, 224, 205, 186, 167, 148, 129,
            110, 91, 72, 53, 34, 15, 356, 337, 318, 299, 280, 261, 242, 223,
            204, 185, 166, 147, 128, 109, 90, 71, 52, 33, 14, 355, 336, 317,
            298, 279, 260, 241, 222, 203, 184, 165, 146, 127, 108, 89, 70, 51,
            32, 13, 354, 335, 316, 297, 278, 259, 240, 221, 202, 183, 164, 145,
            126, 107, 88, 69, 50, 31, 12, 353, 334, 315, 296, 277, 258, 239,
            220, 201, 182, 163, 144, 125, 106, 87, 68, 49, 30, 11, 352, 333,
            314, 295, 276, 257, 238, 219, 200, 181, 162, 143, 124, 105, 86, 67,
            48, 29, 10, 351, 332, 313, 294, 275, 256, 237, 218, 199, 180, 161,
            142, 123, 104, 85, 66, 47, 28, 9, 350, 331, 312, 293, 274, 255, 236,
            217, 198, 179, 160, 141, 122, 103, 84, 65, 46, 27, 8, 349, 330, 311,
            292, 273, 254, 235, 216, 197, 178, 159, 140, 121, 102, 83, 64, 45, 
            26, 7, 348, 329, 310, 291, 272, 253, 234, 215, 196, 177, 158, 139, 
            120, 101, 82, 63, 44, 25, 6, 347, 328, 309, 290, 271, 252, 233, 214,
            195, 176, 157, 138, 119, 100, 81, 62, 43, 24, 5, 346, 327, 308, 289,
            270, 251, 232, 213, 194, 175, 156, 137, 118, 99, 80, 61, 42, 23, 4,
            345, 326, 307, 288, 269, 250, 231, 212, 193, 174, 155, 136, 117, 98,
            79, 60, 41, 22, 3, 344, 325, 306, 287, 268, 249, 230, 211, 192, 173,
            154, 135, 116, 97, 78, 59, 40, 21, 2, 343, 324, 305, 286, 267, 248,
            229, 210, 191, 172, 153, 134, 115, 96, 77, 58, 39, 20, 1, 342, 323,
            304, 285, 266, 247, 228, 209, 190, 171, 152, 133, 114, 95, 76, 57, 
            38, 19, 0 }
};

void Network::benchmark(const GameState * state, int iterations) {
    int cpus = cfg_num_threads;
    int iters_per_thread = (iterations + (cpus - 1)) / cpus;

    Time start;

    ThreadGroup tg(thread_pool);
    for (int i = 0; i < cpus; i++) {
        tg.add_task([iters_per_thread, state]() {
            for (int loop = 0; loop < iters_per_thread; loop++) {
                auto vec = get_scored_moves(state, Ensemble::RANDOM_ROTATION, -1, true);
            }
        });
    };
    tg.wait_all();

    Time end;
    auto elapsed = Time::timediff_seconds(start,end);
    myprintf("%5d evaluations in %5.2f seconds -> %d n/s\n",
             iterations, elapsed, (int)(iterations / elapsed));
}

void Network::process_bn_var(std::vector<float>& weights, const float epsilon) {
    for(auto&& w : weights) {
        w = 1.0f / std::sqrt(w + epsilon);
    }
}

void Network::initialize(void) {
#ifdef USE_OPENCL
    myprintf("Initializing OpenCL\n");
    opencl.initialize();
#endif
    // Count size of the network
    myprintf("Detecting residual layers...");
    std::ifstream wtfile(cfg_weightsfile);
    if (wtfile.fail()) {
        myprintf("Could not open weights file: %s\n", cfg_weightsfile.c_str());
        exit(EXIT_FAILURE);
    }
    std::string line;
    auto linecount = size_t{0};
    auto format_version = -1;
    while (std::getline(wtfile, line)) {
        std::stringstream iss(line);
        // First line is the file format version id
        if (linecount == 0) {
           iss >> format_version;
           if (iss.fail() || format_version != FORMAT_VERSION) {
               myprintf("Weights file is the wrong version.\n");
               exit(EXIT_FAILURE);
           } else {
               myprintf("v%d...", format_version);
           }
        }
        // Third line of parameters are the convolution layer biases,
        // so this tells us the amount of channels in the residual layers.
        // (Provided they're all equally large - that's not actually required!)
        if (linecount == 2) {
            auto count = std::distance(std::istream_iterator<std::string>(iss),
                                       std::istream_iterator<std::string>());
            myprintf("%d channels...", count);
        }
        linecount++;
    }
    // 1 format id, 1 input layer (4 x weights), 14 ending weights,
    // the rest are residuals, every residual has 8 x weight lines
    auto residual_blocks = linecount - (1 + 4 + 14);
    if (residual_blocks % 8 != 0) {
        myprintf("\nInconsistent number of weights in the file.\n");
        exit(EXIT_FAILURE);
    }
    residual_blocks /= 8;
    myprintf("%d blocks\n", residual_blocks);
#ifdef USE_OPENCL
    myprintf("Transferring weights to GPU...");
#endif
    // Re-read file and process
    wtfile.clear();
    wtfile.seekg(0, std::ios::beg);

    // Get the file format id out of the way
    std::getline(wtfile, line);

    auto plain_conv_layers = 1 + (residual_blocks * 2);
    auto plain_conv_wts = plain_conv_layers * 4;
    linecount = 0;
    while (std::getline(wtfile, line)) {
        std::vector<float> weights;
        float weight;
        std::istringstream iss(line);
        while (iss >> weight) {
            weights.emplace_back(weight);
        }
        if (linecount < plain_conv_wts) {
            if (linecount % 4 == 0) {
                conv_weights.emplace_back(weights);
            } else if (linecount % 4 == 1) {
                conv_biases.emplace_back(weights);
            } else if (linecount % 4 == 2) {
                batchnorm_means.emplace_back(weights);
            } else if (linecount % 4 == 3) {
                process_bn_var(weights);
                batchnorm_stddivs.emplace_back(weights);
            }
        } else if (linecount == plain_conv_wts) {
            conv_pol_w = std::move(weights);
        } else if (linecount == plain_conv_wts + 1) {
            conv_pol_b = std::move(weights);
        } else if (linecount == plain_conv_wts + 2) {
            std::copy(begin(weights), end(weights), begin(bn_pol_w1));
        } else if (linecount == plain_conv_wts + 3) {
            process_bn_var(weights);
            std::copy(begin(weights), end(weights), begin(bn_pol_w2));
        } else if (linecount == plain_conv_wts + 4) {
            std::copy(begin(weights), end(weights), begin(ip_pol_w));
        } else if (linecount == plain_conv_wts + 5) {
            std::copy(begin(weights), end(weights), begin(ip_pol_b));
        } else if (linecount == plain_conv_wts + 6) {
            conv_val_w = std::move(weights);
        } else if (linecount == plain_conv_wts + 7) {
            conv_val_b = std::move(weights);
        } else if (linecount == plain_conv_wts + 8) {
            std::copy(begin(weights), end(weights), begin(bn_val_w1));
        } else if (linecount == plain_conv_wts + 9) {
            process_bn_var(weights);
            std::copy(begin(weights), end(weights), begin(bn_val_w2));
        } else if (linecount == plain_conv_wts + 10) {
            std::copy(begin(weights), end(weights), begin(ip1_val_w));
        } else if (linecount == plain_conv_wts + 11) {
            std::copy(begin(weights), end(weights), begin(ip1_val_b));
        } else if (linecount == plain_conv_wts + 12) {
            std::copy(begin(weights), end(weights), begin(ip2_val_w));
        } else if (linecount == plain_conv_wts + 13) {
            std::copy(begin(weights), end(weights), begin(ip2_val_b));
        }
        linecount++;
    }
    wtfile.close();
#ifdef USE_OPENCL
    // input
    size_t weight_index = 0;
    opencl_net.push_convolve(3, conv_weights[weight_index],
                                conv_biases[weight_index]);
    opencl_net.push_batchnorm(361, batchnorm_means[weight_index],
                                   batchnorm_stddivs[weight_index]);
    weight_index++;

    // residual blocks
    for (auto i = size_t{0}; i < residual_blocks; i++) {
        opencl_net.push_residual(3, conv_weights[weight_index],
                                    conv_biases[weight_index],
                                    batchnorm_means[weight_index],
                                    batchnorm_stddivs[weight_index],
                                    conv_weights[weight_index + 1],
                                    conv_biases[weight_index + 1],
                                    batchnorm_means[weight_index + 1],
                                    batchnorm_stddivs[weight_index + 1]);
        weight_index += 2;
    }
    myprintf("done\n");
#endif
#ifdef USE_BLAS
#ifndef __APPLE__
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
    myprintf("BLAS Core: %s\n", openblas_get_corename());
#endif
#ifdef USE_MKL
    //mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);
    MKLVersion Version;
    mkl_get_version(&Version);
    myprintf("BLAS core: MKL %s\n", Version.Processor);
#endif
#endif
#endif
}

#ifdef USE_BLAS
template<unsigned int filter_size>
void convolve(size_t outputs,
              const std::vector<net_t>& input,
              const std::vector<float>& weights,
              const std::vector<float>& biases,
              std::vector<float>& output) {
    // fixed for 19x19
    constexpr unsigned int width = 19;
    constexpr unsigned int height = 19;
    constexpr unsigned int board_squares = width * height;
    constexpr unsigned int filter_len = filter_size * filter_size;
    const auto input_channels = weights.size() / (biases.size() * filter_len);
    const auto filter_dim = filter_len * input_channels;
    assert(outputs * board_squares == output.size());

    std::vector<float> col(filter_dim * width * height);
    im2col<filter_size>(input_channels, input, col);

    // Weight shape (output, input, filter_size, filter_size)
    // 96 22 3 3
    // outputs[96,19x19] = weights[96,22x3x3] x col[22x3x3,19x19]
    // C←αAB + βC
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimention of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                // M        N            K
                outputs, board_squares, filter_dim,
                1.0f, &weights[0], filter_dim,
                &col[0], board_squares,
                0.0f, &output[0], board_squares);

    for (unsigned int o = 0; o < outputs; o++) {
        for (unsigned int b = 0; b < board_squares; b++) {
            output[(o * board_squares) + b] =
                biases[o] + output[(o * board_squares) + b];
        }
    }
}

template<unsigned int inputs,
         unsigned int outputs,
         size_t W, size_t B>
void innerproduct(const std::vector<float>& input,
                  const std::array<float, W>& weights,
                  const std::array<float, B>& biases,
                  std::vector<float>& output) {
    assert(B == outputs);

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                // M     K
                outputs, inputs,
                1.0f, &weights[0], inputs,
                &input[0], 1,
                0.0f, &output[0], 1);

    auto lambda_ReLU = [](float val) { return (val > 0.0f) ?
                                       val : 0.0f; };

    for (unsigned int o = 0; o < outputs; o++) {
        float val = biases[o] + output[o];
        if (outputs == 256) {
            val = lambda_ReLU(val);
        }
        output[o] = val;
    }
}

template <size_t spatial_size>
void batchnorm(size_t channels,
               std::vector<float>& data,
               const float* means,
               const float* stddivs,
               const float* eltwise = nullptr)
{
    auto lambda_ReLU = [](float val) { return (val > 0.0f) ?
                                       val : 0.0f; };

    for (auto c = size_t{0}; c < channels; ++c) {
        auto mean = means[c];
        auto scale_stddiv = stddivs[c];

        if (eltwise == nullptr) {
            // Classical BN
            auto arr = &data[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU(scale_stddiv * (arr[b] - mean));
            }
        } else {
            // BN + residual add
            auto arr = &data[c * spatial_size];
            auto res = &eltwise[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU(res[b] +
                                     (scale_stddiv * (arr[b] - mean)));
            }
        }
    }
}

#ifndef USE_HALF
void Network::forward_cpu(std::vector<float>& input,
                          std::vector<float>& output) {
    // Input convolution
    constexpr int width = 19;
    constexpr int height = 19;
    // Calculate output channels
    const auto output_channels = conv_biases[0].size();
    auto conv_out = std::vector<float>(output_channels * width * height);

    convolve<3>(output_channels, input,
                conv_weights[0], conv_biases[0], conv_out);
    batchnorm<361>(output_channels, conv_out,
                   batchnorm_means[0].data(),
                   batchnorm_stddivs[0].data());

    // Residual tower
    auto conv_in = std::vector<float>(output_channels * width * height);
    auto res = std::vector<float>(output_channels * width * height);
    for (auto i = size_t{1}; i < conv_weights.size(); i += 2) {
        auto output_channels = conv_biases[i].size();
        std::swap(conv_out, conv_in);
        std::copy(begin(conv_in), end(conv_in), begin(res));
        convolve<3>(output_channels, conv_in,
                    conv_weights[i], conv_biases[i],
                    conv_out);
        batchnorm<361>(output_channels, conv_out,
                       batchnorm_means[i].data(),
                       batchnorm_stddivs[i].data());

        output_channels = conv_biases[i + 1].size();
        std::swap(conv_out, conv_in);
        convolve<3>(output_channels, conv_in,
                    conv_weights[i + 1], conv_biases[i + 1],
                    conv_out);
        batchnorm<361>(output_channels, conv_out,
                       batchnorm_means[i + 1].data(),
                       batchnorm_stddivs[i + 1].data(),
                       res.data());
    }
    std::copy(begin(conv_out), end(conv_out), begin(output));
}
#endif

template<typename T>
T relative_difference(T a, T b) {
    // Handle NaN
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<T>::max();
    }
    // Handle sign difference
    if (((a < 0) != (b < 0)) && (a != 0) && (b != 0)) {
        return std::numeric_limits<T>::max();
    }
    a = std::fabs(a);
    b = std::fabs(b);

    // Handle underflow
    constexpr float small_number = 1e-3f;
    a = std::max(a, small_number);
    b = std::max(b, small_number);

    return std::max(fabs((a - b) / a), fabs((a - b) / b));
}

void compare_net_outputs(std::vector<float>& data,
                         std::vector<float>& ref) {
    // We accept an error up to 5%, but output values
    // smaller than 1/1000th are "rounded up" for the comparison.
    constexpr float relative_error = 5e-2f;
    for (auto idx = size_t{0}; idx < data.size(); ++idx) {
        auto err = relative_difference(data[idx], ref[idx]);
        if (err > relative_error) {
            myprintf("Error in OpenCL calculation: expected %f got %f "
                     "(error=%f%%)\n", ref[idx], data[idx], err * 100.0);
            myprintf("Update your GPU drivers or reduce the amount of games "
                     "played simultaneously.\n");
            throw std::runtime_error("OpenCL self-check mismatch.");
        }
    }
}
#endif

void Network::softmax(const std::vector<float>& input,
                      std::vector<float>& output,
                      float temperature) {
    assert(&input != &output);

    auto alpha = *std::max_element(begin(input),
                                   begin(input) + output.size());
    alpha /= temperature;

    auto denom = 0.0f;
    auto helper = std::vector<float>(output.size());
    for (auto i = size_t{0}; i < output.size(); i++) {
        auto val   = std::exp((input[i]/temperature) - alpha);
        helper[i]  = val;
        denom     += val;
    }
    for (auto i = size_t{0}; i < output.size(); i++) {
        output[i] = helper[i] / denom;
    }
}

Network::Netresult Network::get_scored_moves(
    const GameState* state, Ensemble ensemble, int rotation, bool skip_cache) {
    Netresult result;
    if (state->board.get_boardsize() != 19) {
        return result;
    }

    NNPlanes planes;
    gather_features(state, planes);

    // See if we already have this in the cache.
    if (!skip_cache) {
      if (NNCache::get_NNCache().lookup(planes, result)) {
        return result;
      }
    }

    if (ensemble == DIRECT) {
        assert(rotation >= 0 && rotation <= 7);
        result = get_scored_moves_internal(state, planes, rotation);
    } else {
        assert(ensemble == RANDOM_ROTATION);
        assert(rotation == -1);
        auto rand_rot = Random::get_Rng().randfix<8>();
        result = get_scored_moves_internal(state, planes, rand_rot);
    }

    // Insert result into cache.
    NNCache::get_NNCache().insert(planes, result);

    return result;
}

Network::Netresult Network::get_scored_moves_internal(
    const GameState* state, NNPlanes & planes, int rotation) {
    assert(rotation >= 0 && rotation <= 7);
    assert(INPUT_CHANNELS == planes.size());
    constexpr int width = 19;
    constexpr int height = 19;
    const auto convolve_channels = conv_pol_w.size() / conv_pol_b.size();
    std::vector<net_t> input_data;
    std::vector<net_t> output_data(convolve_channels * width * height);
    std::vector<float> policy_data(2 * width * height);
    std::vector<float> value_data(1 * width * height);
    std::vector<float> policy_out((width * height) + 1);
    std::vector<float> softmax_data((width * height) + 1);
    std::vector<float> winrate_data(256);
    std::vector<float> winrate_out(1);
    // Data layout is input_data[(c * height + h) * width + w]
    input_data.reserve(INPUT_CHANNELS * width * height);
    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                auto rot_idx = rotate_nn_idx_table[rotation][h * 19 + w];
                input_data.emplace_back(net_t(planes[c][rot_idx]));
            }
        }
    }
#ifdef USE_OPENCL
    opencl_net.forward(input_data, output_data);
#elif defined(USE_BLAS) && !defined(USE_OPENCL)
    forward_cpu(input_data, output_data);
#endif
#ifdef USE_OPENCL_SELFCHECK
    // Both implementations are available, self-check the OpenCL driver by
    // running both with a probability of 1/2000.
    if (Random::get_Rng().randfix<SELFCHECK_PROBABILITY>() == 0) {
        auto cpu_output_data = std::vector<float>(output_data.size());
        forward_cpu(input_data, cpu_output_data);
        compare_net_outputs(output_data, cpu_output_data);
    }
#endif
    // We calculate both network heads on the CPU. They are irregular
    // and have a much lower compute densitity than the residual layers,
    // which means they don't get much - if any - speedup from being on the
    // GPU. See issue #185.

    // Get the moves
    convolve<1>(2, output_data, conv_pol_w, conv_pol_b, policy_data);
    batchnorm<361>(2, policy_data, bn_pol_w1.data(), bn_pol_w2.data());
    innerproduct<2*361, 362>(policy_data, ip_pol_w, ip_pol_b, policy_out);
    softmax(policy_out, softmax_data, cfg_softmax_temp);
    std::vector<float>& outputs = softmax_data;

    // Now get the score
    convolve<1>(1, output_data, conv_val_w, conv_val_b, value_data);
    batchnorm<361>(1, value_data, bn_val_w1.data(), bn_val_w2.data());
    innerproduct<361, 256>(value_data, ip1_val_w, ip1_val_b, winrate_data);
    innerproduct<256, 1>(winrate_data, ip2_val_w, ip2_val_b, winrate_out);

    // Sigmoid
    float winrate_sig = (1.0f + std::tanh(winrate_out[0])) / 2.0f;

    std::vector<scored_node> result;
    for (auto idx = size_t{0}; idx < outputs.size(); idx++) {
        if (idx < 19*19) {
            auto val = outputs[idx];
            auto rot_idx = rotate_nn_idx_table[rotation][idx];
            int x = rot_idx % 19;
            int y = rot_idx / 19;
            int rot_vtx = state->board.get_vertex(x, y);
            if (state->board.get_square(rot_vtx) == FastBoard::EMPTY) {
                result.emplace_back(val, rot_vtx);
            }
        } else {
            result.emplace_back(outputs[idx], FastBoard::PASS);
        }
    }

    return std::make_pair(result, winrate_sig);
}

void Network::show_heatmap(const FastState * state, Netresult& result, bool topmoves) {
    auto moves = result.first;
    std::vector<std::string> display_map;
    std::string line;

    for (unsigned int y = 0; y < 19; y++) {
        for (unsigned int x = 0; x < 19; x++) {
            int vtx = state->board.get_vertex(x, y);

            auto item = std::find_if(moves.cbegin(), moves.cend(),
                [&vtx](scored_node const& test_item) {
                return test_item.second == vtx;
            });

            float score = 0.0f;
            // Non-empty squares won't be scored
            if (item != moves.end()) {
                score = item->first;
                assert(vtx == item->second);
            }

            line += boost::str(boost::format("%3d ") % int(score * 1000));
            if (x == 18) {
                display_map.push_back(line);
                line.clear();
            }
        }
    }

    for (int i = display_map.size() - 1; i >= 0; --i) {
        myprintf("%s\n", display_map[i].c_str());
    }
    assert(result.first.back().second == FastBoard::PASS);
    int pass_score = int(result.first.back().first * 1000);
    myprintf("pass: %d\n", pass_score);
    myprintf("winrate: %f\n", result.second);

    if (topmoves) {
        std::stable_sort(moves.rbegin(), moves.rend());

        float cum = 0.0f;
        size_t tried = 0;
        while (cum < 0.85f && tried < moves.size()) {
            if (moves[tried].first < 0.01f) break;
            myprintf("%1.3f (%s)\n",
                    moves[tried].first,
                    state->board.move_to_text(moves[tried].second).c_str());
            cum += moves[tried].first;
            tried++;
        }
    }
}

void Network::fill_input_plane_pair(const FullBoard& board,
                                    BoardPlane& black, BoardPlane& white) {
    auto idx = 0;
    for (int j = 0; j < 19; j++) {
        for(int i = 0; i < 19; i++) {
            int vtx = board.get_vertex(i, j);
            auto color = board.get_square(vtx);
            if (color != FastBoard::EMPTY) {
                if (color == FastBoard::BLACK) {
                    black[idx] = true;
                } else {
                    white[idx] = true;
                }
            }
            idx++;
        }
    }
}

void Network::gather_features(const GameState* state, NNPlanes & planes) {
    planes.resize(INPUT_CHANNELS);
    BoardPlane& black_to_move  = planes[2 * INPUT_MOVES];
    BoardPlane& white_to_move  = planes[2 * INPUT_MOVES + 1];

    const auto to_move = state->get_to_move();
    const auto blacks_move = to_move == FastBoard::BLACK;

    const auto black_offset = blacks_move ? 0 : INPUT_MOVES;
    const auto white_offset = blacks_move ? INPUT_MOVES : 0;

    if (blacks_move) {
        black_to_move.set();
    } else {
        white_to_move.set();
    }

    const auto moves = std::min<size_t>(state->get_movenum() + 1, INPUT_MOVES);
    // Go back in time, fill history boards
    for (auto h = size_t{0}; h < moves; h++) {
        // collect white, black occupation planes
        fill_input_plane_pair(state->get_past_board(h),
                              planes[black_offset + h],
                              planes[white_offset + h]);
    }
}

