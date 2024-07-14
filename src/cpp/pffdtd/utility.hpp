///////////////////////////////////////////////////////////////////////////////
// This file is a part of PFFDTD.
//
// PFFTD is released under the MIT License.
// For details see the LICENSE file.
//
// Copyright 2021 Brian Hamilton.
//
// File name: utility.hpp
//
// Description: Header-only misc function definitions not specific to FDTD
// simulation
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>

#ifndef DIV_CEIL
  #define DIV_CEIL(x, y) (((x) + (y)-1) / (y)) // this works for x≥0 and y>0
#endif
#define GET_BIT(var, pos) (((var) >> (pos)) & 1)
#define SET_BIT(var, pos) ((var) |= (1ULL << (pos)))
#define SET_BIT_VAL(var, pos, val)                                               \
  ((var) = ((var) & ~(1ULL << (pos))) | ((val) << (pos)))

void allocate_zeros(void** arr, uint64_t Nbytes);
void sort_keys(int64_t* val_arr, int64_t* key_arr, int64_t N);
