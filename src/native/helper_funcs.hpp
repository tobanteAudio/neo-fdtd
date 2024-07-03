///////////////////////////////////////////////////////////////////////////////
// This file is a part of PFFDTD.
//
// PFFTD is released under the MIT License.
// For details see the LICENSE file.
//
// Copyright 2021 Brian Hamilton.
//
// File name: helper_funcs.h
//
// Description: Header-only misc function definitions not specific to FDTD simulation
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <cstring>

#include <sys/stat.h>

#ifndef DIV_CEIL
#define DIV_CEIL(x, y) (((x) + (y) - 1) / (y)) // this works for x≥0 and y>0
#endif
#define GET_BIT(var, pos) (((var) >> (pos)) & 1)
#define SET_BIT(var, pos) ((var) |= (1ULL << (pos)))
#define CLEAR_BIT(var, pos) ((var) &= ~(1ULL << (pos)))
#define SET_BIT_VAL(var, pos, val) ((var) = ((var) & ~(1ULL << (pos))) | ((val) << (pos)))

typedef enum
{
   FLOAT64,
   FLOAT32,
   INT64,
   INT8,
   BOOL,
} TYPE;

bool check_file_exists(char *filename);
void allocate_zeros(void **arr, uint64_t Nbytes);
int cmpfunc_int64_keys(const void *a, const void *b);
void sort_keys(int64_t *val_arr, int64_t *key_arr, int64_t N);
