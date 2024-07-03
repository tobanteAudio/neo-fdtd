// vim: tabstop=3: ai
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

#include <stdio.h>
#include <stdlib.h> //for malloc
#include <stdint.h> 
#include <assert.h> //for assert
#include <stdbool.h> //for bool
#include <sys/stat.h> //for stat
#include <string.h> //for memset

//some useful macros
#ifndef MIN
   #define MIN(a,b) (((a)<(b))?(a):(b))
#endif
#ifndef MAX
   #define MAX(a,b) (((a)>(b))?(a):(b))
#endif
#ifndef CLAMP
   #define CLAMP(a, min, max) ( MIN(max, MAX(a, min)) )
#endif
#ifndef SWAP
   #define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))
#endif
#ifndef DIV_CEIL
   #define DIV_CEIL(x,y) (((x) + (y) - 1) / (y)) //this works for x≥0 and y>0
#endif
#define GET_BIT(var,pos) (((var)>>(pos)) & 1)
#define SET_BIT(var,pos) ((var) |= (1ULL<<(pos)))
#define CLEAR_BIT(var,pos) ((var) &= ~(1ULL<<(pos)))
#define SET_BIT_VAL(var,pos,val) ((var) = ((var) & ~(1ULL << (pos))) | ((val) << (pos)))

typedef enum {
   FLOAT64,
   FLOAT32,
   INT64,
   INT8,
   BOOL,
} TYPE;


bool check_file_exists(char *filename);
void mymalloc(void **arr, uint64_t Nbytes);
int cmpfunc_int64 (const void * a, const void * b);
int cmpfunc_int64_keys (const void * a, const void * b);
void qsort_keys(int64_t *val_arr, int64_t *key_arr, int64_t N);


